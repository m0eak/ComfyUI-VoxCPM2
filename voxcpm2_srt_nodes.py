import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, List

import folder_paths
import torch
import torchaudio
from comfy_api.latest import io
import comfy.model_management as model_management

from .modules.model_info import AVAILABLE_VOXCPM_MODELS
from .modules.srt_audio_utils import get_audio_duration_seconds, save_numpy_audio
from .modules.srt_manifest import load_completed_manifest, upsert_manifest_item, write_json, write_manifest, write_progress
from .modules.srt_parser import build_preview_json, build_preview_text, parse_srt_file, sanitize_job_name, segments_to_payload
from .modules.srt_timeline import build_premiere_xml
from .voxcpm2_nodes import (
    MAX_REFERENCE_AUDIO_SECONDS,
    _get_audio_duration_seconds,
    _load_patcher,
    _normalize_loudness,
    _save_audio_to_temp,
    _validate_reference_audio_duration,
    get_available_devices,
    offload_asr,
    set_seed,
    transcribe_audio,
)


DEFAULT_CONSISTENCY_PROMPT = "保持同一个说话人的音色、音量、语速和语气稳定一致，使用自然连贯的旁白风格。"


def _available_model_names() -> list[str]:
    names = list(AVAILABLE_VOXCPM_MODELS.keys())
    return names or ["VoxCPM2"]


def _available_loras() -> list[str]:
    return ["None"] + folder_paths.get_filename_list("loras")


def _available_srt_files() -> list[str]:
    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)
    files: list[str] = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.lower().endswith(".srt"):
                full_path = Path(root) / filename
                files.append(os.path.relpath(full_path, input_dir).replace("\\", "/"))
    return sorted(files)


def _resolve_srt_path(srt_file: str, srt_path: str) -> str:
    optional_file = str(srt_file or "").strip()
    if optional_file and optional_file != "None" and optional_file.lower().endswith(".srt"):
        try:
            return folder_paths.get_annotated_filepath(optional_file)
        except Exception:
            return optional_file
    if srt_path and str(srt_path).strip():
        return str(srt_path).strip()
    raise ValueError("Please provide an SRT file path.")


def _resolve_job_dir(output_dir: str, job_name: str) -> Path:
    if output_dir and output_dir.strip():
        base_dir = Path(output_dir.strip())
    else:
        base_dir = Path(folder_paths.get_output_directory()) / "voxcpm2_srt"
    return base_dir / sanitize_job_name(job_name)


def _format_output_name(template: str, segment: dict[str, Any], used_names: set[str]) -> str:
    raw_template = (template or "{index:04d}.wav").strip() or "{index:04d}.wav"
    values = {
        "index": int(segment["index"]),
        "start": str(segment.get("start", "")).replace(":", "-").replace(",", "."),
        "end": str(segment.get("end", "")).replace(":", "-").replace(",", "."),
    }
    try:
        candidate = raw_template.format(**values)
    except Exception:
        candidate = f"{values['index']:04d}.wav"

    if not candidate.lower().endswith(".wav"):
        candidate += ".wav"

    candidate = sanitize_job_name(candidate[:-4]) + ".wav"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    stem = candidate[:-4]
    suffix = 2
    while True:
        deduped = f"{stem}_dup{suffix}.wav"
        if deduped not in used_names:
            used_names.add(deduped)
            return deduped
        suffix += 1


def _resolve_seed(base_seed: int, strategy: str, segment: dict[str, Any]) -> int:
    if int(base_seed) < 0:
        import random
        return random.randint(0, 2**31 - 1)

    strategy = strategy or "increment_by_index"
    if strategy == "fixed":
        return int(base_seed)
    if strategy == "random":
        import random
        return random.randint(0, 2**31 - 1)
    if strategy == "hash_text":
        digest = hashlib.sha256(str(segment.get("text", "")).encode("utf-8")).hexdigest()
        return (int(base_seed) + int(digest[:8], 16)) & 0xFFFFFFFF
    return int(base_seed) + int(segment["index"])


def _build_segment_text(text: str, voice_description: str, use_consistency_prompt: bool, consistency_prompt: str) -> str:
    desc_parts: list[str] = []
    if voice_description and voice_description.strip():
        desc_parts.append(voice_description.strip())
    if use_consistency_prompt and consistency_prompt and consistency_prompt.strip():
        desc_parts.append(consistency_prompt.strip())
    if desc_parts:
        return f"({'；'.join(desc_parts)}){text.strip()}"
    return text.strip()


def _success_counts(manifest: list[dict[str, Any]]) -> tuple[int, int]:
    success = sum(1 for item in manifest if item.get("status") == "ok")
    failed = sum(1 for item in manifest if item.get("status") == "error")
    return success, failed


class VoxCPM2SRTParserNode(io.ComfyNode):
    CATEGORY = "audio/tts/srt"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VoxCPM2_SRT_Parser",
            display_name="VoxCPM2 SRT Parser",
            category=cls.CATEGORY,
            description="Parse an SRT file and output structured subtitle segments with preview.",
            inputs=[
                io.String.Input("srt_path", default="", tooltip="Local path to the SRT file."),
                io.Combo.Input("encoding", options=["auto", "utf-8-sig", "utf-8", "gbk"], default="auto", tooltip="Subtitle encoding. Use gbk if Chinese subtitles are garbled."),
                io.Boolean.Input("skip_empty", default=True, tooltip="Skip subtitle entries with empty text."),
                io.Boolean.Input("normalize_whitespace", default=True, tooltip="Normalize extra whitespace for TTS."),
                io.Boolean.Input("strip_tags", default=True, tooltip="Remove simple SRT/HTML tags."),
                io.Int.Input("preview_limit", default=30, min=0, max=500, tooltip="Number of subtitle entries to preview."),
                io.String.Input("srt_file", default="", tooltip="Optional uploaded/selected SRT path. Leave empty to use srt_path."),
            ],
            outputs=[
                io.AnyType.Output(display_name="SRT Segments"),
                io.String.Output(display_name="Preview Text"),
                io.String.Output(display_name="Preview JSON"),
                io.Int.Output(display_name="Segment Count"),
                io.String.Output(display_name="Resolved SRT Path"),
            ],
        )

    @classmethod
    def execute(cls, srt_path, encoding, skip_empty, normalize_whitespace, strip_tags, preview_limit, srt_file=""):
        resolved_srt_path = _resolve_srt_path(srt_file, srt_path)
        segments = parse_srt_file(
            resolved_srt_path,
            encoding=encoding,
            skip_empty=skip_empty,
            normalize_whitespace=normalize_whitespace,
            strip_tags=strip_tags,
        )
        payload = segments_to_payload(segments, resolved_srt_path)
        preview_text = build_preview_text(segments, preview_limit=preview_limit)
        preview_json = build_preview_json(segments, preview_limit=preview_limit)
        return io.NodeOutput(payload, preview_text, preview_json, len(segments), resolved_srt_path)


class VoxCPM2SRTBatchTTSNode(io.ComfyNode):
    CATEGORY = "audio/tts/srt"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = _available_model_names()
        devices = get_available_devices()
        default_device = devices[0]
        return io.Schema(
            node_id="VoxCPM2_SRT_Batch_TTS",
            display_name="VoxCPM2 SRT Batch TTS",
            category=cls.CATEGORY,
            description="Generate one WAV file per SRT subtitle segment using VoxCPM2.",
            inputs=[
                io.AnyType.Input("segments", tooltip="SRT segments from VoxCPM2 SRT Parser."),
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="Select the VoxCPM model to use."),
                io.Combo.Input("lora_name", options=_available_loras(), default="None", tooltip="LoRA checkpoint from models/loras."),
                io.String.Input("voice_description", multiline=True, default="", tooltip="Voice/style description prepended to each segment."),
                io.String.Input("prompt_text", multiline=True, default="", tooltip="Reference transcript for Ultimate Cloning. For SRT batches, controllable clone mode is usually safer."),
                io.Audio.Input("reference_audio", optional=True, tooltip="Reference audio for voice cloning. Leave unconnected for text-to-speech."),
                io.Boolean.Input("enable_asr", default=False, label_on="ASR", label_off="Off", tooltip="Auto-transcribe reference audio. Used only with clone_mode ultimate/auto."),
                io.Boolean.Input("enable_denoiser", default=False, label_on="Denoise", label_off="Off", tooltip="Denoise reference audio before cloning."),
                io.Boolean.Input("use_consistency_prompt", default=True, tooltip="Use a consistency hint across segments."),
                io.String.Input("consistency_prompt", multiline=True, default=DEFAULT_CONSISTENCY_PROMPT, tooltip="Short prompt to keep segment style consistent."),
                io.String.Input("output_dir", default="", tooltip="Base output directory. Empty uses ComfyUI/output/voxcpm2_srt."),
                io.String.Input("job_name", default="", tooltip="Job folder name. Empty uses timestamp."),
                io.String.Input("filename_template", default="{index:04d}.wav", tooltip="Filename template for each segment."),
                io.Boolean.Input("resume", default=True, tooltip="Resume completed segments from manifest.json."),
                io.Boolean.Input("overwrite", default=False, tooltip="Overwrite existing WAV files."),
                io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Base random seed. Use a fixed number for more stable SRT batches."),
                io.Combo.Input("seed_strategy", options=["fixed", "increment_by_index", "random", "hash_text"], default="fixed", tooltip="Per-segment seed strategy. fixed is usually best for SRT timbre consistency."),
                io.Float.Input("cfg_value", default=2.2, min=1.0, max=10.0, step=0.1, tooltip="Classifier-Free Guidance scale."),
                io.Int.Input("inference_timesteps", default=15, min=1, max=100, step=1, tooltip="Diffusion inference steps."),
                io.Int.Input("max_tokens", default=4096, min=64, max=8192, tooltip="Maximum generation length."),
                io.Boolean.Input("normalize_text", default=True, label_on="Normalize", label_off="Raw", tooltip="Normalize text for natural language input."),
                io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="Maximum retry attempts for bad generations."),
                io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="Bad generation detection threshold."),
                io.Boolean.Input("force_offload", default=False, label_on="Force Offload", label_off="Auto", tooltip="Force offload model after generation."),
                io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto", tooltip="Model dtype."),
                io.Combo.Input("device", options=devices, default=default_device, tooltip="Inference device."),
                io.Boolean.Input("torch_compile", default=False, label_on="Torch Compile", label_off="Standard", tooltip="Enable torch.compile."),
                io.String.Input("clone_mode", default="controllable", tooltip="Clone mode: controllable, ultimate, or auto. controllable is recommended for SRT batches."),
                io.Boolean.Input("export_premiere_xml", default=True, label_on="Export XML", label_off="No XML", tooltip="Export Premiere-compatible timeline XML."),
                io.Int.Input("timeline_fps", default=30, min=0, max=120, step=1, tooltip="Timeline frame rate."),
            ],
            outputs=[
                io.String.Output(display_name="Output Directory"),
                io.String.Output(display_name="Manifest Path"),
                io.String.Output(display_name="Progress Path"),
                io.String.Output(display_name="Timeline XML Path"),
                io.String.Output(display_name="Status"),
                io.AnyType.Output(display_name="SRT TTS Results"),
            ],
        )

    @classmethod
    def execute(cls, model_name, lora_name, device, segments, voice_description, prompt_text,
                enable_asr, enable_denoiser, use_consistency_prompt, consistency_prompt,
                output_dir, job_name, filename_template, resume, overwrite, seed,
                seed_strategy, cfg_value, inference_timesteps, max_tokens, normalize_text,
                retry_max_attempts, retry_threshold, force_offload, dtype, device_input, torch_compile,
                clone_mode="controllable", export_premiere_xml=True, timeline_fps=30,
                reference_audio=None, **kwargs):
        device = device_input
        clone_mode = str(clone_mode or "controllable").strip()
        if clone_mode not in ("controllable", "ultimate", "auto"):
            clone_mode = "controllable"

        if not isinstance(segments, dict) or not segments.get("segments"):
            raise ValueError("SRT segments are required. Connect VoxCPM2 SRT Parser output.")

        segment_items: list[dict[str, Any]] = list(segments["segments"])
        job_dir = _resolve_job_dir(output_dir, job_name)
        wav_dir = job_dir / "wav"
        job_dir.mkdir(parents=True, exist_ok=True)
        wav_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "source_path": segments.get("source_path", ""),
            "model_name": model_name,
            "lora_name": lora_name,
            "voice_description": voice_description,
            "clone_mode": clone_mode,
            "use_consistency_prompt": bool(use_consistency_prompt),
            "consistency_prompt": consistency_prompt,
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "max_tokens": int(max_tokens),
            "normalize_text": bool(normalize_text),
            "seed": int(seed),
            "seed_strategy": seed_strategy,
            "enable_asr": bool(enable_asr),
            "enable_denoiser": bool(enable_denoiser),
        }
        write_json(job_dir / "config.json", config)

        manifest, completed = load_completed_manifest(job_dir) if resume else ([], {})
        used_names = {
            Path(str(item.get("output_file", ""))).name
            for item in manifest
            if str(item.get("output_file", ""))
        }

        success, failed = _success_counts(manifest)
        progress_path = write_progress(
            job_dir,
            job_name=job_dir.name,
            total=len(segment_items),
            processed=len(manifest),
            success=success,
            failed=failed,
            last_subtitle_index=None,
            status="running",
        )
        manifest_path = write_manifest(job_dir, manifest)

        ref_wav_path = None
        try:
            if reference_audio is not None:
                ref_wav_path = _save_audio_to_temp(reference_audio["waveform"], int(reference_audio["sample_rate"]))
                _validate_reference_audio_duration(ref_wav_path)
                if clone_mode in ("ultimate", "auto") and enable_asr and not (prompt_text and prompt_text.strip()):
                    prompt_text = transcribe_audio(ref_wav_path)

            patcher = _load_patcher(model_name, device, torch_compile, dtype)
            model_management.load_model_gpu(patcher)
            voxcpm_model = patcher.model.model
            if not voxcpm_model:
                raise RuntimeError(f"Failed to load model '{model_name}'.")

            if lora_name != "None":
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if not lora_path:
                    raise FileNotFoundError(f"LoRA file not found: {lora_name}")
                voxcpm_model.load_lora(lora_path)
                voxcpm_model.set_lora_enabled(True)
            else:
                voxcpm_model.set_lora_enabled(False)

            results: list[dict[str, Any]] = []
            processed_count = 0
            for segment in segment_items:
                subtitle_index = int(segment["index"])
                if resume and subtitle_index in completed and not overwrite:
                    results.append(completed[subtitle_index])
                    continue

                out_name = _format_output_name(filename_template, segment, used_names)
                relative_output = f"wav/{out_name}"
                out_path = wav_dir / out_name

                if out_path.exists() and not overwrite:
                    duration = get_audio_duration_seconds(out_path)
                    item = {
                        "subtitle_index": subtitle_index,
                        "start": segment["start"],
                        "end": segment["end"],
                        "start_seconds": float(segment["start_seconds"]),
                        "end_seconds": float(segment["end_seconds"]),
                        "text": segment["text"],
                        "output_file": relative_output,
                        "audio_duration_seconds": duration,
                        "timeline_start_seconds": float(segment["start_seconds"]),
                        "timeline_end_seconds": float(segment["start_seconds"]) + duration,
                        "subtitle_end_seconds": float(segment["end_seconds"]),
                        "overlaps_subtitle_end": float(segment["start_seconds"]) + duration > float(segment["end_seconds"]),
                        "model_name": model_name,
                        "lora_name": lora_name,
                        "mode": "existing",
                        "seed": -1,
                        "cfg_value": float(cfg_value),
                        "inference_timesteps": int(inference_timesteps),
                        "max_tokens": int(max_tokens),
                        "status": "ok",
                        "message": "Existing file skipped.",
                    }
                    upsert_manifest_item(manifest, item)
                    results.append(item)
                    continue

                actual_seed = _resolve_seed(int(seed), seed_strategy, segment)
                set_seed(actual_seed)
                final_text = _build_segment_text(str(segment["text"]), voice_description, bool(use_consistency_prompt), consistency_prompt)

                try:
                    has_prompt = bool(prompt_text and str(prompt_text).strip())
                    use_ultimate_clone = bool(ref_wav_path and has_prompt and clone_mode in ("ultimate", "auto"))
                    if use_ultimate_clone:
                        wav_array = voxcpm_model.generate(
                            text=final_text,
                            prompt_text=str(prompt_text).strip(),
                            prompt_wav_path=ref_wav_path,
                            reference_wav_path=ref_wav_path,
                            cfg_value=float(cfg_value),
                            inference_timesteps=int(inference_timesteps),
                            max_len=int(max_tokens),
                            normalize=bool(normalize_text),
                            denoise=bool(enable_denoiser),
                            retry_badcase=int(retry_max_attempts) > 0,
                            retry_badcase_max_times=int(retry_max_attempts),
                            retry_badcase_ratio_threshold=float(retry_threshold),
                        )
                        mode = "ultimate_clone"
                    elif ref_wav_path:
                        wav_array = voxcpm_model.generate(
                            text=final_text,
                            reference_wav_path=ref_wav_path,
                            cfg_value=float(cfg_value),
                            inference_timesteps=int(inference_timesteps),
                            max_len=int(max_tokens),
                            normalize=bool(normalize_text),
                            denoise=bool(enable_denoiser),
                            retry_badcase=int(retry_max_attempts) > 0,
                            retry_badcase_max_times=int(retry_max_attempts),
                            retry_badcase_ratio_threshold=float(retry_threshold),
                        )
                        mode = "voice_clone"
                    else:
                        wav_array = voxcpm_model.generate(
                            text=final_text,
                            cfg_value=float(cfg_value),
                            inference_timesteps=int(inference_timesteps),
                            max_len=int(max_tokens),
                            normalize=bool(normalize_text),
                        )
                        mode = "tts"

                    save_numpy_audio(wav_array, int(voxcpm_model.tts_model.sample_rate), out_path)
                    duration = get_audio_duration_seconds(out_path)
                    item = {
                        "subtitle_index": subtitle_index,
                        "start": segment["start"],
                        "end": segment["end"],
                        "start_seconds": float(segment["start_seconds"]),
                        "end_seconds": float(segment["end_seconds"]),
                        "text": segment["text"],
                        "output_file": relative_output,
                        "audio_duration_seconds": duration,
                        "timeline_start_seconds": float(segment["start_seconds"]),
                        "timeline_end_seconds": float(segment["start_seconds"]) + duration,
                        "subtitle_end_seconds": float(segment["end_seconds"]),
                        "overlaps_subtitle_end": float(segment["start_seconds"]) + duration > float(segment["end_seconds"]),
                        "model_name": model_name,
                        "lora_name": lora_name,
                        "mode": mode,
                        "seed": actual_seed,
                        "cfg_value": float(cfg_value),
                        "inference_timesteps": int(inference_timesteps),
                        "max_tokens": int(max_tokens),
                        "status": "ok",
                        "message": "Done",
                    }
                except Exception as exc:
                    item = {
                        "subtitle_index": subtitle_index,
                        "start": segment["start"],
                        "end": segment["end"],
                        "start_seconds": float(segment["start_seconds"]),
                        "end_seconds": float(segment["end_seconds"]),
                        "text": segment["text"],
                        "output_file": "",
                        "model_name": model_name,
                        "lora_name": lora_name,
                        "mode": "error",
                        "seed": actual_seed,
                        "status": "error",
                        "message": str(exc),
                    }

                upsert_manifest_item(manifest, item)
                results.append(item)
                processed_count += 1
                success, failed = _success_counts(manifest)
                manifest_path = write_manifest(job_dir, manifest)
                progress_path = write_progress(
                    job_dir,
                    job_name=job_dir.name,
                    total=len(segment_items),
                    processed=len(manifest),
                    success=success,
                    failed=failed,
                    last_subtitle_index=subtitle_index,
                    status="running",
                )

            success, failed = _success_counts(manifest)
            manifest_path = write_manifest(job_dir, manifest)
            progress_path = write_progress(
                job_dir,
                job_name=job_dir.name,
                total=len(segment_items),
                processed=len(manifest),
                success=success,
                failed=failed,
                last_subtitle_index=None if not manifest else int(manifest[-1].get("subtitle_index", 0)),
                status="finished",
            )

            timeline_xml_path = ""
            if export_premiere_xml:
                built_xml = build_premiere_xml(
                    job_name=job_dir.name,
                    manifest=manifest,
                    job_dir=job_dir,
                    xml_path=job_dir / "premiere_timeline.xml",
                    fps=int(timeline_fps),
                )
                if built_xml is not None:
                    timeline_xml_path = str(built_xml)

            if force_offload:
                cache_key = f"{model_name}_{device}_opt{patcher.model.optimize}_compile{torch_compile}_dtype{dtype}"
                patcher.force_unload()
                from .voxcpm2_nodes import VOXCPM_PATCHER_CACHE
                VOXCPM_PATCHER_CACHE.pop(cache_key, None)
                offload_asr()

            status = f"SRT 批量配音完成 / Finished SRT TTS job: {job_dir} | 成功 / success={success} | 失败 / failed={failed}"
            result_payload = {
                "job_dir": str(job_dir),
                "manifest_path": str(manifest_path),
                "progress_path": str(progress_path),
                "timeline_xml_path": timeline_xml_path,
                "results": manifest,
            }
            return io.NodeOutput(str(job_dir), str(manifest_path), str(progress_path), timeline_xml_path, status, result_payload)
        finally:
            if ref_wav_path:
                try:
                    os.unlink(ref_wav_path)
                except OSError:
                    pass
