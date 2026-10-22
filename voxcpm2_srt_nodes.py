import json
import logging
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

from .modules.asr_backend import (
    ASR_BACKEND_OPTIONS,
    DEFAULT_ASR_SERVER_URL,
    ASRTranscriber,
)
from .modules.model_info import AVAILABLE_VOXCPM_MODELS
from .modules.srt_audio_utils import (
    get_audio_duration_seconds,
    numpy_audio_to_waveform,
    save_waveform_audio,
    trim_waveform_silence,
    trim_waveform_start,
)
from .modules.srt_generation import (
    _build_segment_text,
    generate_segment_audio,
    prepare_reference_audio,
    _resolve_seed,
)
from .modules.srt_manifest import load_completed_manifest, upsert_manifest_item, write_json, write_manifest, write_progress
from .modules.srt_parser import build_preview_json, build_preview_text, parse_srt_file, sanitize_job_name, segments_to_payload
from .modules.srt_timeline import build_premiere_xml
from .voxcpm2_nodes import (
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

logger = logging.getLogger(__name__)


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


def _resolve_srt_source_name(srt_file: str, srt_path: str, resolved_srt_path: str) -> str:
    optional_file = str(srt_file or "").strip()
    if optional_file and optional_file != "None" and optional_file.lower().endswith(".srt"):
        return Path(optional_file.replace("\\", "/")).name
    if srt_path and str(srt_path).strip():
        return Path(str(srt_path).strip()).name
    return Path(str(resolved_srt_path or "")).name


def _resolve_job_dir(output_dir: str, job_name: str) -> Path:
    if output_dir and output_dir.strip():
        base_dir = Path(output_dir.strip())
    else:
        base_dir = Path(folder_paths.get_output_directory()) / "voxcpm2_srt"
    return base_dir / sanitize_job_name(job_name)


def _sanitize_filename_prefix(name: str) -> str:
    text = str(name or "").strip()
    text = "".join("_" if ch.isspace() else ch for ch in text)
    invalid_chars = '<>:"/\\|?*'
    text = "".join("_" if ch in invalid_chars or ord(ch) < 32 else ch for ch in text)
    text = text.strip("._-")
    return text


def _resolve_srt_name(source_path: str, source_name: str = "", srt_name: str = "") -> str:
    for candidate in (srt_name, Path(str(source_name or "")).stem, Path(str(source_path or "")).stem):
        cleaned = _sanitize_filename_prefix(candidate)
        if cleaned:
            return cleaned
    return ""


def _format_output_name(template: str, segment: dict[str, Any], used_names: set[str], srt_name: str = "", use_srt_name_prefix: bool = False) -> str:
    raw_template = (template or "{index:04d}.wav").strip() or "{index:04d}.wav"
    srt_name = _sanitize_filename_prefix(str(srt_name or "").strip())
    values = {
        "index": int(segment["index"]),
        "start": str(segment.get("start", "")).replace(":", "-").replace(",", "."),
        "end": str(segment.get("end", "")).replace(":", "-").replace(",", "."),
    }
    try:
        candidate = raw_template.format(**values)
    except Exception:
        candidate = f"{values['index']:04d}.wav"

    if bool(use_srt_name_prefix) and srt_name:
        candidate = f"{srt_name}_{candidate}"

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


def _success_counts(manifest: list[dict[str, Any]]) -> tuple[int, int]:
    success = sum(1 for item in manifest if item.get("status") == "ok")
    failed = sum(1 for item in manifest if item.get("status") == "error")
    return success, failed




def _build_srt_tts_common_inputs(model_names: list[str], devices: list[str], default_device: str) -> list:
    return [
        io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="选择要使用的 VoxCPM 模型。批量配音建议统一用一个模型，避免整批音色不一致。"),
        io.Combo.Input("lora_name", options=_available_loras(), default="None", tooltip="从 models/loras 加载 LoRA 微调权重。保持 None 表示不启用。"),
        io.String.Input("voice_description", multiline=True, default="", tooltip="语气/风格描述（可选），会以括号提示的形式加到每一段正文前面。例如“沉稳的男声，语速偏慢”。极致克隆模式下该描述的影响可能较弱。"),
        io.String.Input("prompt_text", multiline=True, default="", tooltip="参考音频对应的台词文本（要逐字对得上）。填写后进入极致克隆，相似度最高；留空则走可控克隆。"),
        io.Audio.Input("reference_audio", optional=True, tooltip="参考音频：用于模仿音色的样本，整批只放一段，建议 5~15 秒干净人声。不接则退化为普通文本转语音。末尾留 0.5~1 秒静音可减少每段开头的杂音。"),
        io.Boolean.Input("enable_asr", default=True, label_on="自动转写", label_off="关闭", tooltip="自动把参考音频转写成台词。仅在克隆模式为 ultimate/auto 且台词为空时生效；手填了台词就不再自动转写。"),
        io.Boolean.Input("enable_denoiser", default=False, label_on="降噪", label_off="关闭", tooltip="克隆前先对参考音频做降噪，适合有底噪的录音。需要 modelscope 依赖。"),
        io.Boolean.Input("use_consistency_prompt", default=True, tooltip="开启后会给每段追加一句“稳定提示词”，让整批的语气和音色更统一。长批量建议保持开启。"),
        io.String.Input("consistency_prompt", multiline=True, default=DEFAULT_CONSISTENCY_PROMPT, tooltip="稳定提示词：会放在每段正文前面的括号提示里，用来约束整批的语气、语速和音量保持一致。不改正文内容。"),
        io.String.Input("output_dir", default="", tooltip="输出根目录。留空则用 ComfyUI/output/voxcpm2_srt。"),
        io.String.Input("job_name", default="", tooltip="任务文件夹名。留空则用时间戳命名。目录批量模式下该项无效，会自动用每个 SRT 的文件名。"),
        io.String.Input("filename_template", default="{index:04d}.wav", tooltip="每段音频的文件名模板。{index} 是字幕序号，默认生成 0001.wav、0002.wav。"),
        io.Boolean.Input("resume", default=True, tooltip="继续任务：从已有的 manifest.json 读取进度，跳过已经生成成功的段落，只补做剩下的部分。"),
        io.Boolean.Input("overwrite", default=False, tooltip="覆盖已有音频。关闭时若同名 WAV 已存在会直接复用、不重新生成；开启则强制重新生成并覆盖。"),
        io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="基础随机种子。填固定数值整批结果可复现、音色更稳定；填 -1 表示每次随机。"),
        io.Combo.Input("seed_strategy", options=["fixed", "increment_by_index", "random", "hash_text"], default="fixed", tooltip="随机种子策略：fixed 每段用同一个种子（音色最稳，推荐）；increment_by_index 按序号递增；random 每段全随机；hash_text 按该段文字哈希取样。"),
        io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="引导强度（CFG）。数值越大越贴合提示描述、越稳定；越小变化越自然但更容易跑偏。批量配音建议 1.8~2.2。"),
        io.Int.Input("inference_timesteps", default=12, min=1, max=100, step=1, tooltip="推理步数。步数越多音质越好但整批耗时翻倍；12~16 步通常是性价比区间。"),
        io.Int.Input("max_tokens", default=4096, min=64, max=8192, tooltip="最大生成长度（token），限制单段能生成多长的音频。某段频繁被截断时调大它。"),
        io.Boolean.Input("normalize_text", default=True, label_on="规范化", label_off="原文", tooltip="自动处理数字、缩写和标点，让读法更自然。字幕里有特殊读法需求时可切到“原文”。"),
        io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="异常重试次数。某段生成出乱语、空白等坏结果时自动重来几次，填 0 表示不重试。"),
        io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="坏结果判定阈值。按“音频时长与文本长度之比”判断是否算失败，明显过短或过长就会触发重试。"),
        io.Boolean.Input("force_offload", default=False, label_on="强制卸载", label_off="自动", tooltip="整批生成结束后把模型从显存和内存中完全卸载。批量任务通常生成完就结束，可以开启。"),
        io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto", tooltip="模型精度。auto 会自动选最合适的一种（新显卡用 bf16，老显卡回退 fp16）。"),
        io.Combo.Input("device", options=devices, default=default_device, tooltip="推理使用的设备，一般选 CUDA 显卡；显存不足时可换 CPU（会很慢）。"),
        io.Boolean.Input("torch_compile", default=False, label_on="编译加速", label_off="标准", tooltip="开启 torch.compile 加速。第一次运行需要编译、会比较慢，之后的运行会明显更快。"),
        io.Combo.Input("clone_mode", options=["auto", "ultimate", "controllable"], default="auto", tooltip="克隆模式：auto 有台词或自动转写结果时走极致克隆，否则退回可控克隆；ultimate 音色最接近参考，但每段开头/结尾更容易出现“耶”声、啁啾等伪影；controllable 只靠参考音频，更适合长时间稳定批量。"),
        io.Boolean.Input("export_premiere_xml", default=True, label_on="导出 XML", label_off="不导出", tooltip="导出 Premiere 可导入的时间线 XML，按字幕时间点把音频排好，省去手动对轨。"),
        io.Int.Input("timeline_fps", default=30, min=0, max=120, step=1, tooltip="时间线的帧率，要和你 Premiere 序列的帧率一致（常见 25/30），否则导入后可能对不齐。"),
        io.Int.Input("trim_start_ms", default=100, min=0, max=2000, step=10, tooltip="开头裁剪：每段固定裁掉开头这么多毫秒，用来消除起始的杂音、啁啾或“耶”声。伪影较重时可加到 150~300。"),
        io.Boolean.Input("auto_trim_silence", default=False, label_on="裁剪静音", label_off="保留静音", tooltip="自动静音裁剪：在固定开头裁剪之后，再自动检测并裁掉首尾多余静音，让每段更贴合字幕时长。"),
        io.Float.Input("silence_threshold_db", default=-45.0, min=-80.0, max=-20.0, step=1.0, tooltip="静音阈值（dBFS）。低于这个响度就当作静音。数值越接近 0 判定越严格，裁剪越多。"),
        io.Int.Input("silence_min_duration_ms", default=200, min=0, max=2000, step=10, tooltip="最短静音时长。只有连续静音超过这么多毫秒才会被裁掉，太短的气口会保留。"),
        io.Int.Input("silence_keep_start_ms", default=80, min=0, max=500, step=10, tooltip="保留开头静音：裁剪后仍然在声音前留这么多毫秒，避免第一个字被削掉。"),
        io.Int.Input("silence_keep_end_ms", default=120, min=0, max=1000, step=10, tooltip="保留结尾静音：裁剪后仍然在尾音后留这么多毫秒，避免听感太突兀。"),
        io.Boolean.Input("use_srt_name_prefix", default=True, label_on="SRT 前缀", label_off="无前缀", tooltip="用 SRT 文件名作为 WAV 前缀（如 教程01_0001.wav）。多个任务素材混在一个工程里时开启，可避免 Premiere 串素材。"),
        io.Combo.Input("asr_backend", options=ASR_BACKEND_OPTIONS, default="embedded", tooltip="自动转写的后端。embedded 用 ComfyUI 进程内的 FunASR（旧行为）；http 走本机 127.0.0.1 上的 ASR Sidecar；auto 先试 embedded，失败再走 Sidecar。ComfyUI 是 Python 3.13、装不上 funasr 时请用 http。"),
        io.String.Input("asr_server_url", default=DEFAULT_ASR_SERVER_URL, tooltip="ASR Sidecar 地址，默认 http://127.0.0.1:8765。它是用 Python 3.12 单独启动的本机服务（tools/asr_sidecar.py），只监听 127.0.0.1，适合 Python 3.13 的 ComfyUI。仅当 asr_backend 为 http 或 auto 时才会使用。"),
    ]


def _scan_srt_files(folder_path: str, recursive: bool) -> list[Path]:
    folder = Path(str(folder_path or '').strip())
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"SRT folder not found: {folder_path}")
    pattern = '**/*.srt' if recursive else '*.srt'
    return sorted([path for path in folder.glob(pattern) if path.is_file()], key=lambda item: str(item).lower())


class VoxCPM2SRTParserNode(io.ComfyNode):
    CATEGORY = "audio/tts/srt"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VoxCPM2_SRT_Parser",
            display_name="VoxCPM2 SRT 解析器",
            category=cls.CATEGORY,
            description="读取 SRT 字幕文件，拆分成带时间轴的字幕段落，并输出可预览的文本与 JSON。",
            inputs=[
                io.String.Input("srt_path", default="", tooltip="SRT 字幕文件的本地路径。可以手填，也可以直接用下面的上传/选择框。"),
                io.Combo.Input("encoding", options=["auto", "utf-8-sig", "utf-8", "gbk"], default="auto", tooltip="字幕文件编码。中文出现乱码时，把它切到 gbk 通常就能解决。"),
                io.Boolean.Input("skip_empty", default=True, tooltip="跳过正文为空白的字幕条目，避免生成无意义的空音频。"),
                io.Boolean.Input("normalize_whitespace", default=True, tooltip="把多余的空格、换行整理成单空格，让朗读更连贯。"),
                io.Boolean.Input("strip_tags", default=True, tooltip="去掉字幕里简单的样式标签（如 <i>、<font> 等），只保留纯文字。"),
                io.Int.Input("preview_limit", default=30, min=0, max=500, tooltip="预览区显示多少条字幕。填 0 表示不显示预览，条目很多时调小可以减轻卡顿。"),
                io.String.Input("srt_file", default="", tooltip="上传或选择得到的 SRT 路径。填了它就以它为准；留空则使用上面的 srt_path。"),
            ],
            outputs=[
                io.AnyType.Output(display_name="SRT 字幕段落"),
                io.String.Output(display_name="预览文本"),
                io.String.Output(display_name="预览 JSON"),
                io.Int.Output(display_name="段落数量"),
                io.String.Output(display_name="实际 SRT 路径"),
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
        source_name = _resolve_srt_source_name(srt_file, srt_path, resolved_srt_path)
        payload = segments_to_payload(segments, resolved_srt_path, source_name=source_name)
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
            display_name="VoxCPM2 SRT 批量配音",
            category=cls.CATEGORY,
            description="按 SRT 字幕逐句生成音频，每一条字幕对应一个 WAV 文件，并输出清单、进度和 Premiere 时间线。",
            inputs=[
                io.AnyType.Input("segments", tooltip="来自“VoxCPM2 SRT 解析器”的字幕段落数据，必须连接。"),
                *_build_srt_tts_common_inputs(model_names, devices, default_device),
            ],
            outputs=[
                io.String.Output(display_name="输出目录"),
                io.String.Output(display_name="清单文件路径"),
                io.String.Output(display_name="进度文件路径"),
                io.String.Output(display_name="时间线 XML 路径"),
                io.String.Output(display_name="状态说明"),
                io.AnyType.Output(display_name="批量配音结果"),
            ],
        )

    @classmethod
    def execute(cls, model_name, lora_name, device, segments, voice_description, prompt_text,
                enable_asr, enable_denoiser, use_consistency_prompt, consistency_prompt,
                output_dir, job_name, filename_template, resume, overwrite, seed,
                seed_strategy, cfg_value, inference_timesteps, max_tokens, normalize_text,
                retry_max_attempts, retry_threshold, force_offload, dtype, torch_compile,
                clone_mode="auto", export_premiere_xml=True, timeline_fps=30,
                trim_start_ms=100, auto_trim_silence=False, silence_threshold_db=-45.0,
                silence_min_duration_ms=200, silence_keep_start_ms=80, silence_keep_end_ms=120,
                use_srt_name_prefix=True, reference_audio=None,
                asr_backend="embedded", asr_server_url=DEFAULT_ASR_SERVER_URL, **kwargs):
        trim_start_ms = max(0, min(2000, int(trim_start_ms or 0)))
        auto_trim_silence = bool(auto_trim_silence)
        silence_threshold_db = max(-80.0, min(-20.0, float(silence_threshold_db)))
        silence_min_duration_ms = max(0, min(2000, int(silence_min_duration_ms or 0)))
        silence_keep_start_ms = max(0, min(500, int(silence_keep_start_ms or 0)))
        silence_keep_end_ms = max(0, min(1000, int(silence_keep_end_ms or 0)))
        clone_mode = str(clone_mode or "controllable").strip()
        if clone_mode not in ("controllable", "ultimate", "auto"):
            clone_mode = "controllable"

        asr_transcriber = ASRTranscriber(
            asr_backend,
            asr_server_url,
            embedded_transcriber=transcribe_audio,
        )

        if not isinstance(segments, dict) or not segments.get("segments"):
            raise ValueError("SRT segments are required. Connect VoxCPM2 SRT Parser output.")

        segment_items: list[dict[str, Any]] = list(segments["segments"])
        source_path = str(segments.get("source_path", "") or "")
        source_name = str(segments.get("source_name", "") or "")
        payload_srt_name = str(segments.get("srt_name", "") or "")
        job_dir = _resolve_job_dir(output_dir, job_name)
        srt_name = _resolve_srt_name(source_path, source_name, payload_srt_name)
        wav_dir = job_dir / "wav"
        job_dir.mkdir(parents=True, exist_ok=True)
        wav_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "source_path": source_path,
            "source_name": source_name,
            "srt_name": payload_srt_name,
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
            "asr_backend": asr_transcriber.requested_backend,
            "asr_server_url": asr_transcriber.server_url,
            "enable_denoiser": bool(enable_denoiser),
            "trim_start_ms": trim_start_ms,
            "auto_trim_silence": auto_trim_silence,
            "silence_threshold_db": silence_threshold_db,
            "silence_min_duration_ms": silence_min_duration_ms,
            "silence_keep_start_ms": silence_keep_start_ms,
            "silence_keep_end_ms": silence_keep_end_ms,
            "timeline_alignment": "align_to_subtitle_start",
            "use_srt_name_prefix": bool(use_srt_name_prefix),
            "filename_prefix": srt_name if bool(use_srt_name_prefix) else "",
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
            if enable_asr:
                logger.info(
                    "[ASR] SRT job %s: reference transcription backend %s",
                    job_dir.name,
                    asr_transcriber.describe(),
                )
            ref_wav_path, prompt_text = prepare_reference_audio(
                reference_audio,
                clone_mode,
                enable_asr,
                prompt_text,
                save_audio_to_temp=_save_audio_to_temp,
                validate_reference_audio_duration=_validate_reference_audio_duration,
                transcribe_audio=asr_transcriber,
            )
            if asr_transcriber.last_backend:
                logger.info(
                    "[ASR] SRT job %s: transcript produced by %s backend (%d chars)",
                    job_dir.name,
                    asr_transcriber.last_backend,
                    len(prompt_text or ""),
                )
                config["asr_backend_used"] = asr_transcriber.last_backend
                write_json(job_dir / "config.json", config)
            elif enable_asr and clone_mode == "controllable":
                logger.info(
                    "[ASR] SRT job %s: clone_mode=controllable, reference audio is not transcribed.",
                    job_dir.name,
                )
            else:
                logger.info(
                    "[ASR] SRT job %s: no transcription requested (prompt_text provided or ASR disabled).",
                    job_dir.name,
                )
            if clone_mode == "ultimate" and not (prompt_text and prompt_text.strip()):
                logger.warning(
                    "[ASR] SRT job %s: clone_mode=ultimate but no transcript is available; "
                    "generation will fall back to plain voice cloning.",
                    job_dir.name,
                )

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

                out_name = _format_output_name(filename_template, segment, used_names, srt_name, bool(use_srt_name_prefix))
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
                        "trim_start_ms": 0,
                        "trim_start_seconds": 0.0,
                        "auto_trim_silence": auto_trim_silence,
                        "silence_trim_enabled": False,
                        "silence_trim_leading_ms": 0,
                        "silence_trim_trailing_ms": 0,
                        "total_trimmed_leading_ms": 0,
                        "total_trimmed_trailing_ms": 0,
                        "original_audio_duration_seconds": duration,
                        "timeline_alignment": "align_to_subtitle_start",
                        "filename_prefix": srt_name if bool(use_srt_name_prefix) else "",
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
                    wav_array, mode = generate_segment_audio(
                        voxcpm_model,
                        text=final_text,
                        reference_audio_path=ref_wav_path,
                        prompt_text=prompt_text,
                        clone_mode=clone_mode,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        max_tokens=max_tokens,
                        normalize_text=normalize_text,
                        enable_denoiser=enable_denoiser,
                        retry_max_attempts=retry_max_attempts,
                        retry_threshold=retry_threshold,
                    )

                    sample_rate = int(voxcpm_model.tts_model.sample_rate)
                    waveform = numpy_audio_to_waveform(wav_array)
                    original_audio_duration_seconds = float(waveform.shape[-1]) / float(sample_rate)
                    waveform, applied_trim_start_ms = trim_waveform_start(waveform, sample_rate, trim_start_ms)
                    fixed_trimmed_duration_seconds = float(waveform.shape[-1]) / float(sample_rate)
                    silence_trim_info = {
                        "silence_trim_leading_ms": 0,
                        "silence_trim_trailing_ms": 0,
                        "silence_trim_threshold_db": silence_threshold_db,
                        "silence_trim_min_duration_ms": silence_min_duration_ms,
                        "silence_trim_keep_start_ms": silence_keep_start_ms,
                        "silence_trim_keep_end_ms": silence_keep_end_ms,
                        "pre_silence_trim_duration_seconds": fixed_trimmed_duration_seconds,
                        "post_silence_trim_duration_seconds": fixed_trimmed_duration_seconds,
                    }
                    if auto_trim_silence:
                        waveform, silence_trim_info = trim_waveform_silence(
                            waveform,
                            sample_rate,
                            threshold_db=silence_threshold_db,
                            min_silence_ms=silence_min_duration_ms,
                            keep_start_ms=silence_keep_start_ms,
                            keep_end_ms=silence_keep_end_ms,
                        )
                    save_waveform_audio(waveform, sample_rate, out_path)
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
                        "trim_start_ms": applied_trim_start_ms,
                        "trim_start_seconds": applied_trim_start_ms / 1000.0,
                        "auto_trim_silence": auto_trim_silence,
                        "silence_trim_enabled": auto_trim_silence,
                        "silence_trim_leading_ms": int(silence_trim_info["silence_trim_leading_ms"]),
                        "silence_trim_trailing_ms": int(silence_trim_info["silence_trim_trailing_ms"]),
                        "silence_trim_threshold_db": float(silence_trim_info["silence_trim_threshold_db"]),
                        "silence_trim_min_duration_ms": int(silence_trim_info["silence_trim_min_duration_ms"]),
                        "silence_trim_keep_start_ms": int(silence_trim_info["silence_trim_keep_start_ms"]),
                        "silence_trim_keep_end_ms": int(silence_trim_info["silence_trim_keep_end_ms"]),
                        "total_trimmed_leading_ms": applied_trim_start_ms + int(silence_trim_info["silence_trim_leading_ms"]),
                        "total_trimmed_trailing_ms": int(silence_trim_info["silence_trim_trailing_ms"]),
                        "original_audio_duration_seconds": original_audio_duration_seconds,
                        "pre_silence_trim_duration_seconds": float(silence_trim_info["pre_silence_trim_duration_seconds"]),
                        "post_silence_trim_duration_seconds": float(silence_trim_info["post_silence_trim_duration_seconds"]),
                        "timeline_alignment": "align_to_subtitle_start",
                        "filename_prefix": srt_name if bool(use_srt_name_prefix) else "",
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

            status = f"SRT 批量配音完成：任务目录 {job_dir} | 成功 {success} 段 | 失败 {failed} 段 | 开头裁剪 {trim_start_ms} 毫秒 | 自动静音裁剪 {'开启' if auto_trim_silence else '关闭'} | ASR 后端 {asr_transcriber.last_backend or '未使用'}"
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


class VoxCPM2SRTSingleLineTTSNode(io.ComfyNode):
    CATEGORY = "audio/tts/srt"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = _available_model_names()
        devices = get_available_devices()
        default_device = devices[0]
        return io.Schema(
            node_id="VoxCPM2_SRT_Single_Line_TTS",
            display_name="VoxCPM2 SRT 单句配音",
            category=cls.CATEGORY,
            description="手动输入一句话，单独生成一个 WAV 文件。用于补录或替换某一句字幕的配音。",
            inputs=[
                io.String.Input("text", multiline=True, default="", tooltip="要生成的那一句台词。通常是从 SRT 里挑出来需要重录或替换的那一条。"),
                *_build_srt_tts_common_inputs(model_names, devices, default_device),
            ],
            outputs=[
                io.String.Output(display_name="WAV 文件路径"),
                io.Float.Output(display_name="音频时长（秒）"),
                io.String.Output(display_name="状态说明"),
                io.AnyType.Output(display_name="单句配音结果"),
            ],
        )

    @classmethod
    def execute(cls, text, model_name, lora_name, device, voice_description, prompt_text,
                enable_asr, enable_denoiser, use_consistency_prompt, consistency_prompt,
                output_dir, job_name, filename_template, resume, overwrite, seed,
                seed_strategy, cfg_value, inference_timesteps, max_tokens, normalize_text,
                retry_max_attempts, retry_threshold, force_offload, dtype, torch_compile,
                clone_mode="auto", export_premiere_xml=True, timeline_fps=30,
                trim_start_ms=100, auto_trim_silence=False, silence_threshold_db=-45.0,
                silence_min_duration_ms=200, silence_keep_start_ms=80, silence_keep_end_ms=120,
                use_srt_name_prefix=True, reference_audio=None,
                asr_backend="embedded", asr_server_url=DEFAULT_ASR_SERVER_URL, **kwargs):
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("Text is required for single line TTS.")

        trim_start_ms = max(0, min(2000, int(trim_start_ms or 0)))
        auto_trim_silence = bool(auto_trim_silence)
        silence_threshold_db = max(-80.0, min(-20.0, float(silence_threshold_db)))
        silence_min_duration_ms = max(0, min(2000, int(silence_min_duration_ms or 0)))
        silence_keep_start_ms = max(0, min(500, int(silence_keep_start_ms or 0)))
        silence_keep_end_ms = max(0, min(1000, int(silence_keep_end_ms or 0)))
        clone_mode = str(clone_mode or "controllable").strip()
        if clone_mode not in ("controllable", "ultimate", "auto"):
            clone_mode = "controllable"

        asr_transcriber = ASRTranscriber(
            asr_backend,
            asr_server_url,
            embedded_transcriber=transcribe_audio,
        )

        output_base = Path(str(output_dir).strip()) if output_dir and str(output_dir).strip() else Path(folder_paths.get_output_directory())
        output_base.mkdir(parents=True, exist_ok=True)
        output_stem = _sanitize_filename_prefix(str(job_name or "").strip())
        if not output_stem:
            output_stem = time.strftime("single_line_%Y%m%d_%H%M%S")
        if output_stem.lower().endswith(".wav"):
            output_name = output_stem
        else:
            output_name = f"{output_stem}.wav"
        out_path = output_base / output_name
        if out_path.exists() and not overwrite:
            output_stem = out_path.stem
            output_name = f"{output_stem}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            out_path = output_base / output_name

        segment = {"index": 1, "text": clean_text}
        actual_seed = _resolve_seed(int(seed), seed_strategy, segment)
        final_text = _build_segment_text(clean_text, voice_description, bool(use_consistency_prompt), consistency_prompt)

        ref_wav_path = None
        try:
            if enable_asr:
                logger.info("[ASR] single line TTS: reference transcription backend %s", asr_transcriber.describe())
            ref_wav_path, prompt_text = prepare_reference_audio(
                reference_audio,
                clone_mode,
                enable_asr,
                prompt_text,
                save_audio_to_temp=_save_audio_to_temp,
                validate_reference_audio_duration=_validate_reference_audio_duration,
                transcribe_audio=asr_transcriber,
            )
            if asr_transcriber.last_backend:
                logger.info(
                    "[ASR] single line TTS: transcript produced by %s backend (%d chars)",
                    asr_transcriber.last_backend,
                    len(prompt_text or ""),
                )
            if clone_mode == "ultimate" and not (prompt_text and prompt_text.strip()):
                logger.warning(
                    "[ASR] single line TTS: clone_mode=ultimate but no transcript is available; "
                    "generation will fall back to plain voice cloning."
                )

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

            set_seed(actual_seed)
            wav_array, mode = generate_segment_audio(
                voxcpm_model,
                text=final_text,
                reference_audio_path=ref_wav_path,
                prompt_text=prompt_text,
                clone_mode=clone_mode,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_tokens=max_tokens,
                normalize_text=normalize_text,
                enable_denoiser=enable_denoiser,
                retry_max_attempts=retry_max_attempts,
                retry_threshold=retry_threshold,
            )

            sample_rate = int(voxcpm_model.tts_model.sample_rate)
            waveform = numpy_audio_to_waveform(wav_array)
            original_audio_duration_seconds = float(waveform.shape[-1]) / float(sample_rate)
            waveform, applied_trim_start_ms = trim_waveform_start(waveform, sample_rate, trim_start_ms)
            fixed_trimmed_duration_seconds = float(waveform.shape[-1]) / float(sample_rate)
            silence_trim_info = {
                "silence_trim_leading_ms": 0,
                "silence_trim_trailing_ms": 0,
                "silence_trim_threshold_db": silence_threshold_db,
                "silence_trim_min_duration_ms": silence_min_duration_ms,
                "silence_trim_keep_start_ms": silence_keep_start_ms,
                "silence_trim_keep_end_ms": silence_keep_end_ms,
                "pre_silence_trim_duration_seconds": fixed_trimmed_duration_seconds,
                "post_silence_trim_duration_seconds": fixed_trimmed_duration_seconds,
            }
            if auto_trim_silence:
                waveform, silence_trim_info = trim_waveform_silence(
                    waveform,
                    sample_rate,
                    threshold_db=silence_threshold_db,
                    min_silence_ms=silence_min_duration_ms,
                    keep_start_ms=silence_keep_start_ms,
                    keep_end_ms=silence_keep_end_ms,
                )
            save_waveform_audio(waveform, sample_rate, out_path)
            duration = get_audio_duration_seconds(out_path)

            if force_offload:
                cache_key = f"{model_name}_{device}_opt{patcher.model.optimize}_compile{torch_compile}_dtype{dtype}"
                patcher.force_unload()
                from .voxcpm2_nodes import VOXCPM_PATCHER_CACHE
                VOXCPM_PATCHER_CACHE.pop(cache_key, None)
                offload_asr()

            item = {
                "text": clean_text,
                "final_text": final_text,
                "output_file": str(out_path),
                "audio_duration_seconds": duration,
                "model_name": model_name,
                "lora_name": lora_name,
                "mode": mode,
                "asr_backend_used": asr_transcriber.last_backend,
                "seed": actual_seed,
                "cfg_value": float(cfg_value),
                "inference_timesteps": int(inference_timesteps),
                "max_tokens": int(max_tokens),
                "trim_start_ms": applied_trim_start_ms,
                "trim_start_seconds": applied_trim_start_ms / 1000.0,
                "auto_trim_silence": auto_trim_silence,
                "silence_trim_leading_ms": int(silence_trim_info["silence_trim_leading_ms"]),
                "silence_trim_trailing_ms": int(silence_trim_info["silence_trim_trailing_ms"]),
                "original_audio_duration_seconds": original_audio_duration_seconds,
                "status": "ok",
                "message": "Done",
            }
            status = f"单句配音完成：{out_path} | 时长 {duration:.2f} 秒 | 模式 {mode} | ASR 后端 {asr_transcriber.last_backend or '未使用'}"
            return io.NodeOutput(str(out_path), float(duration), status, item)
        finally:
            if ref_wav_path:
                try:
                    os.unlink(ref_wav_path)
                except OSError:
                    pass


class VoxCPM2SRTFolderBatchTTSNode(io.ComfyNode):
    CATEGORY = "audio/tts/srt"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = _available_model_names()
        devices = get_available_devices()
        default_device = devices[0]
        return io.Schema(
            node_id="VoxCPM2_SRT_Folder_Batch_TTS",
            display_name="VoxCPM2 SRT 目录批量配音",
            category=cls.CATEGORY,
            description="扫描一个文件夹里的所有 SRT 字幕，每个 SRT 独立生成一套配音音频、清单和 Premiere 时间线。",
            inputs=[
                io.String.Input("srt_folder_path", default="", tooltip="包含 .srt 字幕文件的文件夹路径。"),
                io.Boolean.Input("recursive", default=False, label_on="含子目录", label_off="仅当前层", tooltip="开启后会连子文件夹一起扫描；关闭则只处理当前这一层。"),
                *_build_srt_tts_common_inputs(model_names, devices, default_device),
            ],
            outputs=[
                io.String.Output(display_name="输出根目录"),
                io.String.Output(display_name="汇总文件路径"),
                io.String.Output(display_name="状态说明"),
                io.AnyType.Output(display_name="目录批量结果"),
            ],
        )

    @classmethod
    def execute(cls, srt_folder_path, recursive, model_name, lora_name, device,
                voice_description, prompt_text, enable_asr, enable_denoiser,
                use_consistency_prompt, consistency_prompt, output_dir, job_name,
                filename_template, resume, overwrite, seed, seed_strategy, cfg_value,
                inference_timesteps, max_tokens, normalize_text, retry_max_attempts,
                retry_threshold, force_offload, dtype, torch_compile, clone_mode="auto",
                export_premiere_xml=True, timeline_fps=30, trim_start_ms=100,
                auto_trim_silence=False, silence_threshold_db=-45.0,
                silence_min_duration_ms=200, silence_keep_start_ms=80,
                silence_keep_end_ms=120, use_srt_name_prefix=True, reference_audio=None,
                asr_backend="embedded", asr_server_url=DEFAULT_ASR_SERVER_URL,
                **kwargs):
        srt_files = _scan_srt_files(srt_folder_path, bool(recursive))
        if not srt_files:
            raise ValueError(f"No .srt files found in folder: {srt_folder_path}")

        root_output_dir = Path(output_dir.strip()) if output_dir and str(output_dir).strip() else Path(folder_paths.get_output_directory()) / "voxcpm2_srt"
        root_output_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        success_jobs = 0
        failed_jobs = 0

        for srt_path_obj in srt_files:
            srt_name = _sanitize_filename_prefix(srt_path_obj.stem) or srt_path_obj.stem
            try:
                parsed = parse_srt_file(str(srt_path_obj), encoding="auto", skip_empty=True, normalize_whitespace=True, strip_tags=True)
                payload = segments_to_payload(parsed, str(srt_path_obj), source_name=srt_path_obj.name)
                VoxCPM2SRTBatchTTSNode.execute(
                    model_name, lora_name, device, payload, voice_description, prompt_text,
                    enable_asr, enable_denoiser, use_consistency_prompt, consistency_prompt,
                    str(root_output_dir), srt_name, filename_template, resume, overwrite, seed,
                    seed_strategy, cfg_value, inference_timesteps, max_tokens, normalize_text,
                    retry_max_attempts, retry_threshold, False, dtype, torch_compile,
                    clone_mode=clone_mode, export_premiere_xml=export_premiere_xml,
                    timeline_fps=timeline_fps, trim_start_ms=trim_start_ms,
                    auto_trim_silence=auto_trim_silence,
                    silence_threshold_db=silence_threshold_db,
                    silence_min_duration_ms=silence_min_duration_ms,
                    silence_keep_start_ms=silence_keep_start_ms,
                    silence_keep_end_ms=silence_keep_end_ms,
                    use_srt_name_prefix=use_srt_name_prefix,
                    reference_audio=reference_audio,
                    asr_backend=asr_backend,
                    asr_server_url=asr_server_url,
                )
                job_dir = root_output_dir / srt_name
                jobs.append({
                    "srt_path": str(srt_path_obj),
                    "job_name": srt_name,
                    "job_dir": str(job_dir),
                    "manifest_path": str(job_dir / "manifest.json"),
                    "progress_path": str(job_dir / "progress.json"),
                    "timeline_xml_path": str(job_dir / "premiere_timeline.xml") if export_premiere_xml else "",
                    "status": "ok",
                })
                success_jobs += 1
            except Exception as exc:
                jobs.append({"srt_path": str(srt_path_obj), "job_name": srt_name, "status": "error", "message": str(exc)})
                failed_jobs += 1

        if force_offload:
            offload_asr()

        summary = {
            "status": "finished",
            "srt_folder_path": str(srt_folder_path),
            "output_root_dir": str(root_output_dir),
            "total_srt_files": len(srt_files),
            "success_jobs": success_jobs,
            "failed_jobs": failed_jobs,
            "jobs": jobs,
        }
        summary_path = write_json(root_output_dir / "batch_summary.json", summary)
        status = f"目录批量配音完成：共 {len(srt_files)} 个字幕 | 成功 {success_jobs} 个 | 失败 {failed_jobs} 个"
        return io.NodeOutput(str(root_output_dir), str(summary_path), status, summary)
