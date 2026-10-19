"""Shared SRT generation business logic, independent of ComfyUI schemas."""

import hashlib
import os
import random
from typing import Any, Callable


def _resolve_seed(base_seed: int, strategy: str, segment: dict[str, Any]) -> int:
    if int(base_seed) < 0:
        return random.randint(0, 2**31 - 1)

    strategy = strategy or "increment_by_index"
    if strategy == "fixed":
        return int(base_seed)
    if strategy == "random":
        return random.randint(0, 2**31 - 1)
    if strategy == "hash_text":
        digest = hashlib.sha256(str(segment.get("text", "")).encode("utf-8")).hexdigest()
        return (int(base_seed) + int(digest[:8], 16)) & 0xFFFFFFFF
    return int(base_seed) + int(segment["index"])


def _build_segment_text(
    text: str,
    voice_description: str,
    use_consistency_prompt: bool,
    consistency_prompt: str,
) -> str:
    desc_parts: list[str] = []
    if voice_description and voice_description.strip():
        desc_parts.append(voice_description.strip())
    if use_consistency_prompt and consistency_prompt and consistency_prompt.strip():
        desc_parts.append(consistency_prompt.strip())
    if desc_parts:
        return f"({'；'.join(desc_parts)}){text.strip()}"
    return text.strip()


def should_transcribe_reference_audio(
    clone_mode: str,
    enable_asr: bool,
    prompt_text: str,
) -> bool:
    return bool(
        clone_mode in ("ultimate", "auto")
        and enable_asr
        and not (prompt_text and prompt_text.strip())
    )


def prepare_reference_audio(
    reference_audio: dict[str, Any] | None,
    clone_mode: str,
    enable_asr: bool,
    prompt_text: str,
    *,
    save_audio_to_temp: Callable[[Any, int], str],
    validate_reference_audio_duration: Callable[[str], Any],
    transcribe_audio: Callable[[str], str],
) -> tuple[str | None, str]:
    if reference_audio is None:
        return None, prompt_text

    reference_path = save_audio_to_temp(
        reference_audio["waveform"], int(reference_audio["sample_rate"])
    )
    try:
        validate_reference_audio_duration(reference_path)
        if should_transcribe_reference_audio(clone_mode, enable_asr, prompt_text):
            prompt_text = transcribe_audio(reference_path)
    except Exception:
        try:
            os.unlink(reference_path)
        except OSError:
            pass
        raise

    return reference_path, prompt_text


def generate_segment_audio(
    model: Any,
    *,
    text: str,
    reference_audio_path: str | None,
    prompt_text: str,
    clone_mode: str,
    cfg_value: float,
    inference_timesteps: int,
    max_tokens: int,
    normalize_text: bool,
    enable_denoiser: bool,
    retry_max_attempts: int,
    retry_threshold: float,
) -> tuple[Any, str]:
    has_prompt = bool(prompt_text and str(prompt_text).strip())
    use_ultimate_clone = bool(
        reference_audio_path
        and has_prompt
        and clone_mode in ("ultimate", "auto")
    )
    if use_ultimate_clone:
        wav_array = model.generate(
            text=text,
            prompt_text=str(prompt_text).strip(),
            prompt_wav_path=reference_audio_path,
            reference_wav_path=reference_audio_path,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            max_len=int(max_tokens),
            normalize=bool(normalize_text),
            denoise=bool(enable_denoiser),
            retry_badcase=int(retry_max_attempts) > 0,
            retry_badcase_max_times=int(retry_max_attempts),
            retry_badcase_ratio_threshold=float(retry_threshold),
        )
        return wav_array, "ultimate_clone"
    if reference_audio_path:
        wav_array = model.generate(
            text=text,
            reference_wav_path=reference_audio_path,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            max_len=int(max_tokens),
            normalize=bool(normalize_text),
            denoise=bool(enable_denoiser),
            retry_badcase=int(retry_max_attempts) > 0,
            retry_badcase_max_times=int(retry_max_attempts),
            retry_badcase_ratio_threshold=float(retry_threshold),
        )
        return wav_array, "voice_clone"

    wav_array = model.generate(
        text=text,
        cfg_value=float(cfg_value),
        inference_timesteps=int(inference_timesteps),
        max_len=int(max_tokens),
        normalize=bool(normalize_text),
    )
    return wav_array, "tts"
