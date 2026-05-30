from pathlib import Path

import torch
import torchaudio


def ensure_audio_tensor_shape(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform.float().cpu()


def save_audio(audio: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = ensure_audio_tensor_shape(audio["waveform"])
    sample_rate = int(audio["sample_rate"])
    torchaudio.save(str(path), waveform, sample_rate)
    return path


def numpy_audio_to_waveform(wav_array) -> torch.Tensor:
    waveform = torch.from_numpy(wav_array).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T
    return waveform.cpu()


def trim_waveform_start(waveform: torch.Tensor, sample_rate: int, trim_start_ms: int) -> tuple[torch.Tensor, int]:
    trim_start_ms = max(0, int(trim_start_ms or 0))
    if trim_start_ms <= 0:
        return waveform, 0

    sample_rate = int(sample_rate)
    trim_samples = int(sample_rate * trim_start_ms / 1000)
    if trim_samples <= 0:
        return waveform, 0

    if waveform.dim() == 1:
        total_samples = waveform.shape[0]
        if trim_samples >= total_samples:
            return waveform, 0
        return waveform[trim_samples:], trim_start_ms

    total_samples = waveform.shape[-1]
    if trim_samples >= total_samples:
        return waveform, 0
    return waveform[..., trim_samples:], trim_start_ms


def trim_waveform_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -45.0,
    min_silence_ms: int = 200,
    keep_start_ms: int = 80,
    keep_end_ms: int = 120,
) -> tuple[torch.Tensor, dict]:
    sample_rate = int(sample_rate)
    total_samples = int(waveform.shape[-1]) if waveform.dim() > 1 else int(waveform.shape[0])
    original_duration = float(total_samples) / float(sample_rate) if sample_rate > 0 else 0.0
    info = {
        "silence_trim_leading_ms": 0,
        "silence_trim_trailing_ms": 0,
        "silence_trim_threshold_db": float(threshold_db),
        "silence_trim_min_duration_ms": max(0, int(min_silence_ms or 0)),
        "silence_trim_keep_start_ms": max(0, int(keep_start_ms or 0)),
        "silence_trim_keep_end_ms": max(0, int(keep_end_ms or 0)),
        "pre_silence_trim_duration_seconds": original_duration,
    }
    if sample_rate <= 0 or total_samples <= 0:
        info["post_silence_trim_duration_seconds"] = original_duration
        return waveform, info

    threshold = 10 ** (float(threshold_db) / 20.0)
    if waveform.dim() == 1:
        amplitude = waveform.abs()
    else:
        amplitude = waveform.abs().amax(dim=0)

    sound_indices = torch.nonzero(amplitude > threshold, as_tuple=False).flatten()
    if sound_indices.numel() == 0:
        info["post_silence_trim_duration_seconds"] = original_duration
        return waveform, info

    first_sound = int(sound_indices[0].item())
    last_sound = int(sound_indices[-1].item())
    keep_start_samples = int(sample_rate * info["silence_trim_keep_start_ms"] / 1000)
    keep_end_samples = int(sample_rate * info["silence_trim_keep_end_ms"] / 1000)
    min_silence_samples = int(sample_rate * info["silence_trim_min_duration_ms"] / 1000)

    proposed_start = max(0, first_sound - keep_start_samples)
    proposed_end = min(total_samples, last_sound + keep_end_samples + 1)

    leading_trim_samples = proposed_start if proposed_start >= min_silence_samples else 0
    trailing_trim_samples = (total_samples - proposed_end) if (total_samples - proposed_end) >= min_silence_samples else 0

    start = leading_trim_samples
    end = total_samples - trailing_trim_samples
    if end <= start:
        info["post_silence_trim_duration_seconds"] = original_duration
        return waveform, info

    trimmed = waveform[start:end] if waveform.dim() == 1 else waveform[..., start:end]
    info["silence_trim_leading_ms"] = round(leading_trim_samples * 1000.0 / sample_rate)
    info["silence_trim_trailing_ms"] = round(trailing_trim_samples * 1000.0 / sample_rate)
    info["post_silence_trim_duration_seconds"] = float(trimmed.shape[-1] if trimmed.dim() > 1 else trimmed.shape[0]) / float(sample_rate)
    return trimmed, info


def save_waveform_audio(waveform: torch.Tensor, sample_rate: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.cpu(), int(sample_rate))
    return path


def save_numpy_audio(wav_array, sample_rate: int, path: Path) -> Path:
    waveform = numpy_audio_to_waveform(wav_array)
    return save_waveform_audio(waveform, sample_rate, path)


def get_audio_duration_seconds(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate for audio file: {path}")
    return float(info.frames) / float(info.samplerate)
