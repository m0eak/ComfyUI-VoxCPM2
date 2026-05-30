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
