import os
import re
import tempfile
import torch
import torchaudio
import logging
from typing import List

import folder_paths
import comfy.model_management as model_management
from comfy_api.latest import ComfyExtension, io, ui

from .modules.model_info import AVAILABLE_VOXCPM_MODELS
from .modules.loader import VoxCPMModelHandler
from .modules.patcher import VoxCPMPatcher

# LoRA training nodes are intentionally opt-in because importing them pulls in
# voxcpm.training -> datasets -> pyarrow. On some Windows/ComfyUI setups these
# native extensions can hard-crash the process during custom node discovery.
# Set VOXCPM2_ENABLE_TRAINING_NODES=1 to expose training nodes.
_ENABLE_TRAINING_NODES = os.environ.get("VOXCPM2_ENABLE_TRAINING_NODES", "0").strip().lower() in {"1", "true", "yes", "on"}
_TRAINING_NODES = []
if _ENABLE_TRAINING_NODES:
    from .voxcpm2_train_nodes import VoxCPM_TrainConfig, VoxCPM_DatasetMaker, VoxCPM_LoraTrainer
    _TRAINING_NODES = [VoxCPM_TrainConfig, VoxCPM_DatasetMaker, VoxCPM_LoraTrainer]

logger = logging.getLogger(__name__)

VOXCPM_PATCHER_CACHE = {}
MAX_REFERENCE_AUDIO_SECONDS = 50.0

# ---------------------------------------------------------------------------
# ASR (Auto Speech Recognition) — lazy-loaded singleton
# ---------------------------------------------------------------------------
_ASR_MODEL = None


def get_asr_model():
    """Lazily load and cache the FunASR SenseVoiceSmall model for auto-transcription."""
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL
    try:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        asr_model_ref = "FunAudioLLM/SenseVoiceSmall"

        # Cache in ComfyUI's audio_encoders directory
        encoders_dir = os.path.join(folder_paths.models_dir, "audio_encoders")
        os.makedirs(encoders_dir, exist_ok=True)
        asr_local_dir = os.path.join(encoders_dir, "SenseVoiceSmall")

        # Download only if not already cached locally
        if not os.path.isdir(asr_local_dir) or not os.listdir(asr_local_dir):
            logger.info(f"Downloading ASR model '{asr_model_ref}' to {asr_local_dir}...")
            snapshot_download(
                repo_id=asr_model_ref,
                local_dir=asr_local_dir,
                local_dir_use_symlinks=False,
            )
        else:
            logger.info(f"Using cached ASR model from: {asr_local_dir}")

        logger.info("Loading ASR model on CPU ...")
        _ASR_MODEL = AutoModel(
            model=asr_local_dir,
            disable_update=True,
            log_level="INFO",
            device="cpu",
        )
        logger.info("ASR model loaded.")
        return _ASR_MODEL
    except ImportError:
        raise ImportError(
            "ASR requires funasr. Install with: pip install funasr"
        )


def transcribe_audio(wav_path: str) -> str:
    """Run ASR on a wav file and return the transcribed text."""
    asr_model = get_asr_model()
    res = asr_model.generate(input=wav_path, language="auto", use_itn=True)
    if not res:
        return ""
    first_item = res[0]
    if isinstance(first_item, dict):
        return re.sub(r"<\|[^|>]*\|>", "", str(first_item.get("text", ""))).strip()
    return ""


def offload_asr():
    """Release the ASR model from memory."""
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        del _ASR_MODEL
        _ASR_MODEL = None
        import gc
        gc.collect()
        logger.info("ASR model offloaded.")


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def set_seed(seed: int):
    import random
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed & 0xFFFFFFFF)
    except ImportError:
        pass


def _save_audio_to_temp(waveform: torch.Tensor, sample_rate: int) -> str:
    """Save a ComfyUI audio tensor to a temporary WAV file and return the path."""
    # Ensure shape is [1, T]
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        torchaudio.save(tmp.name, waveform, sample_rate)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return tmp.name


def _get_audio_duration_seconds(wav_path: str) -> float:
    """Get audio duration in seconds using soundfile."""
    import soundfile as sf
    info = sf.info(wav_path)
    return float(info.frames) / float(info.samplerate)


def _validate_reference_audio_duration(wav_path: str):
    """Raise ValueError if reference audio exceeds the maximum allowed duration."""
    duration = _get_audio_duration_seconds(wav_path)
    if duration > MAX_REFERENCE_AUDIO_SECONDS:
        raise ValueError(
            f"Reference audio is {duration:.1f}s — max allowed is {MAX_REFERENCE_AUDIO_SECONDS:.0f}s. "
            "Trim the audio and try again."
        )


def _normalize_loudness(wav_path: str):
    """Normalize audio loudness to -20 LUFS in-place."""
    audio, sr = torchaudio.load(wav_path)
    loudness = torchaudio.functional.loudness(audio, sr)
    # loudness returns per-channel tensor — average to scalar
    loudness_val = loudness.mean().item()
    normalized_audio = torchaudio.functional.gain(audio, -20 - loudness_val)
    torchaudio.save(wav_path, normalized_audio, sr)


def _load_patcher(model_name, device, torch_compile, dtype="auto", optimize=False):
    """Get or create a patcher for the given model/device/config.
    Fully unloads any existing patchers with a different config to free VRAM/RAM."""
    if device == "cuda":
        load_device = model_management.get_torch_device()
        offload_device = model_management.intermediate_device()
    else:
        load_device = torch.device("cpu")
        offload_device = torch.device("cpu")

    cache_key = f"{model_name}_{device}_opt{optimize}_compile{torch_compile}_dtype{dtype}"

    # Fast path: already cached with the right key, skip eviction
    if cache_key in VOXCPM_PATCHER_CACHE:
        return VOXCPM_PATCHER_CACHE[cache_key]

    # Evict all cached patchers that don't match the requested config
    stale_keys = [k for k in VOXCPM_PATCHER_CACHE if k != cache_key]
    for key in stale_keys:
        old_patcher = VOXCPM_PATCHER_CACHE.pop(key)
        old_patcher.force_unload()

    if cache_key not in VOXCPM_PATCHER_CACHE:
        handler = VoxCPMModelHandler(model_name, optimize=optimize, torch_compile=torch_compile, dtype=dtype)
        patcher = VoxCPMPatcher(handler, load_device=load_device, offload_device=offload_device, size=handler.size)
        VOXCPM_PATCHER_CACHE[cache_key] = patcher

    return VOXCPM_PATCHER_CACHE[cache_key]


class VoxCPM2TTSNode(io.ComfyNode):
    """VoxCPM2 Text-to-Speech node. Supports voice design via parenthetical descriptions."""
    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())

        available_devices = get_available_devices()
        default_device = available_devices[0]

        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id="VoxCPM2_TTS",
            display_name="VoxCPM2 文本转语音",
            category=cls.CATEGORY,
            description="用任意 VoxCPM 模型把文字转成语音。支持在原文前加括号描述来设计音色，例如“（温柔的女声）你好，世界”。",
            inputs=[
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="选择要使用的 VoxCPM 模型。VoxCPM2 效果最好，0.5B 更快更省显存。"),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="从 models/loras 加载 LoRA 微调权重。保持 None 表示不启用。"),
                io.String.Input("voice_description", multiline=True, default="", tooltip="音色设计描述（可选）。例如“年轻女性，声音温柔甜美”。会被自动加括号并放到正文前面，影响整体音色。"),
                io.String.Input("text", multiline=True, default="Hello, welcome to VoxCPM!", tooltip="要合成的正文内容。这里直接写台词即可。"),
                io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="引导强度（CFG）。数值越大越贴合提示描述、越稳定；越小变化越自然但更容易跑偏。一般 1.5~2.5 之间。"),
                io.Int.Input("inference_timesteps", default=10, min=1, max=100, step=1, tooltip="推理步数。步数越多音质越好但越慢；10~16 步通常是性价比区间。"),
                io.Int.Input("max_tokens", default=4096, min=64, max=8192, tooltip="最大生成长度（token）。限制单次能生成多长的音频，长句被截断时调大它。"),
                io.Boolean.Input("normalize_text", default=True, label_on="规范化", label_off="原文", tooltip="自动处理数字、缩写和标点，让读法更自然。若你输入的是音素或不想被改写，请切到“原文”。"),
                io.Int.Input("seed", default=42, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="随机种子。固定数值可复现同一结果；填 -1 表示每次运行都随机。"),
                io.Boolean.Input("force_offload", default=False, label_on="强制卸载", label_off="自动", tooltip="生成结束后把模型从显存和内存中完全卸载，释放内存给其他节点使用。"),
                io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto", tooltip="模型精度。auto 会自动选最合适的一种（新显卡用 bf16，老显卡回退 fp16）。"),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="推理使用的设备，一般选 CUDA 显卡；显存不足时可换 CPU（会很慢）。"),
                io.Boolean.Input("torch_compile", default=False, label_on="编译加速", label_off="标准", tooltip="开启 torch.compile 加速。第一次运行需要编译、会比较慢，之后的运行会明显更快。"),
            ],
            outputs=[
                io.Audio.Output(display_name="生成音频"),
            ],
        )

    @classmethod
    def execute(cls, model_name, lora_name, device, text, cfg_value, inference_timesteps,
                max_tokens, normalize_text, seed, force_offload,
                torch_compile,
                voice_description="", dtype="auto", **kwargs):

        # Prepend voice description in parentheses if provided
        if voice_description and voice_description.strip():
            text = f"({voice_description.strip()}){text}"

        patcher = _load_patcher(model_name, device, torch_compile, dtype)
        model_management.load_model_gpu(patcher)
        voxcpm_model = patcher.model.model

        if not voxcpm_model:
            raise RuntimeError(f"Failed to load model '{model_name}'.")

        # LoRA handling
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                raise FileNotFoundError(f"LoRA file not found: {lora_name}")
            try:
                voxcpm_model.load_lora(lora_path)
                voxcpm_model.set_lora_enabled(True)
            except Exception as e:
                raise RuntimeError(f"Failed to load LoRA '{lora_name}': {e}")
        else:
            voxcpm_model.set_lora_enabled(False)

        set_seed(seed)

        try:
            wav_array = voxcpm_model.generate(
                text=text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_len=max_tokens,
                normalize=normalize_text,
            )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            output_sr = voxcpm_model.tts_model.sample_rate
            output_audio = {"waveform": output_tensor, "sample_rate": output_sr}

            logger.info(f"VoxCPM TTS generation complete ({model_name}).")

            if force_offload:
                cache_key = f"{model_name}_{device}_opt{patcher.model.optimize}_compile{torch_compile}_dtype{dtype}"
                patcher.force_unload()
                VOXCPM_PATCHER_CACHE.pop(cache_key, None)
                offload_asr()

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except Exception as e:
            logger.error(f"VoxCPM2 TTS error: {e}")
            raise e


class VoxCPM2CloneNode(io.ComfyNode):
    """VoxCPM2 Voice Cloning node. Supports controllable and ultimate cloning."""
    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())

        available_devices = get_available_devices()
        default_device = available_devices[0]

        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id="VoxCPM2_Clone",
            display_name="VoxCPM2 音色克隆",
            category=cls.CATEGORY,
            description="用参考音频克隆说话人的音色。VoxCPM2 支持两种方式：可控克隆（只给参考音频）和极致克隆（参考音频 + 台词文本）。V1 模型必须提供台词文本。",
            inputs=[
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="选择要使用的 VoxCPM 模型。只有 VoxCPM2 支持不带台词的“可控克隆”。"),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="从 models/loras 加载 LoRA 微调权重。保持 None 表示不启用。"),
                io.String.Input("voice_description", multiline=True, default="", tooltip="语气/风格描述（可选）。例如“语速稍快，语气轻快”。会被加括号放到正文前。注意：走极致克隆（填了台词或开了自动转写）时该项会被忽略。"),
                io.String.Input("text", multiline=True, default="This is a cloned voice generated by VoxCPM.", tooltip="要让克隆音色朗读的正文内容。"),
                io.Audio.Input("reference_audio", tooltip="参考音频：用于模仿音色的样本，建议 5~15 秒干净人声，不要有背景音乐。参考音频末尾留 0.5~1 秒静音，可以明显减少生成开头的杂音。"),
                io.String.Input("prompt_text", multiline=True, tooltip="参考音频对应的台词文本（要逐字对得上）。填写后进入极致克隆，相似度最高；留空则走可控克隆（仅 VoxCPM2 支持）。"),
                io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="引导强度（CFG）。数值越大越贴合提示描述、越稳定；越小变化越自然但更容易跑偏。一般 1.5~2.5 之间。"),
                io.Int.Input("inference_timesteps", default=10, min=1, max=100, step=1, tooltip="推理步数。步数越多音质越好但越慢；10~16 步通常是性价比区间。"),
                io.Int.Input("max_tokens", default=4096, min=64, max=8192, tooltip="最大生成长度（token）。限制单次能生成多长的音频，长句被截断时调大它。"),
                io.Boolean.Input("normalize_text", default=True, label_on="规范化", label_off="原文", tooltip="自动处理数字、缩写和标点，让读法更自然。若你输入的是音素或不想被改写，请切到“原文”。"),
                io.Boolean.Input("enable_denoiser", default=False, label_on="降噪", label_off="关闭", tooltip="克隆前先对参考音频做降噪处理，适合有底噪的录音。需要安装 modelscope 依赖；开启后还会顺带做一次响度归一。"),
                io.Int.Input("seed", default=42, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="随机种子。固定数值可复现同一结果；填 -1 表示每次运行都随机。"),
                io.Boolean.Input("force_offload", default=False, label_on="强制卸载", label_off="自动", tooltip="生成结束后把模型从显存和内存中完全卸载，释放内存给其他节点使用。"),
                io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto", tooltip="模型精度。auto 会自动选最合适的一种（新显卡用 bf16，老显卡回退 fp16）。"),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="推理使用的设备，一般选 CUDA 显卡；显存不足时可换 CPU（会很慢）。"),
                io.Boolean.Input("enable_asr", default=False, label_on="自动转写", label_off="关闭", tooltip="自动把参考音频转写成台词，省去手动输入。需要安装 funasr 依赖，首次运行会下载约 400MB 模型。已经手填台词时该项不生效。"),
                io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="异常重试次数。生成出乱语、空白等坏结果时自动重来几次，填 0 表示不重试。"),
                io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="坏结果判定阈值。按“音频时长与文本长度之比”判断是否算失败：生成明显过短或过长时触发重试。"),
                io.Boolean.Input("torch_compile", default=False, label_on="编译加速", label_off="标准", tooltip="开启 torch.compile 加速。第一次运行需要编译、会比较慢，之后的运行会明显更快。"),
            ],
            outputs=[
                io.Audio.Output(display_name="克隆音频"),
            ],
        )

    @classmethod
    def execute(cls, model_name, lora_name, device, text, cfg_value, inference_timesteps,
                max_tokens, normalize_text, enable_denoiser, seed, force_offload,
                enable_asr, retry_max_attempts, retry_threshold, torch_compile,
                reference_audio=None, prompt_text=None, voice_description="", dtype="auto", **kwargs):

        # Voice description handling
        has_voice_desc = bool(voice_description and voice_description.strip())
        has_prompt = bool(prompt_text and prompt_text.strip())

        if has_voice_desc and (has_prompt or enable_asr):
            logger.warning("voice_description is ignored in Ultimate Cloning mode (prompt_text/ASR provided).")
        elif has_voice_desc:
            logger.info(f"Controllable Clone — voice_description: \"{voice_description.strip()}\"")
            text = f"({voice_description.strip()}){text}"

        if reference_audio is None:
            raise ValueError("Reference audio is required for voice cloning.")

        # Save reference audio to temp file for duration check and later use
        ref_waveform = reference_audio['waveform']
        ref_sample_rate = reference_audio['sample_rate']
        ref_wav_path = _save_audio_to_temp(ref_waveform, ref_sample_rate)

        # Validate reference audio duration
        _validate_reference_audio_duration(ref_wav_path)

        # ASR auto-transcription: only when no prompt_text is provided
        if enable_asr and (not prompt_text or not prompt_text.strip()):
            logger.info("Running ASR on reference audio...")
            try:
                prompt_text = transcribe_audio(ref_wav_path)
                if prompt_text:
                    logger.info(f"ASR result: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
                else:
                    logger.warning("ASR returned empty transcript. Proceeding without transcript.")
            except Exception as e:
                logger.warning(f"ASR failed: {e}. Proceeding without transcript.")
        elif enable_asr and prompt_text and prompt_text.strip():
            logger.info("ASR enabled but prompt_text already provided — using manual transcript instead.")

        patcher = _load_patcher(model_name, device, torch_compile, dtype)
        model_management.load_model_gpu(patcher)
        voxcpm_model = patcher.model.model

        if not voxcpm_model:
            raise RuntimeError(f"Failed to load model '{model_name}'.")

        is_v2 = voxcpm_model._is_v2

        # V1 models require prompt_text for voice cloning (no controllable mode)
        has_transcript = prompt_text and prompt_text.strip()
        if not is_v2 and not has_transcript:
            raise ValueError(
                f"Model '{model_name}' does not support controllable cloning (no transcript). "
                "Provide prompt_text or enable ASR for ultimate cloning, or use a VoxCPM2 model."
            )

        # LoRA handling
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                raise FileNotFoundError(f"LoRA file not found: {lora_name}")
            try:
                voxcpm_model.load_lora(lora_path)
                voxcpm_model.set_lora_enabled(True)
            except Exception as e:
                raise RuntimeError(f"Failed to load LoRA '{lora_name}': {e}")
        else:
            voxcpm_model.set_lora_enabled(False)

        set_seed(seed)

        try:
            if has_transcript and is_v2:
                # V2 ultimate cloning: reference + prompt + text
                wav_array = voxcpm_model.generate(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_wav_path=ref_wav_path,
                    reference_wav_path=ref_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_tokens,
                    normalize=normalize_text,
                    denoise=enable_denoiser,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )
            elif has_transcript and not is_v2:
                # V1 cloning: prompt_wav + prompt_text only (no reference_wav_path)
                wav_array = voxcpm_model.generate(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_wav_path=ref_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_tokens,
                    normalize=normalize_text,
                    denoise=enable_denoiser,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )
            else:
                # V2 controllable cloning: reference only
                wav_array = voxcpm_model.generate(
                    text=text,
                    reference_wav_path=ref_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_tokens,
                    normalize=normalize_text,
                    denoise=enable_denoiser,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            output_sr = voxcpm_model.tts_model.sample_rate

            # Loudness normalization when denoiser is active
            if enable_denoiser:
                norm_tmp = None
                try:
                    norm_tmp = _save_audio_to_temp(output_tensor, output_sr)
                    _normalize_loudness(norm_tmp)
                    norm_waveform, _ = torchaudio.load(norm_tmp)
                    output_tensor = norm_waveform.unsqueeze(0)
                except Exception as e:
                    logger.warning(f"Loudness normalization failed: {e}. Using raw output.")
                finally:
                    if norm_tmp:
                        try:
                            os.unlink(norm_tmp)
                        except OSError:
                            pass

            output_audio = {"waveform": output_tensor, "sample_rate": output_sr}

            logger.info(f"VoxCPM voice cloning complete ({model_name}).")

            if force_offload:
                cache_key = f"{model_name}_{device}_opt{patcher.model.optimize}_compile{torch_compile}_dtype{dtype}"
                patcher.force_unload()
                VOXCPM_PATCHER_CACHE.pop(cache_key, None)
                offload_asr()

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except Exception as e:
            logger.error(f"VoxCPM2 clone error: {e}")
            raise e

        finally:
            # Always clean up the temp file
            try:
                os.unlink(ref_wav_path)
            except OSError:
                pass


class VoxCPMExtension(ComfyExtension):
    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        from .voxcpm2_srt_nodes import VoxCPM2SRTBatchTTSNode, VoxCPM2SRTFolderBatchTTSNode, VoxCPM2SRTParserNode, VoxCPM2SRTSingleLineTTSNode

        nodes = [
            VoxCPM2TTSNode,
            VoxCPM2CloneNode,
            VoxCPM2SRTParserNode,
            VoxCPM2SRTBatchTTSNode,
            VoxCPM2SRTFolderBatchTTSNode,
            VoxCPM2SRTSingleLineTTSNode,
        ]
        if _TRAINING_NODES:
            nodes.extend(_TRAINING_NODES)
        else:
            logger.info("VoxCPM2 training nodes disabled. Set VOXCPM2_ENABLE_TRAINING_NODES=1 to enable them.")
        return nodes

async def comfy_entrypoint() -> VoxCPMExtension:
    return VoxCPMExtension()
