# ASR Sidecar：给 Python 3.13 的 ComfyUI 补回自动转写

## 这是什么

插件的自动转写（`enable_asr`）原本在 ComfyUI 进程内直接调用 `funasr`。在 Windows + Python 3.13 上这条路走不通：`funasr` 的传递依赖 `editdistance` 没有 3.13 的预编译 wheel，源码编译会因为没有 `python313t.lib` / 缺少 MSVC 而失败。

Sidecar 把这部分工作挪到**另一个 Python 3.12 进程**里：

```
ComfyUI (Python 3.13)                      ASR Sidecar (Python 3.12)
  nodes / srt nodes                          tools/asr_sidecar.py
        │                                            │
        │  POST http://127.0.0.1:8765/transcribe     │
        │  multipart 音频字节（不传路径）             │
        └───────────────────────────────────────────►│  funasr + SenseVoiceSmall
                                                     │
        ◄───────────────────────────────────────────┘
                     {"text": "..."}
```

好处：

- ComfyUI 的 Python 环境**完全不用动**，不装 funasr、不建 venv、不改依赖。
- 上游 VoxCPM 源码、ComfyUI 主程序、训练节点都不需要改。
- Sidecar 只监听 `127.0.0.1`，不进公网；上传的是音频字节，不是本机路径。

## 切换方式（ComfyUI 侧）

节点里新增两个输入，都追加在参数列表末尾，旧 workflow 不填时行为不变：

| 输入 | 默认值 | 说明 |
| --- | --- | --- |
| `asr_backend` | `embedded` | `embedded` 走进程内 funasr（旧行为）；`http` 走 Sidecar；`auto` 先 embedded，失败再 Sidecar |
| `asr_server_url` | `http://127.0.0.1:8765` | Sidecar 地址，仅 `http` / `auto` 使用 |

涉及的节点：

- `VoxCPM2 音色克隆`
- `VoxCPM2 SRT 批量配音`
- `VoxCPM2 SRT 单句配音`
- `VoxCPM2 SRT 目录批量配音`

公司机（ComfyUI 是 Python 3.13）建议：`asr_backend = http`。

## 部署步骤（公司机，仅列出，不执行）

### 1. 准备 Python 3.12

任选其一，装在 ComfyUI 之外的目录，例如 `D:\asr-sidecar\python312`：

- 官方安装包：<https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe>
- 或已有的 Python 3.12 / 3.11 解释器（3.10～3.12 都行，**不要用 3.13**）

### 2. 建一个独立环境

```bat
cd /d D:\asr-sidecar
D:\asr-sidecar\python312\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
```

### 3. 装 Sidecar 依赖（只装在 3.12 环境里）

```bat
.venv\Scripts\python.exe -m pip install funasr torch torchaudio soundfile modelscope
```

CPU 推理足够，不需要 CUDA 版 torch。

### 4. 预先准备好 SenseVoiceSmall 模型（Sidecar 不会自己下载）

```bat
.venv\Scripts\python.exe -m modelscope download --model iic/SenseVoiceSmall --local_dir D:\asr-sidecar\models\SenseVoiceSmall
```

> 也可以直接从已经跑过 embedded ASR 的机器上，把 `ComfyUI\models\audio_encoders\SenseVoiceSmall` 整个目录拷贝过来。

确认目录里有 `config.yaml`、`model.safetensors`、`tokens.json` 等文件。

### 5. 启动 Sidecar

```bat
D:\asr-sidecar\.venv\Scripts\python.exe ^
  D:\Comfyui\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-VoxCPM2\tools\asr_sidecar.py ^
  --model-dir D:\asr-sidecar\models\SenseVoiceSmall ^
  --device cpu --port 8765
```

首行日志应类似：

```
2026-10-22 ... [INFO] voxcpm2.asr_sidecar: ASR sidecar listening on http://127.0.0.1:8765 (POST /transcribe)
```

自检：

```bat
curl http://127.0.0.1:8765/health
```

### 6. 让 Sidecar 跟着开机（可选）

用「任务计划程序」新建任务，触发器选「登录时」，操作填第 5 步的命令，起始于 `D:\asr-sidecar`。

### 7. 回到 ComfyUI

1. 刷新浏览器（Ctrl+F5）
2. 在克隆 / SRT 节点里把 `asr_backend` 设为 `http`
3. 运行一次，确认 ComfyUI 控制台出现：

```
[ASR] ... backend: http (http://127.0.0.1:8765)
[ASR] ... transcript produced by http backend
```

SRT 任务的 `config.json` 里会多出：

```json
{ "asr_backend": "http", "asr_server_url": "http://127.0.0.1:8765", "asr_backend_used": "http" }
```

## 接口约定

`POST /transcribe`

- `multipart/form-data`，字段名 `file`，内容是音频字节；或
- `application/json`，`{"audio_base64": "...", "filename": "ref.wav"}`

返回：

```json
{ "text": "转写结果", "backend": "sidecar:funasr", "elapsed_ms": 420 }
```

`GET /health` → `{"status": "ok", "device": "cpu", "loaded": true, ...}`

失败时返回 `{"error": "..."}`，HTTP 400 / 500；ComfyUI 侧会把错误原样抛出来，不会静默降级。

## 故障排查

| 现象 | 处理 |
| --- | --- |
| `Cannot reach the ASR sidecar ... Connection refused` | Sidecar 没启动，或端口不一致 |
| `funasr is not installed in this interpreter` | Sidecar 用错了解释器，确认是 `.venv\Scripts\python.exe` |
| `No model directory given` / `Model directory ... is missing or empty` | 补 `--model-dir`，或设环境变量 `VOXCPM2_ASR_MODEL_DIR` |
| `Refusing to bind 0.0.0.0` | Sidecar 只允许回环地址，改回 `127.0.0.1` |
| `HTTP 500 ... transcription failed` | 看 Sidecar 窗口的完整堆栈，通常是模型文件不完整 |
| 转写结果为空 | 参考音频太短/纯静音；不会静默跳过，会直接报错 |

## 已知限制

- Sidecar 一次只加载一个 SenseVoiceSmall 实例，模型首次加载约数秒到数十秒（`--no-preload` 可以让它懒加载）。
- 单个请求有大小上限，默认 64 MiB（`--max-upload-mb`）。
- 只支持「整段转写」，不返回时间戳；SRT 流程只需要整段文本。
- 转写失败一律报错：`ultimate` 模式不会退化成 `controllable`，`auto` 只有在 embedded 和 http 都失败时才报错。
