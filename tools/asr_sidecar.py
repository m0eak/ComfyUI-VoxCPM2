#!/usr/bin/env python3
"""Loopback-only ASR sidecar for ComfyUI-VoxCPM2.

Why this exists
---------------
The plugin's legacy ASR path calls ``funasr`` inside the ComfyUI process.  On
hosts where ComfyUI runs Python 3.13 that is not installable (``funasr``'s
``editdistance`` dependency has no wheel and fails to build), and the ComfyUI
environment must not be modified.  This script runs the *same* SenseVoiceSmall
model behind a tiny HTTP endpoint on a separate Python 3.12 interpreter, so the
ComfyUI side only needs the standard library.

Contract
--------
``POST /transcribe`` accepts either

* ``multipart/form-data`` with an ``file`` part holding audio bytes, or
* ``application/json`` with ``{"audio_base64": "...", "filename": "ref.wav"}``.

It answers ``{"text": "...", "backend": "...", "elapsed_ms": 123}``.
``GET /health`` reports readiness.

The service binds ``127.0.0.1`` only, refuses any other bind address, never
downloads a model and never touches the network.  Point it at a model directory
that was fetched once, out of band.

Usage
-----
    python tools/asr_sidecar.py --model-dir D:\\asr-sidecar\\models\\SenseVoiceSmall

Requires (in a Python 3.12 environment): ``funasr``, ``torch``, ``torchaudio``,
``soundfile``, ``modelscope``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("voxcpm2.asr_sidecar")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MAX_UPLOAD_MB = 64
DEFAULT_TIMEOUT_SECONDS = 120

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^;]+))', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SenseVoiceEngine:
    """Lazy, thread-safe wrapper around a local FunASR SenseVoiceSmall model.

    Two independent locks:

    * ``_lock`` guards the one-shot model load (double-checked locking).
    * ``_inference_lock`` serialises ``model.generate()``.  FunASR models keep
      per-call state on the instance and are not safe to run concurrently, so
      every inference takes this lock even though requests arrive in parallel
      threads.  Temp-file I/O stays *outside* it so concurrent requests can
      prepare their payload while another request is transcribing.
    """

    def __init__(self, model_dir: str, device: str = "cpu", language: str = "auto", use_itn: bool = True) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.language = language
        self.use_itn = bool(use_itn)
        self._model = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from funasr import AutoModel  # imported late so --help works without deps

            LOGGER.info("Loading FunASR model from %s on %s ...", self.model_dir, self.device)
            started = time.perf_counter()
            self._model = AutoModel(
                model=str(self.model_dir),
                device=self.device,
                disable_update=True,
                log_level="INFO",
            )
            LOGGER.info("Model ready in %.1fs.", time.perf_counter() - started)
            return self._model

    def warmup(self) -> None:
        self._load()

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        model = self._load()
        suffix = suffix if suffix.startswith(".") and suffix[1:].isalnum() else ".wav"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="voxcpm2_asr_")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(audio_bytes)
            # Serialise inference only. The lock is taken as late as possible so
            # the temp-file write above can overlap with someone else's generate().
            with self._inference_lock:
                result = model.generate(
                    input=tmp_path,
                    language=self.language,
                    use_itn=self.use_itn,
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not result:
            return ""
        first = result[0]
        raw = first.get("text", "") if isinstance(first, dict) else str(first)
        return _SPECIAL_TOKEN_RE.sub("", str(raw)).strip()


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def _extract_boundary(content_type: str) -> Optional[str]:
    match = _BOUNDARY_RE.search(content_type or "")
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip() or None


def parse_multipart_audio(body: bytes, content_type: str, field_name: str = "file") -> tuple[bytes, str]:
    """Return ``(audio_bytes, filename)`` from a multipart body.

    Deliberately tiny: only the first matching part is used, and only the byte
    payload is read — no filenames from the request are ever opened on disk.
    """
    boundary = _extract_boundary(content_type)
    if not boundary:
        raise ValueError("multipart request is missing a boundary parameter")

    delimiter = b"--" + boundary.encode("latin-1")
    for raw_part in body.split(delimiter)[1:]:
        if raw_part.startswith(b"--"):
            break
        part = raw_part[2:] if raw_part.startswith(b"\r\n") else raw_part
        header_blob, separator, data = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers = header_blob.decode("utf-8", "replace")
        if f'name="{field_name}"' not in headers:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        match = re.search(r'filename="([^"]*)"', headers)
        filename = match.group(1) if match else "audio.wav"
        return data, os.path.basename(filename) or "audio.wav"

    raise ValueError(f"multipart request has no '{field_name}' part")


def _suffix_of(filename: str) -> str:
    suffix = Path(str(filename or "audio.wav")).suffix.lower()
    if not suffix or len(suffix) > 8 or not suffix[1:].isalnum():
        return ".wav"
    return suffix


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class ASRSidecarHandler(BaseHTTPRequestHandler):
    server_version = "VoxCPM2ASRSidecar/1.0"
    protocol_version = "HTTP/1.1"

    engine: SenseVoiceEngine = None  # type: ignore[assignment]
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib signature
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            raise ValueError("invalid Content-Length header")
        if length <= 0:
            raise ValueError("empty request body")
        if length > self.max_upload_bytes:
            raise ValueError(
                f"payload of {length} bytes exceeds the {self.max_upload_bytes} byte limit"
            )
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("truncated request body")
        return body

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") in ("/health", "/healthz"):
            model_dir = getattr(self.engine, "model_dir", None)
            self._send_json(
                {
                    "status": "ok",
                    "backend": "sidecar:funasr",
                    "device": getattr(self.engine, "device", "unknown"),
                    "model_dir": str(model_dir) if model_dir else "",
                    "loaded": getattr(self.engine, "_model", None) is not None,
                }
            )
            return
        self._send_json({"error": f"unknown path {self.path}"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") != "/transcribe":
            self._send_json({"error": f"unknown path {self.path}"}, HTTPStatus.NOT_FOUND)
            return

        started = time.perf_counter()
        content_type = self.headers.get("Content-Type", "") or ""
        try:
            body = self._read_body()
            if content_type.lower().startswith("multipart/form-data"):
                audio_bytes, filename = parse_multipart_audio(body, content_type)
            elif content_type.lower().startswith("application/json"):
                payload = json.loads(body.decode("utf-8"))
                encoded = payload.get("audio_base64")
                if not isinstance(encoded, str) or not encoded.strip():
                    raise ValueError("JSON body requires a non-empty 'audio_base64' field")
                audio_bytes = base64.b64decode(encoded, validate=True)
                filename = str(payload.get("filename") or "audio.wav")
            else:
                raise ValueError(
                    "unsupported Content-Type; use multipart/form-data or application/json"
                )
        except (ValueError, binascii.Error, json.JSONDecodeError) as exc:
            self._send_json({"error": f"bad request: {exc}"}, HTTPStatus.BAD_REQUEST)
            return

        if not audio_bytes:
            self._send_json({"error": "empty audio payload"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            text = self.engine.transcribe_bytes(audio_bytes, _suffix_of(filename))
        except Exception as exc:  # noqa: BLE001 - report, never kill the server
            LOGGER.exception("Transcription failed")
            self._send_json(
                {"error": f"transcription failed: {exc}", "backend": "sidecar:funasr"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info("Transcribed %d bytes in %dms (%d chars).", len(audio_bytes), elapsed_ms, len(text))
        self._send_json({"text": text, "backend": "sidecar:funasr", "elapsed_ms": elapsed_ms})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loopback-only SenseVoiceSmall ASR sidecar for ComfyUI-VoxCPM2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"bind port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("VOXCPM2_ASR_MODEL_DIR", ""),
        help="local SenseVoiceSmall directory (or set VOXCPM2_ASR_MODEL_DIR). Never downloaded automatically.",
    )
    parser.add_argument("--device", default="cpu", help="funasr device, e.g. cpu or cuda:0 (default: cpu)")
    parser.add_argument("--language", default="auto", help="funasr language hint (default: auto)")
    parser.add_argument("--no-itn", action="store_true", help="disable inverse text normalization")
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=DEFAULT_MAX_UPLOAD_MB,
        help=f"maximum accepted upload size in MiB (default: {DEFAULT_MAX_UPLOAD_MB})",
    )
    parser.add_argument("--no-preload", action="store_true", help="load the model on first request instead of at startup")
    parser.add_argument("--log-level", default="INFO", help="logging level (default: INFO)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if str(args.host).strip().lower() not in _LOOPBACK_HOSTS:
        LOGGER.error(
            "Refusing to bind %s: this sidecar is loopback-only by design. Use --host 127.0.0.1.",
            args.host,
        )
        return 2

    model_dir = str(args.model_dir or "").strip()
    if not model_dir:
        LOGGER.error(
            "No model directory given. Pass --model-dir <path to SenseVoiceSmall> or set "
            "VOXCPM2_ASR_MODEL_DIR. This sidecar never downloads models."
        )
        return 2
    if not Path(model_dir).is_dir() or not any(Path(model_dir).iterdir()):
        LOGGER.error(
            "Model directory %r is missing or empty. Point it at a prepared SenseVoiceSmall "
            "snapshot (config.yaml + model.safetensors + tokens.json ...).",
            model_dir,
        )
        return 2

    try:
        import funasr  # noqa: F401
    except ImportError as exc:
        LOGGER.error(
            "funasr is not installed in this interpreter (%s).\n"
            "Create a Python 3.12 environment and install: funasr torch torchaudio soundfile modelscope",
            exc,
        )
        return 1

    engine = SenseVoiceEngine(
        model_dir=model_dir,
        device=str(args.device),
        language=str(args.language),
        use_itn=not bool(args.no_itn),
    )
    if not args.no_preload:
        try:
            engine.warmup()
        except Exception as exc:  # noqa: BLE001 - startup failure must be loud
            LOGGER.error("Failed to load the ASR model from %s: %s", model_dir, exc)
            return 1

    ASRSidecarHandler.engine = engine
    ASRSidecarHandler.max_upload_bytes = max(1, int(args.max_upload_mb)) * 1024 * 1024

    try:
        server = ThreadingHTTPServer((str(args.host), int(args.port)), ASRSidecarHandler)
    except OSError as exc:
        LOGGER.error("Cannot bind %s:%s (%s). Is the port already in use?", args.host, args.port, exc)
        return 1

    server.daemon_threads = True
    host, port = server.server_address[0], server.server_address[1]
    LOGGER.info("ASR sidecar listening on http://%s:%s (POST %s)", host, port, "/transcribe")
    LOGGER.info("Health check: curl http://%s:%s/health", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
