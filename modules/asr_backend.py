"""ASR backend adapter shared by the VoxCPM2 clone and SRT nodes.

The plugin historically transcribed reference audio in-process with ``funasr``
(see ``voxcpm2_nodes.transcribe_audio``).  That path cannot be installed on every
host — most notably Windows + Python 3.13, where ``funasr``'s transitive
``editdistance`` dependency has no prebuilt wheel and fails to build.

This module keeps the legacy path untouched and adds an HTTP transport that
delegates transcription to a separate, loopback-only sidecar process
(``tools/asr_sidecar.py``) running on a Python version that does have wheels.

Design rules:

* No ComfyUI imports here, so the module stays importable and testable outside
  a ComfyUI runtime.
* ``http`` never degrades silently: every transport or parsing problem raises.
* ``auto`` is allowed to fall through from ``embedded`` to ``http``, but
  :attr:`ASRTranscriber.last_backend` always records which backend actually
  produced the transcript so callers can log and persist it.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ASR_BACKEND_EMBEDDED = "embedded"
ASR_BACKEND_HTTP = "http"
ASR_BACKEND_AUTO = "auto"

#: Ordered options for the ``asr_backend`` combo input.  ``embedded`` first so the
#: rendered default matches the historical behaviour.
ASR_BACKEND_OPTIONS = [ASR_BACKEND_EMBEDDED, ASR_BACKEND_HTTP, ASR_BACKEND_AUTO]

DEFAULT_ASR_SERVER_URL = "http://127.0.0.1:8765"
DEFAULT_ASR_HTTP_TIMEOUT_SECONDS = 180.0

TRANSCRIBE_PATH = "/transcribe"
HEALTH_PATH = "/health"

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class ASRError(RuntimeError):
    """Base class for every ASR adapter failure."""


class ASRBackendError(ASRError):
    """A single backend failed while transcribing."""


class ASRUnavailableError(ASRError):
    """No configured backend could produce a transcript."""


# ---------------------------------------------------------------------------
# Option normalisation
# ---------------------------------------------------------------------------


def normalize_asr_backend(value: object) -> str:
    """Return a supported backend name, falling back to ``embedded``.

    Unknown values fall back to the legacy behaviour instead of raising, so an
    old workflow that somehow carries a stale value keeps working.
    """
    text = str(value or "").strip().lower()
    if text in ASR_BACKEND_OPTIONS:
        return text
    if text:
        logger.warning("[ASR] Unknown asr_backend=%r, falling back to %s.", value, ASR_BACKEND_EMBEDDED)
    return ASR_BACKEND_EMBEDDED


def normalize_asr_server_url(value: object) -> str:
    """Normalise the sidecar base URL, adding a scheme when omitted."""
    text = str(value or "").strip()
    if not text:
        text = DEFAULT_ASR_SERVER_URL
    if "://" not in text:
        text = "http://" + text
    return text.rstrip("/")


def is_loopback_url(url: str) -> bool:
    """Return True when the URL host is a loopback address."""
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return False
    return host.strip().lower() in _LOOPBACK_HOSTS


def clean_transcript(text: object) -> str:
    """Strip SenseVoice special tokens and surrounding whitespace."""
    if not isinstance(text, str):
        return ""
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


def build_multipart_body(
    audio_bytes: bytes,
    *,
    field_name: str = "file",
    filename: str = "audio.wav",
    boundary: Optional[str] = None,
) -> tuple[bytes, str]:
    """Build a minimal multipart/form-data body for a single audio part.

    Only the audio *bytes* travel over the wire — the local filesystem path is
    never sent, so the sidecar cannot be tricked into reading arbitrary files.
    """
    if not audio_bytes:
        raise ASRBackendError("Refusing to upload an empty audio payload to the ASR sidecar.")
    boundary = boundary or "----VoxCPM2ASR7d1c3f5b9a"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n"
        "\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + audio_bytes + tail, boundary


def _upload_filename(wav_path: str) -> str:
    """Return a path-free upload filename derived only from the file suffix."""
    suffix = Path(str(wav_path)).suffix.lower()
    if not suffix or len(suffix) > 8 or not suffix[1:].isalnum():
        suffix = ".wav"
    return "audio" + suffix


def transcribe_via_http(
    wav_path: str,
    server_url: str = DEFAULT_ASR_SERVER_URL,
    *,
    timeout: float = DEFAULT_ASR_HTTP_TIMEOUT_SECONDS,
) -> str:
    """POST ``wav_path`` to the sidecar and return the transcript.

    Raises :class:`ASRBackendError` for every failure — connection refused,
    timeout, non-2xx status or an unexpected payload shape.
    """
    base_url = normalize_asr_server_url(server_url)
    if not is_loopback_url(base_url):
        logger.warning(
            "[ASR] Sidecar URL %s is not a loopback address. The sidecar is meant to stay on 127.0.0.1.",
            base_url,
        )

    source = Path(str(wav_path))
    try:
        audio_bytes = source.read_bytes()
    except OSError as exc:
        raise ASRBackendError(f"Cannot read reference audio for ASR upload: {source} ({exc})") from exc

    body, boundary = build_multipart_body(audio_bytes, filename=_upload_filename(str(source)))
    endpoint = base_url + TRANSCRIBE_PATH
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
            "User-Agent": "ComfyUI-VoxCPM2-ASR",
        },
    )

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace").strip()
        except Exception:  # pragma: no cover - best effort only
            detail = ""
        suffix = f" Body: {detail[:400]}" if detail else ""
        raise ASRBackendError(
            f"ASR sidecar at {endpoint} returned HTTP {exc.code}.{suffix}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ASRBackendError(
            f"Cannot reach the ASR sidecar at {endpoint} ({exc.reason}). "
            "Start it with: python tools/asr_sidecar.py --model-dir <SenseVoiceSmall dir>"
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise ASRBackendError(
            f"ASR sidecar request to {endpoint} failed after {timeout:.0f}s: {exc}"
        ) from exc

    try:
        status = int(getattr(response, "status", 200) or 200)
        raw = response.read()
    finally:
        try:
            response.close()
        except Exception:  # pragma: no cover - close is best effort
            pass

    if status >= 400:
        raise ASRBackendError(
            f"ASR sidecar at {endpoint} returned HTTP {status}. "
            f"Body: {raw[:400].decode('utf-8', 'replace')}"
        )

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ASRBackendError(
            f"ASR sidecar at {endpoint} returned a non-JSON response: {raw[:200]!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise ASRBackendError(
            f"ASR sidecar at {endpoint} returned {type(payload).__name__}, expected a JSON object."
        )
    if "text" not in payload:
        error = payload.get("error")
        if error:
            raise ASRBackendError(f"ASR sidecar at {endpoint} reported an error: {error}")
        raise ASRBackendError(
            f"ASR sidecar at {endpoint} response has no 'text' field: {str(payload)[:200]}"
        )
    if not isinstance(payload["text"], str):
        raise ASRBackendError(
            f"ASR sidecar at {endpoint} returned a non-string 'text' field: {payload['text']!r}"
        )

    return clean_transcript(payload["text"])


# ---------------------------------------------------------------------------
# Transcriber facade
# ---------------------------------------------------------------------------


class ASRTranscriber:
    """Callable adapter used by the clone and SRT nodes.

    Instances are callables so they can be passed straight into
    :func:`modules.srt_generation.prepare_reference_audio`, which only expects a
    ``Callable[[str], str]``.
    """

    def __init__(
        self,
        backend: object = ASR_BACKEND_EMBEDDED,
        server_url: object = DEFAULT_ASR_SERVER_URL,
        *,
        embedded_transcriber: Optional[Callable[[str], str]] = None,
        http_timeout: float = DEFAULT_ASR_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self.requested_backend = normalize_asr_backend(backend)
        self.server_url = normalize_asr_server_url(server_url)
        self._embedded_transcriber = embedded_transcriber
        self._http_timeout = float(http_timeout)
        self._last_backend = ""
        self._last_error = ""

    # -- introspection ----------------------------------------------------
    @property
    def last_backend(self) -> str:
        """Backend that produced the most recent transcript ("" when unused)."""
        return self._last_backend

    @property
    def last_error(self) -> str:
        """Message of the most recent backend failure ("" when none)."""
        return self._last_error

    def describe(self) -> str:
        """Human readable one-line description of the configured backend."""
        if self.requested_backend == ASR_BACKEND_HTTP:
            return f"http ({self.server_url})"
        if self.requested_backend == ASR_BACKEND_AUTO:
            return f"auto (embedded -> http {self.server_url})"
        return "embedded"

    # -- legacy path ------------------------------------------------------
    def _resolve_embedded(self) -> Callable[[str], str]:
        if self._embedded_transcriber is not None:
            return self._embedded_transcriber
        # Imported lazily: keeps this module free of ComfyUI/torch imports and
        # avoids a circular import at package load time.
        from ..voxcpm2_nodes import transcribe_audio  # type: ignore

        return transcribe_audio

    def _transcribe_embedded(self, wav_path: str) -> str:
        try:
            return clean_transcript(self._resolve_embedded()(wav_path))
        except ASRError:
            raise
        except Exception as exc:
            raise ASRBackendError(
                f"Embedded FunASR transcription failed: {exc}. "
                "If this ComfyUI runs on Python 3.13 (no funasr/editdistance wheel), "
                "run tools/asr_sidecar.py on Python 3.12 and set asr_backend=http."
            ) from exc

    def _transcribe_http(self, wav_path: str) -> str:
        return transcribe_via_http(wav_path, self.server_url, timeout=self._http_timeout)

    # -- public API -------------------------------------------------------
    def transcribe(self, wav_path: str) -> str:
        """Transcribe ``wav_path`` using the configured backend.

        ``embedded`` and ``http`` raise on failure.  ``auto`` tries ``embedded``
        first and then ``http``, raising only when both fail.
        """
        backend = self.requested_backend
        self._last_backend = ""
        self._last_error = ""

        if backend == ASR_BACKEND_EMBEDDED:
            text = self._transcribe_embedded(wav_path)
            self._last_backend = ASR_BACKEND_EMBEDDED
            return text

        if backend == ASR_BACKEND_HTTP:
            text = self._transcribe_http(wav_path)
            self._last_backend = ASR_BACKEND_HTTP
            return text

        embedded_error = ""
        try:
            text = self._transcribe_embedded(wav_path)
            self._last_backend = ASR_BACKEND_EMBEDDED
            logger.info("[ASR] auto: embedded backend succeeded.")
            return text
        except Exception as exc:
            embedded_error = str(exc)
            logger.warning("[ASR] auto: embedded backend failed (%s); trying http sidecar.", exc)

        try:
            text = self._transcribe_http(wav_path)
            self._last_backend = ASR_BACKEND_HTTP
            logger.info("[ASR] auto: http sidecar backend succeeded at %s.", self.server_url)
            return text
        except Exception as exc:
            self._last_error = f"embedded: {embedded_error} | http: {exc}"
            raise ASRUnavailableError(
                "ASR failed on every configured backend.\n"
                f"  embedded: {embedded_error}\n"
                f"  http ({self.server_url}): {exc}\n"
                "Start tools/asr_sidecar.py on a Python 3.12 interpreter, or fill prompt_text manually."
            ) from exc

    def __call__(self, wav_path: str) -> str:
        return self.transcribe(wav_path)
