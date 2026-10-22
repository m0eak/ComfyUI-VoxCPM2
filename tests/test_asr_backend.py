"""Unit tests for the ASR backend adapter and the sidecar's request parsing.

These tests run without ComfyUI, torch or funasr installed: the modules under
test are loaded straight from their file paths.

    python3 tests/test_asr_backend.py
"""

from __future__ import annotations

import importlib.util
import io
import sys
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


asr_backend = _load_module("voxcpm2_asr_backend_under_test", "modules/asr_backend.py")
asr_sidecar = _load_module("voxcpm2_asr_sidecar_under_test", "tools/asr_sidecar.py")
srt_generation = _load_module("voxcpm2_srt_generation_under_test", "modules/srt_generation.py")


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        self.closed = True


class MultipartBodyTests(unittest.TestCase):
    def test_body_carries_bytes_only_never_the_local_path(self):
        audio = b"RIFF....WAVEdata"
        body, boundary = asr_backend.build_multipart_body(
            audio,
            filename=asr_backend._upload_filename(r"D:\secret\reference.wav"),
        )

        self.assertIn(boundary.encode(), body)
        self.assertIn(b'name="file"', body)
        self.assertIn(b'filename="audio.wav"', body)
        self.assertIn(audio, body)
        self.assertNotIn(b"secret", body)
        self.assertNotIn(b"reference.wav", body)
        self.assertNotIn(b"D:", body)

    def test_empty_audio_is_rejected(self):
        with self.assertRaises(asr_backend.ASRBackendError):
            asr_backend.build_multipart_body(b"")

    def test_upload_filename_keeps_only_a_safe_suffix(self):
        self.assertEqual(asr_backend._upload_filename("C:/tmp/x.WAV"), "audio.wav")
        self.assertEqual(asr_backend._upload_filename("C:/tmp/x.flac"), "audio.flac")
        self.assertEqual(asr_backend._upload_filename("C:/tmp/x.weird-ext"), "audio.wav")


class OptionNormalisationTests(unittest.TestCase):
    def test_known_backends_are_kept(self):
        for value in ("embedded", "http", "auto", " HTTP "):
            self.assertIn(
                asr_backend.normalize_asr_backend(value),
                asr_backend.ASR_BACKEND_OPTIONS,
            )
        self.assertEqual(asr_backend.normalize_asr_backend(" HTTP "), "http")

    def test_unknown_backend_falls_back_to_embedded(self):
        self.assertEqual(asr_backend.normalize_asr_backend("whisper"), "embedded")
        self.assertEqual(asr_backend.normalize_asr_backend(None), "embedded")

    def test_server_url_normalisation(self):
        self.assertEqual(
            asr_backend.normalize_asr_server_url(""),
            asr_backend.DEFAULT_ASR_SERVER_URL,
        )
        self.assertEqual(
            asr_backend.normalize_asr_server_url("127.0.0.1:9999/"),
            "http://127.0.0.1:9999",
        )

    def test_loopback_detection(self):
        self.assertTrue(asr_backend.is_loopback_url("http://127.0.0.1:8765"))
        self.assertTrue(asr_backend.is_loopback_url("http://localhost:8765"))
        self.assertFalse(asr_backend.is_loopback_url("http://192.168.1.10:8765"))


class HttpTransportTests(unittest.TestCase):
    def _write_wav(self, tmp_path: Path) -> Path:
        path = tmp_path / "ref.wav"
        path.write_bytes(b"RIFFfakewavedata")
        return path

    def test_success_response_is_parsed_and_cleaned(self):
        wav_path = self._write_wav(Path(self._tmpdir()))
        response = _FakeResponse('{"text": " <|zh|>你好世界 <|NEUTRAL|>"}'.encode("utf-8"))

        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            text = asr_backend.transcribe_via_http(str(wav_path), "http://127.0.0.1:8765")

        self.assertEqual(text, "你好世界")
        self.assertTrue(response.closed)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8765/transcribe")
        self.assertEqual(request.get_method(), "POST")
        # The upload payload must contain audio bytes, not the local path.
        self.assertIn(b"RIFFfakewavedata", request.data)
        self.assertNotIn(str(wav_path).encode(), request.data)

    def test_http_error_status_raises_with_detail(self):
        wav_path = self._write_wav(Path(self._tmpdir()))
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8765/transcribe",
            500,
            "Internal Server Error",
            {},
            io.BytesIO(b'{"error": "model not loaded"}'),
        )

        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(asr_backend.ASRBackendError) as ctx:
                asr_backend.transcribe_via_http(str(wav_path))

        message = str(ctx.exception)
        self.assertIn("HTTP 500", message)
        self.assertIn("model not loaded", message)

    def test_unreachable_sidecar_raises_with_startup_hint(self):
        wav_path = self._write_wav(Path(self._tmpdir()))
        error = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(asr_backend.ASRBackendError) as ctx:
                asr_backend.transcribe_via_http(str(wav_path))

        message = str(ctx.exception)
        self.assertIn("Cannot reach the ASR sidecar", message)
        self.assertIn("tools/asr_sidecar.py", message)

    def test_response_without_text_field_raises(self):
        wav_path = self._write_wav(Path(self._tmpdir()))
        response = _FakeResponse(b'{"backend": "sidecar"}')

        with mock.patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(asr_backend.ASRBackendError) as ctx:
                asr_backend.transcribe_via_http(str(wav_path))

        self.assertIn("no 'text' field", str(ctx.exception))

    def test_non_json_response_raises(self):
        wav_path = self._write_wav(Path(self._tmpdir()))
        response = _FakeResponse(b"<html>oops</html>")

        with mock.patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(asr_backend.ASRBackendError):
                asr_backend.transcribe_via_http(str(wav_path))

    def test_empty_transcript_is_preserved_not_raised(self):
        wav_path = self._write_wav(Path(self._tmpdir()))
        response = _FakeResponse(b'{"text": ""}')

        with mock.patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(asr_backend.transcribe_via_http(str(wav_path)), "")

    @staticmethod
    def _tmpdir() -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="voxcpm2_asr_test_")


class TranscriberSelectionTests(unittest.TestCase):
    def setUp(self):
        self.wav_path = "/tmp/does-not-matter.wav"
        self.embedded_calls = []
        self.http_calls = []

    def _embedded(self, text: str):
        def _fn(wav_path: str) -> str:
            self.embedded_calls.append(wav_path)
            if isinstance(text, Exception):
                raise text
            return text

        return _fn

    def _fake_http(self, text):
        def _fn(wav_path, server_url="", *, timeout=0.0):
            self.http_calls.append((wav_path, server_url))
            if isinstance(text, Exception):
                raise text
            return text

        return _fn

    def test_default_backend_is_embedded_and_never_touches_http(self):
        transcriber = asr_backend.ASRTranscriber(embedded_transcriber=self._embedded(" 你好 "))
        with mock.patch.object(asr_backend, "transcribe_via_http", self._fake_http("nope")):
            text = transcriber.transcribe(self.wav_path)

        self.assertEqual(transcriber.requested_backend, "embedded")
        self.assertEqual(text, "你好")
        self.assertEqual(transcriber.last_backend, "embedded")
        self.assertEqual(self.embedded_calls, [self.wav_path])
        self.assertEqual(self.http_calls, [])

    def test_embedded_failure_propagates_instead_of_silently_degrading(self):
        transcriber = asr_backend.ASRTranscriber(
            "embedded",
            embedded_transcriber=self._embedded(ImportError("No module named 'funasr'")),
        )
        with mock.patch.object(asr_backend, "transcribe_via_http", self._fake_http("sidecar")):
            with self.assertRaises(asr_backend.ASRError) as ctx:
                transcriber.transcribe(self.wav_path)

        self.assertIn("asr_backend=http", str(ctx.exception))
        self.assertEqual(transcriber.last_backend, "")
        self.assertEqual(self.http_calls, [])

    def test_http_backend_uses_sidecar_only(self):
        transcriber = asr_backend.ASRTranscriber(
            "http",
            "http://127.0.0.1:9999",
            embedded_transcriber=self._embedded("embedded"),
        )
        with mock.patch.object(asr_backend, "transcribe_via_http", self._fake_http("来自 Sidecar")):
            text = transcriber.transcribe(self.wav_path)

        self.assertEqual(text, "来自 Sidecar")
        self.assertEqual(transcriber.last_backend, "http")
        self.assertEqual(self.embedded_calls, [])
        self.assertEqual(self.http_calls, [(self.wav_path, "http://127.0.0.1:9999")])

    def test_http_backend_failure_is_loud(self):
        transcriber = asr_backend.ASRTranscriber(
            "http",
            embedded_transcriber=self._embedded("embedded would have worked"),
        )
        failure = asr_backend.ASRBackendError("sidecar down")
        with mock.patch.object(asr_backend, "transcribe_via_http", self._fake_http(failure)):
            with self.assertRaises(asr_backend.ASRBackendError):
                transcriber.transcribe(self.wav_path)

        self.assertEqual(self.embedded_calls, [])
        self.assertEqual(transcriber.last_backend, "")

    def test_auto_falls_back_to_http_when_embedded_fails(self):
        transcriber = asr_backend.ASRTranscriber(
            "auto",
            embedded_transcriber=self._embedded(ImportError("No module named 'funasr'")),
        )
        with mock.patch.object(asr_backend, "transcribe_via_http", self._fake_http("sidecar text")):
            text = transcriber.transcribe(self.wav_path)

        self.assertEqual(text, "sidecar text")
        self.assertEqual(transcriber.last_backend, "http")
        self.assertEqual(len(self.embedded_calls), 1)

    def test_auto_prefers_embedded_when_it_works(self):
        transcriber = asr_backend.ASRTranscriber(
            "auto",
            embedded_transcriber=self._embedded("embedded text"),
        )
        with mock.patch.object(asr_backend, "transcribe_via_http", self._fake_http("sidecar")):
            text = transcriber.transcribe(self.wav_path)

        self.assertEqual(text, "embedded text")
        self.assertEqual(transcriber.last_backend, "embedded")
        self.assertEqual(self.http_calls, [])

    def test_auto_failure_reports_both_backends(self):
        transcriber = asr_backend.ASRTranscriber(
            "auto",
            embedded_transcriber=self._embedded(ImportError("No module named 'funasr'")),
        )
        with mock.patch.object(
            asr_backend, "transcribe_via_http", self._fake_http(asr_backend.ASRBackendError("refused"))
        ):
            with self.assertRaises(asr_backend.ASRUnavailableError) as ctx:
                transcriber.transcribe(self.wav_path)

        message = str(ctx.exception)
        self.assertIn("embedded:", message)
        self.assertIn("http (http://127.0.0.1:8765):", message)
        self.assertIn("funasr", message)
        self.assertIn("refused", message)
        self.assertEqual(transcriber.last_backend, "")

    def test_transcriber_is_callable_for_prepare_reference_audio(self):
        transcriber = asr_backend.ASRTranscriber(embedded_transcriber=self._embedded("ok"))
        self.assertEqual(transcriber("/tmp/x.wav"), "ok")


class SrtReferenceAudioTests(unittest.TestCase):
    """ASR failures must abort instead of silently degrading to voice cloning."""

    def _reference(self):
        return {"waveform": object(), "sample_rate": 16000}

    def test_ultimate_mode_propagates_asr_failure(self):
        def _boom(_path):
            raise asr_backend.ASRUnavailableError("all backends failed")

        with self.assertRaises(asr_backend.ASRUnavailableError):
            srt_generation.prepare_reference_audio(
                self._reference(),
                "ultimate",
                True,
                "",
                save_audio_to_temp=lambda *_: "/tmp/ref.wav",
                validate_reference_audio_duration=lambda _p: None,
                transcribe_audio=_boom,
            )

    def test_auto_mode_propagates_asr_failure(self):
        def _boom(_path):
            raise asr_backend.ASRUnavailableError("all backends failed")

        with self.assertRaises(asr_backend.ASRUnavailableError):
            srt_generation.prepare_reference_audio(
                self._reference(),
                "auto",
                True,
                "",
                save_audio_to_temp=lambda *_: "/tmp/ref.wav",
                validate_reference_audio_duration=lambda _p: None,
                transcribe_audio=_boom,
            )

    def test_controllable_mode_does_not_call_asr(self):
        calls = []

        def _spy(path):
            calls.append(path)
            return "should not run"

        path, prompt = srt_generation.prepare_reference_audio(
            self._reference(),
            "controllable",
            True,
            "",
            save_audio_to_temp=lambda *_: "/tmp/ref.wav",
            validate_reference_audio_duration=lambda _p: None,
            transcribe_audio=_spy,
        )

        self.assertEqual(path, "/tmp/ref.wav")
        self.assertEqual(prompt, "")
        self.assertEqual(calls, [])

    def test_manual_prompt_text_skips_asr(self):
        calls = []
        path, prompt = srt_generation.prepare_reference_audio(
            self._reference(),
            "ultimate",
            True,
            "手动台词",
            save_audio_to_temp=lambda *_: "/tmp/ref.wav",
            validate_reference_audio_duration=lambda _p: None,
            transcribe_audio=lambda p: calls.append(p) or "asr",
        )

        self.assertEqual(prompt, "手动台词")
        self.assertEqual(calls, [])


class SidecarParsingTests(unittest.TestCase):
    def test_multipart_audio_roundtrip(self):
        body, boundary = asr_backend.build_multipart_body(b"WAVE-BYTES", filename="audio.wav")
        audio, filename = asr_sidecar.parse_multipart_audio(
            body, f"multipart/form-data; boundary={boundary}"
        )

        self.assertEqual(audio, b"WAVE-BYTES")
        self.assertEqual(filename, "audio.wav")

    def test_multipart_without_file_part_raises(self):
        body = b'--b\r\nContent-Disposition: form-data; name="other"\r\n\r\nx\r\n--b--\r\n'
        with self.assertRaises(ValueError):
            asr_sidecar.parse_multipart_audio(body, "multipart/form-data; boundary=b")

    def test_boundary_extraction_handles_quotes(self):
        self.assertEqual(
            asr_sidecar._extract_boundary('multipart/form-data; boundary="abc123"'),
            "abc123",
        )
        self.assertIsNone(asr_sidecar._extract_boundary("application/json"))

    def test_suffix_sanitisation(self):
        self.assertEqual(asr_sidecar._suffix_of("ref.wav"), ".wav")
        self.assertEqual(asr_sidecar._suffix_of("ref"), ".wav")
        self.assertEqual(asr_sidecar._suffix_of("ref.bad-suffix"), ".wav")


class _StubFunASRModel:
    """Stands in for ``funasr.AutoModel`` so engine tests need no funasr install."""

    def __init__(self, text: str = "<|zh|>你好<|NEUTRAL|>") -> None:
        self.text = text
        self.seen_paths = []

    def generate(self, input, language="auto", use_itn=True):  # noqa: A002 - funasr API
        self.seen_paths.append(input)
        self.last_input_exists = Path(input).is_file()
        return [{"text": self.text}]


class _BlockingFunASRModel:
    """Blocks inside ``generate`` until the test releases it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, input, language="auto", use_itn=True):  # noqa: A002 - funasr API
        self.entered.set()
        if not self.release.wait(timeout=10.0):
            raise AssertionError("test never released the model")
        return [{"text": "released"}]


class _ParallelismProbeModel:
    """Tracks how many ``generate`` calls are in flight at the same time.

    ``generate`` rendezvouses on a barrier: if a second caller really overlaps,
    the barrier trips at once and ``max_active`` reaches 2.  If the engine
    correctly serialises inference, the first caller waits out the barrier
    timeout and leaves ``max_active`` at 1.
    """

    def __init__(self, participants: int = 2, timeout: float = 1.0) -> None:
        self.barrier = threading.Barrier(participants)
        self.timeout = timeout
        self._state = threading.Lock()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    def generate(self, input, language="auto", use_itn=True):  # noqa: A002 - funasr API
        with self._state:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=self.timeout)
        except threading.BrokenBarrierError:
            # Expected when inference is serialised: nobody else could join.
            pass
        finally:
            with self._state:
                self.active -= 1
        return [{"text": "probe"}]


class SidecarEngineTests(unittest.TestCase):
    def _engine(self, text: str = "<|zh|>你好<|NEUTRAL|>"):
        engine = asr_sidecar.SenseVoiceEngine(model_dir=".", device="cpu")
        stub = _StubFunASRModel(text)
        engine._model = stub  # bypass the lazy funasr import
        return engine, stub

    def test_transcribe_bytes_writes_a_temp_file_and_strips_tokens(self):
        engine, stub = self._engine()
        text = engine.transcribe_bytes(b"WAVE-DATA", ".wav")

        self.assertEqual(text, "你好")
        self.assertEqual(len(stub.seen_paths), 1)
        self.assertTrue(stub.last_input_exists, "the temp file must exist while funasr runs")
        self.assertFalse(Path(stub.seen_paths[0]).exists(), "the temp file must be removed afterwards")
        self.assertTrue(stub.seen_paths[0].endswith(".wav"))

    def test_unexpected_suffix_is_replaced(self):
        engine, stub = self._engine("ok")
        engine.transcribe_bytes(b"data", ".wav;rm -rf")

        self.assertTrue(stub.seen_paths[0].endswith(".wav"))

    def test_empty_model_result_returns_empty_string(self):
        engine, _ = self._engine("<|zh|>")
        self.assertEqual(engine.transcribe_bytes(b"data", ".wav"), "")

    def test_lazy_import_error_is_not_swallowed(self):
        engine = asr_sidecar.SenseVoiceEngine(model_dir=".", device="cpu")
        with mock.patch.dict(sys.modules, {"funasr": None}):
            with self.assertRaises(ImportError):
                engine.transcribe_bytes(b"data", ".wav")

    # -- inference serialisation ------------------------------------------
    def test_generate_runs_while_holding_the_inference_lock(self):
        engine = asr_sidecar.SenseVoiceEngine(model_dir=".", device="cpu")
        model = _BlockingFunASRModel()
        engine._model = model

        collected = {}

        def worker():
            collected["text"] = engine.transcribe_bytes(b"WAVE", ".wav")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(model.entered.wait(timeout=10.0), "generate() was never reached")
            self.assertTrue(
                engine._inference_lock.locked(),
                "model.generate() must run inside the inference lock",
            )
        finally:
            model.release.set()
            thread.join(timeout=10.0)

        self.assertFalse(thread.is_alive(), "transcribe_bytes did not finish")
        self.assertEqual(collected.get("text"), "released")
        self.assertFalse(
            engine._inference_lock.locked(),
            "the inference lock must be released after generate() returns",
        )

    def test_concurrent_transcribe_never_enters_generate_in_parallel(self):
        engine = asr_sidecar.SenseVoiceEngine(model_dir=".", device="cpu")
        probe = _ParallelismProbeModel(participants=2, timeout=1.0)
        engine._model = probe

        start_together = threading.Barrier(2)
        results = []
        errors = []

        def worker(index: int) -> None:
            try:
                start_together.wait(timeout=10.0)
                results.append(engine.transcribe_bytes(b"WAVE-%d" % index, ".wav"))
            except BaseException as exc:  # noqa: BLE001 - surfaced through `errors`
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(index,), daemon=True) for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15.0)

        self.assertEqual(errors, [])
        self.assertFalse(any(thread.is_alive() for thread in threads), "a worker hung")
        self.assertEqual(probe.calls, 2, "both requests must reach generate()")
        self.assertEqual(
            probe.max_active,
            1,
            "two model.generate() calls overlapped on the same engine instance",
        )
        self.assertEqual(results, ["probe", "probe"])
        self.assertFalse(engine._inference_lock.locked())



if __name__ == "__main__":
    unittest.main(verbosity=2)
