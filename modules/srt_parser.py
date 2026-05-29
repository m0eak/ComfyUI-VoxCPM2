import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SRTSegment:
    index: int
    start: str
    end: str
    start_seconds: float
    end_seconds: float
    text: str


_TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
_TAG_RE = re.compile(r"<[^>]+>")


def sanitize_job_name(name: str | None) -> str:
    import time

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("_")
    return cleaned or time.strftime("%Y%m%d_%H%M%S")


def parse_srt_timestamp(timestamp: str) -> float:
    matched = _TIMESTAMP_RE.fullmatch((timestamp or "").strip())
    if matched is None:
        raise ValueError(f"Invalid SRT timestamp: {timestamp}")

    hours = int(matched.group(1))
    minutes = int(matched.group(2))
    seconds = int(matched.group(3))
    millis = int(matched.group(4))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _read_text_auto(path: Path, encoding: str) -> str:
    if encoding != "auto":
        return path.read_text(encoding=encoding)

    errors: list[str] = []
    for candidate in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            return path.read_text(encoding=candidate)
        except UnicodeDecodeError as exc:
            errors.append(f"{candidate}: {exc}")

    raise UnicodeDecodeError("auto", b"", 0, 1, "Could not decode SRT file: " + "; ".join(errors))


def _clean_text(text: str, *, normalize_whitespace: bool, strip_tags: bool) -> str:
    cleaned = text or ""
    if strip_tags:
        cleaned = _TAG_RE.sub("", cleaned)
        cleaned = html.unescape(cleaned)
    if normalize_whitespace:
        lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()


def parse_srt_file(
    srt_path: str,
    *,
    encoding: str = "auto",
    skip_empty: bool = True,
    normalize_whitespace: bool = True,
    strip_tags: bool = True,
) -> list[SRTSegment]:
    path = Path(srt_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    raw = _read_text_auto(path, encoding)
    blocks = re.split(r"\r?\n\s*\r?\n", raw.strip())
    segments: list[SRTSegment] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        try:
            subtitle_index = int(lines[0].strip())
            timing = lines[1].strip()
            text_lines = lines[2:]
        except ValueError:
            subtitle_index = len(segments) + 1
            timing = lines[0].strip()
            text_lines = lines[1:]

        if "-->" not in timing:
            continue

        start, end = [part.strip() for part in timing.split("-->", 1)]
        # Ignore optional SRT cue settings after the end timestamp.
        end = end.split()[0]
        text = _clean_text("\n".join(text_lines), normalize_whitespace=normalize_whitespace, strip_tags=strip_tags)
        if skip_empty and not text:
            continue

        segments.append(
            SRTSegment(
                index=subtitle_index,
                start=start,
                end=end,
                start_seconds=parse_srt_timestamp(start),
                end_seconds=parse_srt_timestamp(end),
                text=text,
            )
        )

    return segments


def segments_to_payload(segments: list[SRTSegment], source_path: str) -> dict[str, Any]:
    return {
        "source_path": str(source_path),
        "count": len(segments),
        "segments": [asdict(segment) for segment in segments],
    }


def build_preview_text(segments: list[SRTSegment], *, preview_limit: int = 30) -> str:
    limit = max(0, int(preview_limit))
    shown = segments[:limit]
    lines = [f"Parsed {len(segments)} subtitle segments. Previewing first {len(shown)}:"]
    for segment in shown:
        lines.extend(
            [
                "",
                f"[{segment.index:04d}] {segment.start} --> {segment.end}",
                segment.text,
            ]
        )
    return "\n".join(lines)


def build_preview_json(segments: list[SRTSegment], *, preview_limit: int = 30) -> str:
    return json.dumps([asdict(segment) for segment in segments[: max(0, int(preview_limit))]], ensure_ascii=False, indent=2)
