import json
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_manifest(job_dir: Path, manifest: list[dict[str, Any]]) -> Path:
    return write_json(job_dir / "manifest.json", manifest)


def write_progress(
    job_dir: Path,
    *,
    job_name: str,
    total: int,
    processed: int,
    success: int,
    failed: int,
    last_subtitle_index: int | None,
    status: str,
) -> Path:
    payload = {
        "job_name": job_name,
        "status": status,
        "total": int(total),
        "processed": int(processed),
        "success": int(success),
        "failed": int(failed),
        "last_subtitle_index": None if last_subtitle_index is None else int(last_subtitle_index),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return write_json(job_dir / "progress.json", payload)


def load_completed_manifest(job_dir: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.exists():
        return [], {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed: dict[int, dict[str, Any]] = {}
    for item in manifest:
        try:
            subtitle_index = int(item.get("subtitle_index"))
        except Exception:
            continue

        output_file = str(item.get("output_file", "") or "")
        if item.get("status") != "ok" or not output_file:
            continue

        output_path = job_dir / output_file
        if output_path.exists() and output_path.is_file():
            completed[subtitle_index] = item

    return manifest, completed


def upsert_manifest_item(manifest: list[dict[str, Any]], item: dict[str, Any]) -> None:
    subtitle_index = item.get("subtitle_index")
    for idx, existing in enumerate(manifest):
        if existing.get("subtitle_index") == subtitle_index:
            manifest[idx] = item
            return
    manifest.append(item)
