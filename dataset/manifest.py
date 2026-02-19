from __future__ import annotations

from pathlib import Path
from typing import Iterable


DELIMITER = "|"


def write_manifest(manifest_path: Path, rows: Iterable[tuple[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        for audio_rel, text in rows:
            safe_text = text.replace("\n", " ").strip()
            f.write(f"{audio_rel}{DELIMITER}{safe_text}\n")


def read_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            if DELIMITER not in raw:
                raise ValueError(f"Line {line_no}: missing '{DELIMITER}' delimiter")
            audio, text = raw.split(DELIMITER, maxsplit=1)
            if audio.lower() in {"audio", "audio_path", "path"}:
                raise ValueError("Manifest must not contain CSV header")
            rows.append((audio.strip(), text.strip()))
    return rows


def resolve_audio_path(project_dir: Path, audio_rel: str) -> Path:
    normalized = Path(audio_rel.replace("\\", "/").strip())
    if normalized.is_absolute():
        return normalized

    candidate = (
        project_dir / "recordings" / "wav_22050" / normalized.name
        if len(normalized.parts) == 1
        else project_dir / normalized
    )
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".wav")
