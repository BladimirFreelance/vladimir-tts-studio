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
    raw_path = project_dir / audio_rel
    if raw_path.exists():
        return raw_path
    if Path(audio_rel).suffix:
        return raw_path
    wav_path = raw_path.with_suffix(".wav")
    return wav_path if wav_path.exists() else raw_path
