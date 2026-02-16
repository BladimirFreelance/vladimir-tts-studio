from __future__ import annotations

from pathlib import Path

from dataset.manifest import write_manifest
from dataset.segmenter import indexed_segments, split_to_segments
from dataset.text_cleaner import normalize_text
from utils import load_yaml


def prepare_dataset(text_file: Path, project: str) -> None:
    cfg = load_yaml(Path("configs/train_default.yaml"))
    project_dir = Path("data/projects") / project
    (project_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (project_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (project_dir / "recordings" / "wav_22050").mkdir(parents=True, exist_ok=True)

    raw = text_file.read_text(encoding="utf-8")
    cleaned = normalize_text(
        raw,
        replacements=cfg["text"]["replacements"],
        expand_abbreviations=cfg["text"].get("expand_common_abbreviations", True),
    )
    segments = split_to_segments(cleaned, cfg["text"]["split_regex"], cfg["text"]["max_chars"])
    indexed = indexed_segments(segments, prefix=project)

    prompts_path = project_dir / "prompts" / "segments.txt"
    prompts_path.write_text("\n".join(f"{audio_id}|{text}" for audio_id, text in indexed) + "\n", encoding="utf-8")
    test20 = project_dir / "prompts" / "test_script_20.txt"
    test20.write_text("\n".join(text for _id, text in indexed[:20]) + "\n", encoding="utf-8")

    manifest_rows = [(f"recordings/wav_22050/{audio_id}.wav", text) for audio_id, text in indexed]
    write_manifest(project_dir / "metadata" / "train.csv", manifest_rows)
