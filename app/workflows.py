from __future__ import annotations

import csv
import re
from pathlib import Path

from dataset.audio_tools import convert_wav, inspect_wav
from dataset.manifest import write_manifest
from dataset.segmenter import split_to_segments
from dataset.text_cleaner import normalize_text
from utils import load_yaml


def _resample_existing_audio(
    project_dir: Path, audio_ids: list[str], sample_rate: int = 22050
) -> None:
    recordings_dir = project_dir / "recordings" / "wav_22050"
    for audio_id in audio_ids:
        wav_path = recordings_dir / f"{audio_id}.wav"
        raw_path = recordings_dir / audio_id
        source = (
            wav_path if wav_path.exists() else raw_path if raw_path.exists() else None
        )
        if source is None:
            continue

        info = None
        if source.suffix.lower() == ".wav":
            try:
                info = inspect_wav(source)
            except Exception:
                info = None
        if (
            info
            and info.sample_rate == sample_rate
            and info.channels == 1
            and info.bits_per_sample == 16
        ):
            continue

        target = wav_path
        tmp = target.with_name(f"{audio_id}.tmp.wav")
        convert_wav(source, tmp, sample_rate=sample_rate)
        tmp.replace(target)
        if source != target and source.exists():
            source.unlink(missing_ok=True)


def _extract_text_entries(text_file: Path) -> list[str]:
    suffix = text_file.suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return [text_file.read_text(encoding="utf-8")]

    delimiter = "\t" if suffix == ".tsv" else ","
    entries: list[str] = []
    with text_file.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            clean_cells = [cell.strip() for cell in row]
            if not any(clean_cells):
                continue

            if clean_cells[0].lower() in {
                "audio",
                "audio_path",
                "path",
                "text",
                "transcript",
                "sentence",
            }:
                continue

            if len(clean_cells) >= 2 and clean_cells[1]:
                entries.append(clean_cells[1])
                continue

            entries.append(clean_cells[0])

    return entries


def parse_existing_index(manifest_path: Path, project_name: str) -> int:
    if not manifest_path.exists():
        return 0

    pattern = re.compile(rf"^{re.escape(project_name)}_(\d{{5}})\.wav$", re.IGNORECASE)
    max_index = 0
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line in manifest:
            row = line.strip()
            if not row or "|" not in row:
                continue
            wav_name, _text = row.split("|", maxsplit=1)
            match = pattern.match(Path(wav_name.strip()).name)
            if not match:
                continue
            max_index = max(max_index, int(match.group(1)))
    return max_index


def generate_next_wav_name(project_name: str, index: int) -> str:
    return f"{project_name}_{index:05d}.wav"


def _normalize_manifest_text(text: str) -> str:
    return " ".join(text.split()).strip()


def prepare_dataset(
    text_file: Path,
    project: str,
    mode: str = "append",
    overwrite_confirmed: bool = False,
) -> None:
    text_file = text_file.expanduser()
    if not text_file.exists() or not text_file.is_file():
        raise FileNotFoundError(
            f"Текстовый файл не найден: {text_file}. "
            "Укажите существующий путь через --text."
        )

    normalized_mode = mode.lower()
    if normalized_mode not in {"append", "overwrite"}:
        raise ValueError("Unsupported prepare mode. Use 'append' or 'overwrite'.")
    if normalized_mode == "overwrite" and not overwrite_confirmed:
        raise ValueError(
            "Overwrite mode requires explicit confirmation. "
            "Use --overwrite with --force or --confirm OVERWRITE."
        )

    cfg = load_yaml(Path("configs/train_default.yaml"))
    project_dir = Path("data/projects") / project
    (project_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (project_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (project_dir / "recordings" / "wav_22050").mkdir(parents=True, exist_ok=True)

    raw_entries = _extract_text_entries(text_file)
    all_segments: list[str] = []
    for raw in raw_entries:
        cleaned = normalize_text(
            raw,
            replacements=cfg["text"]["replacements"],
            expand_abbreviations=cfg["text"].get("expand_common_abbreviations", True),
        )
        all_segments.extend(
            split_to_segments(
                cleaned, cfg["text"]["split_regex"], cfg["text"]["max_chars"]
            )
        )

    prompts_path = project_dir / "prompts" / "segments.txt"
    manifest_path = project_dir / "metadata" / "train.csv"

    existing_rows: list[tuple[str, str]] = []
    existing_prompts: list[tuple[str, str]] = []
    if normalized_mode == "append" and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as manifest:
            for line in manifest:
                row = line.strip()
                if not row or "|" not in row:
                    continue
                wav_name, text = row.split("|", maxsplit=1)
                wav_leaf = Path(wav_name.strip()).name
                clean_text = _normalize_manifest_text(text)
                if not wav_leaf or not clean_text:
                    continue
                existing_rows.append((wav_leaf, clean_text))
                existing_prompts.append((Path(wav_leaf).stem, clean_text))

    start_index = (
        parse_existing_index(manifest_path, project) + 1 if normalized_mode == "append" else 1
    )
    indexed = [
        (f"{project}_{idx:05d}", text)
        for idx, text in enumerate(all_segments, start=start_index)
    ]

    prompt_rows = existing_prompts + indexed if normalized_mode == "append" else indexed
    prompts_path.write_text(
        "\n".join(f"{audio_id}|{text}" for audio_id, text in prompt_rows) + "\n",
        encoding="utf-8",
    )
    test20 = project_dir / "prompts" / "test_script_20.txt"
    test20.write_text(
        "\n".join(text for _id, text in prompt_rows[:20]) + "\n", encoding="utf-8"
    )

    _resample_existing_audio(
        project_dir, [audio_id for audio_id, _text in indexed], sample_rate=22050
    )

    new_rows = [
        (generate_next_wav_name(project, idx), _normalize_manifest_text(text))
        for idx, text in enumerate(all_segments, start=start_index)
    ]
    manifest_rows = existing_rows + new_rows if normalized_mode == "append" else new_rows
    write_manifest(manifest_path, manifest_rows)
