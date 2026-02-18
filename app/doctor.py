from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
from pathlib import Path

from dataset.audio_tools import convert_wav, inspect_wav
from dataset.manifest import read_manifest, resolve_audio_path

LOGGER = logging.getLogger(__name__)


def check_imports() -> list[str]:
    issues: list[str] = []
    try:
        importlib.import_module("piper.espeakbridge")
    except Exception:
        issues.append("ImportError: piper.espeakbridge (fix: pip install piper-tts)")

    if (
        importlib.util.find_spec("piper.train.vits") is None
        and importlib.util.find_spec("piper_train") is None
    ):
        issues.append(
            "Piper training modules not found (fix: install Piper training build or set PIPER_TRAIN_CMD, e.g. python -m piper_train)"
        )

    try:
        import torch

        cuda = torch.cuda.is_available()
        LOGGER.info("torch=%s cuda=%s", torch.__version__, cuda)
    except Exception:
        issues.append("PyTorch not importable (training unavailable)")

    if shutil.which("espeak-ng") is None:
        issues.append("espeak-ng not found in PATH (required for espeak phonemization)")
    return issues


def check_manifest(project_dir: Path, auto_fix: bool = False) -> dict[str, int | str]:
    manifest = project_dir / "metadata" / "train.csv"
    try:
        rows = read_manifest(manifest)
    except FileNotFoundError:
        if "<" in project_dir.name or ">" in project_dir.name:
            LOGGER.error(
                "Project name looks like a placeholder: %s. Use a real project name instead of <name>.",
                project_dir.name,
            )
        LOGGER.error("Manifest not found: %s", manifest)
        LOGGER.info(
            "Run 'prepare' first to generate metadata/train.csv for this project."
        )
        return {
            "rows": 0,
            "ok": 0,
            "missing": 0,
            "fixed": 0,
            "invalid_paths": 0,
            "sample_rate_mismatch": 0,
            "error": "manifest_missing",
        }
    except ValueError as exc:
        LOGGER.error("Invalid manifest format in %s: %s", manifest, exc)
        return {
            "rows": 0,
            "ok": 0,
            "missing": 0,
            "fixed": 0,
            "invalid_paths": 0,
            "sample_rate_mismatch": 0,
            "error": "manifest_invalid",
        }

    ok = 0
    missing = 0
    fixed = 0
    invalid_paths = 0
    sample_rate_mismatch = 0

    project_root = project_dir.resolve()

    for audio_rel, _text in rows:
        rel_path = Path(audio_rel)
        if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
            LOGGER.warning(
                "Invalid manifest audio path (must be relative to project root): %s",
                audio_rel,
            )
            invalid_paths += 1
            continue

        wav_path = resolve_audio_path(project_dir, audio_rel)

        try:
            wav_path.resolve().relative_to(project_root)
        except ValueError:
            LOGGER.warning(
                "Invalid manifest audio path (escapes project root): %s", audio_rel
            )
            invalid_paths += 1
            continue

        if not wav_path.exists():
            LOGGER.warning("Missing file referenced in manifest: %s", wav_path)
            missing += 1
            continue
        info = inspect_wav(wav_path)
        needs_fix = not (
            info.sample_rate == 22050
            and info.channels == 1
            and info.bits_per_sample == 16
        )
        if info.sample_rate != 22050:
            sample_rate_mismatch += 1
            LOGGER.warning(
                "Sample rate mismatch for %s: %s Hz (expected 22050 Hz)",
                wav_path.name,
                info.sample_rate,
            )
        if needs_fix and auto_fix:
            tmp = wav_path.with_name(f"{wav_path.stem}.fixed.wav")
            method = convert_wav(wav_path, tmp)
            tmp.replace(wav_path)
            LOGGER.info("Fixed %s using %s", wav_path.name, method)
            fixed += 1
            info = inspect_wav(wav_path)
        if 1 <= info.duration_sec <= 12:
            ok += 1
    return {
        "rows": len(rows),
        "ok": ok,
        "missing": missing,
        "fixed": fixed,
        "invalid_paths": invalid_paths,
        "sample_rate_mismatch": sample_rate_mismatch,
        "error": "",
    }


def run_doctor(
    project_dir: Path, auto_fix: bool = False, require_audio: bool = True
) -> int:
    issues = check_imports()
    for issue in issues:
        LOGGER.warning(issue)

    stats = check_manifest(project_dir, auto_fix=auto_fix)
    LOGGER.info(
        "manifest rows=%s ok=%s missing=%s fixed=%s invalid_paths=%s sample_rate_mismatch=%s",
        stats["rows"],
        stats["ok"],
        stats["missing"],
        stats["fixed"],
        stats.get("invalid_paths", 0),
        stats.get("sample_rate_mismatch", 0),
    )

    if require_audio and stats["ok"] == 0:
        if stats.get("error") == "manifest_missing":
            LOGGER.error(
                "No manifest found. Run prepare first: python scripts/01_prepare_dataset.py --text <path_to_txt> --project <project_name>"
            )
        elif stats.get("error") == "manifest_invalid":
            LOGGER.error(
                "Manifest format is invalid. Fix metadata/train.csv and run doctor again."
            )
        else:
            LOGGER.error(
                "0 utterances ready for training. Record audio and run doctor again."
            )
        return 2
    return 1 if issues else 0
