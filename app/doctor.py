from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
from pathlib import Path

from dataset.audio_tools import convert_wav, inspect_wav
from dataset.manifest import read_manifest, resolve_audio_path, write_manifest

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


def _coerce_manifest_path(project_dir: Path, audio_rel: str) -> str | None:
    candidate = audio_rel.replace("\\", "/").strip()
    if not candidate:
        return None

    raw = Path(candidate)
    if raw.is_absolute():
        try:
            rel = raw.resolve().relative_to(project_dir.resolve())
            candidate = rel.as_posix()
        except Exception:
            return None

    normalized = Path(candidate)
    if any(part == ".." for part in normalized.parts):
        return None

    if normalized.suffix.lower() == ".wav":
        normalized = normalized.with_suffix("")
    return normalized.as_posix()


def _duration_warning(total_sec: float) -> str:
    if total_sec < 30 * 60:
        return (
            "Training data is likely insufficient (<30 minutes). "
            "Mumbling voices are often caused by too little data."
        )
    if total_sec < 60 * 60:
        return (
            "Training data is below the recommended 60 minutes. "
            "Mumbling risk is still elevated."
        )
    return ""


def check_manifest(project_dir: Path, auto_fix: bool = False) -> dict[str, int | float | str]:
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
    total_duration_sec = 0.0
    updated_rows: list[tuple[str, str]] = []
    path_fixed = 0

    project_root = project_dir.resolve()

    for audio_rel, text in rows:
        normalized_rel = _coerce_manifest_path(project_dir, audio_rel)
        if normalized_rel is None:
            LOGGER.warning(
                "Invalid manifest audio path (must be relative to project root): %s",
                audio_rel,
            )
            invalid_paths += 1
            continue

        if normalized_rel != audio_rel:
            if auto_fix:
                LOGGER.info("Auto-fixed manifest path: %s -> %s", audio_rel, normalized_rel)
                path_fixed += 1
            else:
                LOGGER.warning("Non-normalized manifest path: %s", audio_rel)

        rel_path = Path(normalized_rel)
        updated_rows.append((normalized_rel, text))

        wav_path = resolve_audio_path(project_dir, normalized_rel)

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
        total_duration_sec += info.duration_sec
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

    if auto_fix and updated_rows and path_fixed > 0:
        write_manifest(manifest, updated_rows)

    duration_warning = _duration_warning(total_duration_sec)
    if duration_warning:
        LOGGER.warning(duration_warning)

    return {
        "rows": len(rows),
        "ok": ok,
        "missing": missing,
        "fixed": fixed,
        "path_fixed": path_fixed,
        "invalid_paths": invalid_paths,
        "sample_rate_mismatch": sample_rate_mismatch,
        "duration_min": round(total_duration_sec / 60.0, 2),
        "duration_warning": duration_warning,
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
        "manifest rows=%s ok=%s missing=%s fixed=%s path_fixed=%s invalid_paths=%s sample_rate_mismatch=%s duration_min=%s",
        stats["rows"],
        stats["ok"],
        stats["missing"],
        stats["fixed"],
        stats.get("path_fixed", 0),
        stats.get("invalid_paths", 0),
        stats.get("sample_rate_mismatch", 0),
        stats.get("duration_min", 0.0),
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
