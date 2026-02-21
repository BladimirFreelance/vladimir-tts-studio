from __future__ import annotations

import logging
import importlib
import os
import shutil
from pathlib import Path

from dataset.audio_tools import convert_wav, inspect_wav
from dataset.manifest import read_manifest, resolve_audio_path, write_manifest

LOGGER = logging.getLogger(__name__)


def check_imports() -> list[str]:
    issues: list[str] = []

    try:
        importlib.import_module("piper.espeakbridge")
        from training.piper_train_bootstrap import validate_runtime_and_training_imports

        validate_runtime_and_training_imports()
        LOGGER.info("phonemizer backend OK: piper.espeakbridge")
        LOGGER.info("training module OK: piper.train.vits (via bootstrap)")
    except SystemExit:
        issues.append(
            "Piper training module not found (piper.train.vits via bootstrap; fix: pip install -r requirements/train.txt)"
        )
    except Exception:
        issues.append(
            "Piper training module not found (piper.train.vits via bootstrap; fix: pip install -r requirements/train.txt)"
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


def _coerce_manifest_path(
    project_dir: Path, audio_rel: str, *, allow_salvage: bool = False
) -> str | None:
    candidate = audio_rel.replace("\\", "/").strip()
    if not candidate:
        return None

    raw = Path(candidate)
    if raw.is_absolute():
        try:
            rel = raw.resolve().relative_to(project_dir.resolve())
            candidate = rel.as_posix()
        except Exception:
            if allow_salvage and raw.name:
                return Path(raw.name).as_posix()
            return None

    normalized = Path(candidate)
    if any(part == ".." for part in normalized.parts):
        if allow_salvage and normalized.name:
            return Path(normalized.name).as_posix()
        return None

    return Path(normalized.name).as_posix()


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


def check_manifest(
    project_dir: Path, auto_fix: bool = False, audio_dir: Path | None = None
) -> dict[str, int | float | str]:
    manifest = project_dir / "metadata" / "train.csv"
    try:
        rows = read_manifest(manifest)
    except FileNotFoundError:
        if "<" in project_dir.name or ">" in project_dir.name:
            LOGGER.error(
                "Project name looks like a placeholder: %s. Use repository project name or pass --project explicitly.",
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
        normalized_rel = _coerce_manifest_path(
            project_dir, audio_rel, allow_salvage=auto_fix
        )
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

        wav_path = resolve_audio_path(project_dir, normalized_rel, audio_dir=audio_dir)

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





def build_training_preflight_report(
    project_dir: Path, *, audio_dir: Path | None = None
) -> tuple[list[str], list[str]]:
    """Return (blocking_issues, remediation_steps) for training startup."""
    issues = check_imports()
    stats = check_manifest(project_dir, auto_fix=False, audio_dir=audio_dir)

    blocking: list[str] = []
    remediation: list[str] = []

    if not os.getenv("PIPER_TRAIN_CMD") and any(
        ("Piper training module not found" in issue)
        or ("Piper runtime/training modules are not importable" in issue)
        for issue in issues
    ):
        blocking.append("Не найден Piper training модуль (piper.train.vits).")
        remediation.append("pip install -r requirements/train.txt")

    if not os.getenv("PIPER_TRAIN_CMD") and any("PyTorch not importable" in issue for issue in issues):
        blocking.append("PyTorch не импортируется: обучение недоступно.")
        remediation.append("pip install torch torchvision torchaudio")

    if stats.get("error") == "manifest_missing":
        blocking.append("Не найден metadata/train.csv для проекта.")
        remediation.append(
            f"python -m app.main prepare --project {project_dir.name}"
        )
    elif stats.get("error") == "manifest_invalid":
        blocking.append("Формат metadata/train.csv некорректен.")
        remediation.append("Исправьте metadata/train.csv и запустите doctor ещё раз.")

    if int(stats.get("rows", 0)) == 0:
        blocking.append("Нет готовых записей для обучения (0 utterances).")
        remediation.append(
            f"python -m app.main doctor --project {project_dir.name} --auto-fix"
        )

    if int(stats.get("missing", 0)) > 0:
        blocking.append(
            f"В manifest есть отсутствующие WAV: {stats.get('missing', 0)} шт."
        )
        remediation.append("Проверьте записи и заново запустите doctor с --auto-fix.")

    if any("espeak-ng not found" in issue for issue in issues):
        remediation.append("Установите espeak-ng и добавьте его в PATH.")

    return blocking, remediation


def assert_training_preflight(
    project_dir: Path, *, audio_dir: Path | None = None
) -> None:
    """Raise RuntimeError with actionable hints if training preflight fails."""
    blocking, remediation = build_training_preflight_report(
        project_dir, audio_dir=audio_dir
    )
    if not blocking:
        return

    details = "\n".join(f"- {item}" for item in blocking)
    fixes = "\n".join(f"  * {step}" for step in dict.fromkeys(remediation))
    raise RuntimeError(
        "Training preflight failed before launch:\n"
        f"{details}\n"
        "Как починить:\n"
        f"{fixes}"
    )


def run_doctor(
    project_dir: Path,
    auto_fix: bool = False,
    require_audio: bool = True,
    audio_dir: Path | None = None,
) -> int:
    if auto_fix:
        base_audio_dir = audio_dir or project_dir / "recordings" / "wav_22050"
        base_audio_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "cache").mkdir(parents=True, exist_ok=True)

    issues = check_imports()
    if auto_fix and any(("Piper training module not found" in issue) or ("Piper runtime/training modules are not importable" in issue) for issue in issues):
        LOGGER.warning(
            "Auto-fix for training dependencies is not available. Run manually: pip install -r requirements/train.txt"
        )

    for issue in issues:
        LOGGER.warning(issue)

    stats = check_manifest(project_dir, auto_fix=auto_fix, audio_dir=audio_dir)
    LOGGER.info(
        "manifest rows=%s ok=%s missing=%s fixed=%s path_fixed=%s invalid_paths=%s sample_rate_mismatch=%s duration_min=%s",
        stats["rows"],
        stats["ok"],
        stats["missing"],
        stats["fixed"],
        stats.get("path_fixed", 0),
        stats["invalid_paths"],
        stats["sample_rate_mismatch"],
        stats.get("duration_min", 0.0),
    )

    if require_audio and stats["ok"] == 0:
        if stats.get("error") == "manifest_missing":
            LOGGER.error(
                "No manifest found. Run prepare first: python -m app.main prepare --project %s",
                project_dir.name,
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

    return 0 if not issues else 1
