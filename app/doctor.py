from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
from pathlib import Path

from dataset.audio_tools import convert_wav, inspect_wav
from dataset.manifest import read_manifest

LOGGER = logging.getLogger(__name__)


def check_imports() -> list[str]:
    issues: list[str] = []
    try:
        importlib.import_module("piper.espeakbridge")
    except Exception:
        issues.append("ImportError: piper.espeakbridge (fix: pip install piper-tts)")

    if importlib.util.find_spec("piper.train.vits") is None and importlib.util.find_spec("piper_train") is None:
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


def check_manifest(project_dir: Path, auto_fix: bool = False) -> dict[str, int]:
    manifest = project_dir / "metadata" / "train.csv"
    rows = read_manifest(manifest)
    ok = 0
    missing = 0
    fixed = 0
    for audio_rel, _text in rows:
        wav_path = project_dir / audio_rel
        if not wav_path.exists():
            missing += 1
            continue
        info = inspect_wav(wav_path)
        needs_fix = not (info.sample_rate == 22050 and info.channels == 1 and info.bits_per_sample == 16)
        if needs_fix and auto_fix:
            tmp = wav_path.with_name(f"{wav_path.stem}.fixed.wav")
            method = convert_wav(wav_path, tmp)
            tmp.replace(wav_path)
            LOGGER.info("Fixed %s using %s", wav_path.name, method)
            fixed += 1
            info = inspect_wav(wav_path)
        if 1 <= info.duration_sec <= 12:
            ok += 1
    return {"rows": len(rows), "ok": ok, "missing": missing, "fixed": fixed}


def run_doctor(project_dir: Path, auto_fix: bool = False, require_audio: bool = True) -> int:
    issues = check_imports()
    for issue in issues:
        LOGGER.warning(issue)

    stats = check_manifest(project_dir, auto_fix=auto_fix)
    LOGGER.info("manifest rows=%s ok=%s missing=%s fixed=%s", stats["rows"], stats["ok"], stats["missing"], stats["fixed"])

    if require_audio and stats["ok"] == 0:
        LOGGER.error("0 utterances ready for training. Record audio and run doctor again.")
        return 2
    return 1 if issues else 0
