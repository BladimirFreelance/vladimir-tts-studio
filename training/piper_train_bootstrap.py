from __future__ import annotations

import importlib
import runpy
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def extend_piper_namespace() -> None:
    """
    Make `piper.train.*` importable from third_party while keeping runtime `piper`
    from the piper-tts wheel (so piper.espeakbridge stays available).

    Strategy:
    1) Import runtime `piper` first (wheel).
    2) Append third_party/src/piper to piper.__path__ so subpackages like piper.train are discoverable.
    3) Also append third_party/src to sys.path (low priority) for any absolute imports inside training.
    """
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "third_party" / "piper1-gpl" / "src"
    src_piper = src_root / "piper"

    if not src_piper.exists():
        raise RuntimeError(f"Training sources not found: {src_piper}")

    # 1) Load runtime wheel first
    import piper  # noqa: WPS433

    # 2) Extend piper package search path
    src_piper_str = str(src_piper.resolve())
    if hasattr(piper, "__path__"):
        if src_piper_str not in list(piper.__path__):
            piper.__path__.append(src_piper_str)

    # 3) Keep src_root available (low priority)
    src_root_str = str(src_root.resolve())
    if src_root_str not in sys.path:
        sys.path.append(src_root_str)

def validate_runtime_and_training_imports() -> None:
    """
    Preflight check:
    - runtime phonemizer must exist (wheel: piper.espeakbridge)
    - training sources must exist in third_party (piper1-gpl)
    We do NOT import piper.train.vits directly because runtime `piper` comes from wheel
    and is not a namespace package.
    """
    import importlib

    # runtime phonemizer must exist (wheel)
    importlib.import_module("piper.espeakbridge")

    # training sources must exist on disk
    repo_root = Path(__file__).resolve().parent.parent
    train_dir = repo_root / "third_party" / "piper1-gpl" / "src" / "piper" / "train" / "vits"
    if not train_dir.exists():
        raise RuntimeError(f"Training missing: sources not found: {train_dir}")


def main() -> None:
    extend_piper_namespace()
    runpy.run_module("piper.train", run_name="__main__")


if __name__ == "__main__":
    main()
