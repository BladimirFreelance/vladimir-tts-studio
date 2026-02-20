from __future__ import annotations

import importlib
import runpy
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def extend_piper_namespace() -> None:
    """
    Keep runtime piper (wheel, gives piper.espeakbridge) but extend training subpackages
    by adding third_party sources into piper.train.__path__.
    """
    import piper  # runtime from wheel

    repo_root = _repo_root()
    src_piper = repo_root / "third_party" / "piper1-gpl" / "src" / "piper"
    src_train = src_piper / "train"

    if not src_train.exists():
        raise RuntimeError(f"Training sources not found: {src_train}")

    # Optional: let `piper` namespace see src/piper too
    p = str(src_piper)
    if p not in list(getattr(piper, "__path__", [])):
        piper.__path__.append(p)

    # Critical: piper-tts already provides piper.train without vits.
    # We must extend piper.train.__path__ to include third_party training sources.
    import piper.train  # comes from wheel
    t = str(src_train)
    if t not in list(getattr(piper.train, "__path__", [])):
        piper.train.__path__.append(t)

    importlib.invalidate_caches()


def validate_runtime_and_training_imports() -> None:
    import importlib.util as u

    if u.find_spec("piper.espeakbridge") is None and u.find_spec("espeakbridge") is None:
        raise RuntimeError("Phonemizer missing: install piper-tts runtime wheel.")

    extend_piper_namespace()

    if u.find_spec("piper.train.vits") is None:
        raise RuntimeError("Training missing: piper.train.vits not importable (check third_party clone).")


def main() -> None:
    extend_piper_namespace()
    runpy.run_module("piper.train", run_name="__main__")


if __name__ == "__main__":
    main()
