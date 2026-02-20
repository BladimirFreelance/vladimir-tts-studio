from __future__ import annotations

import runpy
from pathlib import Path

def extend_piper_namespace() -> None:
    import piper  # runtime (from piper-tts wheel)
    repo_root = Path(__file__).resolve().parents[1]
    train_piper_dir = repo_root / "third_party" / "piper1-gpl" / "src" / "piper"

    if not train_piper_dir.exists():
        raise RuntimeError(f"Training sources not found: {train_piper_dir}")

    # Make piper a namespace that also searches training sources
    p = str(train_piper_dir)
    if p not in list(getattr(piper, "__path__", [])):
        piper.__path__.append(p)

def validate_runtime_and_training_imports() -> None:
    import importlib.util as u

    # 1) runtime phonemizer must exist
    if u.find_spec("piper.espeakbridge") is None and u.find_spec("espeakbridge") is None:
        raise RuntimeError("Phonemizer missing: install piper-tts (runtime wheel).")

    # 2) training must become visible after namespace extension
    extend_piper_namespace()
    if u.find_spec("piper.train.vits") is None:
        raise RuntimeError("Training missing: third_party/piper1-gpl present but piper.train.vits not importable.")

def main() -> None:
    extend_piper_namespace()
    # Execute piper.train CLI entrypoint from training sources
    runpy.run_module("piper.train", run_name="__main__")

if __name__ == "__main__":
    main()
