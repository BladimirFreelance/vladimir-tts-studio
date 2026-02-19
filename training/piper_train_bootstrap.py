from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _attach_training_package() -> None:
    import piper

    training_package = (
        Path(__file__).resolve().parent.parent / "third_party" / "piper1-gpl" / "src" / "piper"
    )
    if not training_package.exists():
        raise RuntimeError(
            "Не найден third_party/piper1-gpl/src/piper. "
            "Запустите: python scripts/00_setup_env.py --require-piper-training"
        )

    training_path = str(training_package)
    if training_path not in piper.__path__:
        piper.__path__.append(training_path)


def main() -> None:
    _attach_training_package()
    runpy.run_module("piper.train", run_name="__main__")


if __name__ == "__main__":
    main()
