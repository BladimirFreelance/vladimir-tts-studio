from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _attach_training_package() -> None:
    import piper

    training_src = Path(__file__).resolve().parent.parent / "third_party" / "piper1-gpl" / "src"
    if not training_src.exists():
        raise RuntimeError(
            "Не найден third_party/piper1-gpl/src. "
            "Запустите: python scripts/00_setup_env.py --require-piper-training"
        )

    training_src_str = str(training_src)
    if training_src_str not in sys.path:
        sys.path.insert(0, training_src_str)

    # Runtime piper (из piper-tts) уже импортирован выше; теперь расширяем пакетный путь,
    # чтобы одновременно работал piper.train из third_party.
    training_package = training_src / "piper"
    training_package_str = str(training_package)
    if training_package_str not in piper.__path__:
        piper.__path__.append(training_package_str)


def main() -> None:
    _attach_training_package()
    runpy.run_module("piper.train", run_name="__main__")


if __name__ == "__main__":
    main()
