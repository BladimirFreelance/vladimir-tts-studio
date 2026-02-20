from __future__ import annotations

import runpy
from pathlib import Path


def _training_package_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parent.parent
    return root / "third_party" / "piper1-gpl" / "src" / "piper"


def attach_piper_training_namespace(repo_root: Path | None = None) -> None:
    import piper

    training_package = _training_package_path(repo_root)
    if not training_package.exists():
        raise RuntimeError(
            "Не найден third_party/piper1-gpl/src/piper. "
            "Запустите: python scripts/00_setup_env.py --require-piper-training"
        )

    training_package_str = str(training_package)
    if training_package_str not in piper.__path__:
        piper.__path__.append(training_package_str)


def validate_runtime_and_training_imports(repo_root: Path | None = None) -> None:
    import piper.espeakbridge  # noqa: F401

    attach_piper_training_namespace(repo_root)

    import piper.train.vits  # noqa: F401


def main() -> None:
    attach_piper_training_namespace()
    runpy.run_module("piper.train.__main__", run_name="__main__")


if __name__ == "__main__":
    main()
