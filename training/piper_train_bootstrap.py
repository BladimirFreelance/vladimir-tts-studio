from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _training_package_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[1]
    return root / "third_party" / "piper1-gpl" / "src" / "piper"


def extend_piper_namespace(repo_root: Path | None = None) -> None:
    import piper

    training_package = _training_package_path(repo_root)
    if not training_package.exists():
        return

    training_package_str = str(training_package)
    if training_package_str not in piper.__path__:
        piper.__path__.append(training_package_str)


def validate_runtime_and_training_imports(repo_root: Path | None = None) -> None:
    try:
        import piper  # noqa: F401
        import piper.espeakbridge  # noqa: F401
    except Exception as exc:
        print(
            "[FAIL] Runtime piper-tts не импортируется (piper.espeakbridge). "
            "Установите: python -m pip install piper-tts==1.4.1",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    extend_piper_namespace(repo_root)

    try:
        import piper.train.vits  # noqa: F401
    except Exception as exc:
        print(
            "[FAIL] Не удалось импортировать piper.train.vits через bootstrap. "
            "Проверьте наличие third_party/piper1-gpl/src/piper и повторите setup.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print("[OK] Runtime piper.espeakbridge + training piper.train.vits доступны")


def main() -> None:
    extend_piper_namespace()
    runpy.run_module("piper.train", run_name="__main__")


if __name__ == "__main__":
    main()
