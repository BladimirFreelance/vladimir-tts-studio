from __future__ import annotations

import importlib
import sys


def ensure_espeakbridge_import() -> None:
    """Make `piper.espeakbridge` importable, with fallback for mixed piper installs."""
    try:
        importlib.import_module("piper.espeakbridge")
        return
    except Exception:
        pass

    # fallback: if package layout has top-level `espeakbridge`
    try:
        module = importlib.import_module("espeakbridge")
        sys.modules["piper.espeakbridge"] = module
    except Exception as exc:
        raise RuntimeError(
            "Не удалось импортировать piper.espeakbridge. Установите piper-tts: pip install piper-tts"
        ) from exc
