from __future__ import annotations

import importlib
import sys


def ensure_espeakbridge_import() -> bool:
    """Try to make `piper.espeakbridge` importable.

    Returns True on success and False on failure (with warning),
    so callers can decide whether this is fatal for their flow.
    """
    try:
        importlib.import_module("piper.espeakbridge")
        return True
    except Exception:
        pass

    # fallback: if package layout has top-level `espeakbridge`
    try:
        module = importlib.import_module("espeakbridge")
        sys.modules["piper.espeakbridge"] = module
        return True
    except Exception as exc:
        print(
            "[WARN] Не удалось импортировать piper.espeakbridge. "
            "Продолжаю без жёсткой проверки (doctor/training подскажут, если это критично). "
            f"Детали: {exc}",
            file=sys.stderr,
        )
        return False
