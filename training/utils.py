from __future__ import annotations

import importlib
import platform
import sys


def ensure_espeakbridge_import() -> bool:
    """Try to make `piper.espeakbridge` importable."""
    try:
        importlib.import_module("piper.espeakbridge")
        return True
    except Exception:
        pass

    for module_name in ("espeakbridge", "piper_phonemize"):
        try:
            module = importlib.import_module(module_name)
            sys.modules["piper.espeakbridge"] = module
            sys.modules.setdefault("espeakbridge", module)
            return True
        except Exception:
            continue

    windows_hint = (
        " На Windows установите training-зависимости через "
        "`pip install -r requirements/train.txt`."
        if platform.system().lower() == "windows"
        else ""
    )
    print(
        "[WARN] Не удалось импортировать piper.espeakbridge. "
        "Продолжаю без жёсткой проверки (doctor/training подскажут, если это критично). "
        f"Детали: module not found.{windows_hint}",
        file=sys.stderr,
    )
    return False
