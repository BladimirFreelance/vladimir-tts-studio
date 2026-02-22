from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys
import types

EXPORT_MODULE_CANDIDATES: tuple[str, ...] = (
    "piper.train.export_onnx",
    "piper.train.vits.export_onnx",
    "piper.export_onnx",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _force_register_third_party_train() -> None:
    src_root = (_repo_root() / "third_party" / "piper1-gpl" / "src").resolve()
    train_dir = (src_root / "piper" / "train").resolve()
    train_init = train_dir / "__init__.py"

    if not train_dir.exists():
        raise RuntimeError(f"Training sources not found: {train_dir}")

    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.append(src_root_str)

    for module_name in list(sys.modules.keys()):
        if module_name == "piper.train" or module_name.startswith("piper.train."):
            sys.modules.pop(module_name, None)

    pkg = types.ModuleType("piper.train")
    pkg.__file__ = str(train_init)
    pkg.__path__ = [str(train_dir)]
    pkg.__package__ = "piper.train"
    sys.modules["piper.train"] = pkg


def _patch_torch_onnx_export() -> None:
    """Force legacy Torch ONNX exporter path used by piper.export_onnx."""
    import torch.onnx

    original_export = torch.onnx.export

    def export_with_compat(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        kwargs.setdefault("fallback", True)
        return original_export(*args, **kwargs)

    torch.onnx.export = export_with_compat


def _ensure_onnxscript() -> None:
    """Install onnxscript if it is missing from the current environment."""
    if importlib.util.find_spec("onnxscript") is not None:
        return

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "onnxscript"],
        check=True,
    )


def _choose_export_module() -> str:
    for module_name in EXPORT_MODULE_CANDIDATES:
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None

        if spec is None:
            continue

        origin = getattr(spec, "origin", None)
        normalized_origin = str(origin).replace("\\", "/") if origin else ""
        if "site-packages/piper/train/export_onnx.py" in normalized_origin:
            continue

        if spec is not None:
            return module_name

    searched = ", ".join(EXPORT_MODULE_CANDIDATES)
    raise RuntimeError(
        "Unable to locate Piper ONNX export module. "
        f"Searched: {searched}. "
        "Verify third_party/piper sources are present and up to date."
    )


def main() -> None:
    _force_register_third_party_train()
    module_name = _choose_export_module()
    _ensure_onnxscript()
    _patch_torch_onnx_export()
    sys.argv[0] = module_name
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
