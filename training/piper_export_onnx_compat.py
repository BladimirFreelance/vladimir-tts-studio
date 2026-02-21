from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys

from training.piper_train_bootstrap import extend_piper_namespace


EXPORT_MODULE_CANDIDATES: tuple[str, ...] = (
    "piper.export_onnx",
    "piper.train.export_onnx",
    "piper.train.vits.export_onnx",
)


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
        if importlib.util.find_spec(module_name) is not None:
            return module_name

    searched = ", ".join(EXPORT_MODULE_CANDIDATES)
    raise RuntimeError(
        "Unable to locate Piper ONNX export module. "
        f"Searched: {searched}. "
        "Verify third_party/piper sources are present and up to date."
    )


def main() -> None:
    extend_piper_namespace()
    module_name = _choose_export_module()
    _ensure_onnxscript()
    _patch_torch_onnx_export()
    sys.argv[0] = module_name
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
