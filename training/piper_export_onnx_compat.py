from __future__ import annotations

import runpy
import sys


def _patch_torch_onnx_export() -> None:
    """Force legacy Torch ONNX exporter path used by piper.export_onnx."""
    import torch.onnx

    original_export = torch.onnx.export

    def export_with_compat(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        kwargs.setdefault("fallback", True)
        return original_export(*args, **kwargs)

    torch.onnx.export = export_with_compat


def main() -> None:
    _patch_torch_onnx_export()
    sys.argv[0] = "piper.export_onnx"
    runpy.run_module("piper.export_onnx", run_name="__main__")


if __name__ == "__main__":
    main()
