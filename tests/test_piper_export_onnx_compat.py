from __future__ import annotations

import importlib.machinery

from training import piper_export_onnx_compat


def test_choose_export_module_prefers_first_available(monkeypatch) -> None:
    available = {
        "piper.train.export_onnx": importlib.machinery.ModuleSpec(
            name="piper.train.export_onnx",
            loader=None,
        ),
        "piper.train.vits.export_onnx": importlib.machinery.ModuleSpec(
            name="piper.train.vits.export_onnx",
            loader=None,
        ),
    }

    monkeypatch.setattr(
        "training.piper_export_onnx_compat.importlib.util.find_spec",
        lambda name: available.get(name),
    )

    chosen = piper_export_onnx_compat._choose_export_module()

    assert chosen == "piper.train.export_onnx"


def test_choose_export_module_raises_clear_error_when_none_found(monkeypatch) -> None:
    monkeypatch.setattr(
        "training.piper_export_onnx_compat.importlib.util.find_spec",
        lambda _name: None,
    )

    try:
        piper_export_onnx_compat._choose_export_module()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    for module_name in piper_export_onnx_compat.EXPORT_MODULE_CANDIDATES:
        assert module_name in message
    assert "third_party" in message
