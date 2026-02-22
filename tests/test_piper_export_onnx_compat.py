from __future__ import annotations

import importlib.machinery
from pathlib import Path
import types

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


def test_choose_export_module_skips_broken_site_packages_runtime(monkeypatch) -> None:
    available = {
        "piper.train.export_onnx": importlib.machinery.ModuleSpec(
            name="piper.train.export_onnx",
            loader=None,
            origin="/tmp/.venv/site-packages/piper/train/export_onnx.py",
        ),
        "piper.train.vits.export_onnx": importlib.machinery.ModuleSpec(
            name="piper.train.vits.export_onnx",
            loader=None,
            origin="/workspace/repo/third_party/piper1-gpl/src/piper/train/vits/export_onnx.py",
        ),
    }

    monkeypatch.setattr(
        "training.piper_export_onnx_compat.importlib.util.find_spec",
        lambda name: available.get(name),
    )

    chosen = piper_export_onnx_compat._choose_export_module()

    assert chosen == "piper.train.vits.export_onnx"


def test_choose_export_module_handles_missing_parent(monkeypatch) -> None:
    def fake_find_spec(name: str):
        if name == "piper.train.export_onnx":
            raise ModuleNotFoundError("piper.train")
        if name == "piper.train.vits.export_onnx":
            return importlib.machinery.ModuleSpec(name=name, loader=None)
        return None

    monkeypatch.setattr(
        "training.piper_export_onnx_compat.importlib.util.find_spec",
        fake_find_spec,
    )

    chosen = piper_export_onnx_compat._choose_export_module()

    assert chosen == "piper.train.vits.export_onnx"


def test_force_register_third_party_train(monkeypatch, tmp_path: Path) -> None:
    src_root = tmp_path / "third_party" / "piper1-gpl" / "src"
    train_dir = src_root / "piper" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        piper_export_onnx_compat,
        "_repo_root",
        lambda: tmp_path,
    )

    existing = types.ModuleType("piper.train.old")
    monkeypatch.setitem(piper_export_onnx_compat.sys.modules, "piper.train.old", existing)

    piper_export_onnx_compat._force_register_third_party_train()

    assert "piper.train.old" not in piper_export_onnx_compat.sys.modules
    train_pkg = piper_export_onnx_compat.sys.modules["piper.train"]
    assert train_pkg.__package__ == "piper.train"
    assert train_pkg.__path__ == [str(train_dir.resolve())]
    assert str(src_root.resolve()) in piper_export_onnx_compat.sys.path


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
