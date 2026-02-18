import types

import pytest

from training.utils import ensure_espeakbridge_import


def test_ensure_espeakbridge_import_uses_direct_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        if name == "piper.espeakbridge":
            return object()
        raise ImportError(name)

    monkeypatch.setattr("training.utils.importlib.import_module", fake_import)

    ensure_espeakbridge_import()

    assert calls == ["piper.espeakbridge"]


def test_ensure_espeakbridge_import_falls_back_to_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(name="espeakbridge")

    def fake_import(name: str):
        if name == "piper.espeakbridge":
            raise ImportError(name)
        if name == "espeakbridge":
            return fake_module
        raise AssertionError(name)

    monkeypatch.setattr("training.utils.importlib.import_module", fake_import)

    ensure_espeakbridge_import()

    assert __import__("sys").modules["piper.espeakbridge"] is fake_module


def test_ensure_espeakbridge_import_raises_runtime_error_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr("training.utils.importlib.import_module", fake_import)

    with pytest.raises(
        RuntimeError, match="Не удалось импортировать piper.espeakbridge"
    ):
        ensure_espeakbridge_import()
