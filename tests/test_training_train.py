from __future__ import annotations

from pathlib import Path

import pytest

from training.train import resolve_train_base_command, run_training


def _prepare_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    metadata = project_dir / "metadata"
    recordings = project_dir / "recordings" / "wav_22050"
    metadata.mkdir(parents=True)
    recordings.mkdir(parents=True)
    (metadata / "train.csv").write_text("stub", encoding="utf-8")
    (recordings / "001.wav").write_bytes(b"RIFF")
    return project_dir


def _stub_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("training.train.ensure_espeakbridge_import", lambda: None)
    monkeypatch.setattr(
        "training.train.load_yaml",
        lambda _path: {
            "project_defaults": {"sample_rate": 22050, "use_warmstart_if_found": False},
            "training": {
                "phoneme_type": "espeak",
                "espeak_voice": "ru",
                "batch_size": 16,
                "learning_rate": 0.0002,
            },
        },
    )
    monkeypatch.setattr("training.train.read_manifest", lambda _manifest: [("recordings/wav_22050/001.wav", "text")])
    monkeypatch.setattr("training.train.write_json", lambda *args, **kwargs: None)


def test_resolve_train_base_command_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPER_TRAIN_CMD", "python -m my.custom.train")
    assert resolve_train_base_command() == ["python", "-m", "my.custom.train"]


def test_resolve_train_base_command_uses_piper_train_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPER_TRAIN_CMD", raising=False)

    def fake_find_spec(name: str):
        if name == "piper.train.vits":
            return object()
        if name == "piper_train":
            return None
        raise AssertionError(name)

    monkeypatch.setattr("training.train.importlib.util.find_spec", fake_find_spec)

    assert resolve_train_base_command() == [__import__("sys").executable, "-m", "piper.train"]


def test_resolve_train_base_command_prefers_piper_train_when_both_layouts_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPER_TRAIN_CMD", raising=False)

    calls: list[str] = []

    def fake_find_spec(name: str):
        calls.append(name)
        if name in {"piper.train.vits", "piper_train"}:
            return object()
        raise AssertionError(name)

    monkeypatch.setattr("training.train.importlib.util.find_spec", fake_find_spec)

    assert resolve_train_base_command() == [__import__("sys").executable, "-m", "piper.train"]
    assert calls[0] == "piper.train.vits"


def test_resolve_train_base_command_falls_back_to_piper_train(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPER_TRAIN_CMD", raising=False)

    def fake_find_spec(name: str):
        if name == "piper.train.vits":
            return None
        if name == "piper_train":
            return object()
        raise AssertionError(name)

    monkeypatch.setattr("training.train.importlib.util.find_spec", fake_find_spec)

    assert resolve_train_base_command() == [__import__("sys").executable, "-m", "piper_train"]


def test_resolve_train_base_command_raises_clear_error_when_training_module_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPER_TRAIN_CMD", raising=False)
    monkeypatch.setattr("training.train.importlib.util.find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="Не найден модуль обучения Piper"):
        resolve_train_base_command()


def test_run_training_builds_command_with_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_kwargs):
        captured["cmd"] = cmd

    monkeypatch.setattr("training.train.resolve_train_base_command", lambda: ["python", "-m", "my.custom.train"])
    monkeypatch.setattr("training.train.subprocess.run", fake_run)

    run_training(project_dir, epochs=1)

    assert captured["cmd"][:4] == ["python", "-m", "my.custom.train", "fit"]
