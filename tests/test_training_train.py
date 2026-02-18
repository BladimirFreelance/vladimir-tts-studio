from __future__ import annotations

import types
from pathlib import Path

import pytest

import training.train as train_module
from training.train import run_training


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
    monkeypatch.setattr("training.train.detect_supported_gpu_or_raise", lambda: None)
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


def test_run_training_uses_current_interpreter_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_kwargs):
        captured["cmd"] = cmd

    monkeypatch.delenv("PIPER_TRAIN_CMD", raising=False)
    monkeypatch.setattr("training.train.importlib.util.find_spec", lambda name: object() if name == "piper.train.vits" else None)
    monkeypatch.setattr("training.train.subprocess.run", fake_run)

    run_training(project_dir, epochs=1)

    assert captured["cmd"][:4] == [__import__("sys").executable, "-m", "piper.train", "fit"]


def test_run_training_respects_custom_train_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_kwargs):
        captured["cmd"] = cmd

    monkeypatch.setenv("PIPER_TRAIN_CMD", "python -m my.custom.train")
    monkeypatch.setattr("training.train.subprocess.run", fake_run)

    run_training(project_dir, epochs=1)

    assert captured["cmd"][:4] == ["python", "-m", "my.custom.train", "fit"]


def test_run_training_uses_vocoder_warmstart_ckpt_and_not_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr("training.train.resolve_train_base_command", lambda: ["python", "-m", "piper.train"])
    monkeypatch.setattr("training.train.subprocess.run", lambda cmd, **_kwargs: captured.setdefault("cmd", cmd))

    run_training(project_dir, epochs=1, base_ckpt="/tmp/base.ckpt")

    assert "--model.vocoder_warmstart_ckpt" in captured["cmd"]
    assert "/tmp/base.ckpt" in captured["cmd"]
    assert "--ckpt_path" not in captured["cmd"]


def test_run_training_allows_batch_size_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr("training.train.resolve_train_base_command", lambda: ["python", "-m", "piper.train"])
    monkeypatch.setattr("training.train.subprocess.run", lambda cmd, **_kwargs: captured.setdefault("cmd", cmd))

    run_training(project_dir, epochs=1, batch_size=8)

    index = captured["cmd"].index("--model.batch_size")
    assert captured["cmd"][index + 1] == "8"


def test_detect_supported_gpu_or_raise_reports_unsupported_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _idx: "Old GPU",
            get_device_capability=lambda _idx: (5, 0),
            synchronize=lambda _idx: None,
        ),
        zeros=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("CUDA error: no kernel image is available for execution on the device")),
        version=types.SimpleNamespace(cuda="12.1"),
    )
    monkeypatch.setattr(train_module, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="Old GPU"):
        train_module.detect_supported_gpu_or_raise()
