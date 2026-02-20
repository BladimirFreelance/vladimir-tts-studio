from __future__ import annotations

import sys
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
    monkeypatch.setattr("training.train.detect_supported_gpu_or_raise", lambda **_kwargs: {})
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
    monkeypatch.setattr(
        "training.train.read_manifest",
        lambda _manifest: [("001.wav", "text")],
    )
    monkeypatch.setattr("training.train.write_json", lambda *args, **kwargs: None)


def test_run_training_uses_bootstrap_module_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_kwargs):
        captured["cmd"] = cmd

    monkeypatch.delenv("PIPER_TRAIN_CMD", raising=False)
    monkeypatch.setattr(
        "training.train.importlib.util.find_spec",
        lambda name: object() if name == "training.piper_train_bootstrap" else None,
    )
    monkeypatch.setattr("training.train.subprocess.run", fake_run)

    run_training(project_dir, epochs=1)

    assert captured["cmd"][:4] == [
        __import__("sys").executable,
        "-m",
        "training.piper_train_bootstrap",
        "fit",
    ]


def test_run_training_respects_custom_train_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_kwargs):
        captured["cmd"] = cmd

    monkeypatch.setenv("PIPER_TRAIN_CMD", "python -m my.custom.train")
    monkeypatch.setattr("training.train.subprocess.run", fake_run)

    run_training(project_dir, epochs=1)

    assert captured["cmd"][:4] == ["python", "-m", "my.custom.train", "fit"]


def test_run_training_uses_vocoder_warmstart_ckpt_and_not_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        "training.train.resolve_train_base_command",
        lambda: ["python", "-m", "training.piper_train_bootstrap"],
    )
    monkeypatch.setattr(
        "training.train.subprocess.run",
        lambda cmd, **_kwargs: captured.setdefault("cmd", cmd),
    )

    run_training(project_dir, epochs=1, base_ckpt="/tmp/base.ckpt")

    assert "--model.vocoder_warmstart_ckpt" in captured["cmd"]
    assert "/tmp/base.ckpt" in captured["cmd"]
    assert "--ckpt_path" not in captured["cmd"]


def test_run_training_allows_batch_size_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        "training.train.resolve_train_base_command",
        lambda: ["python", "-m", "training.piper_train_bootstrap"],
    )
    monkeypatch.setattr(
        "training.train.subprocess.run",
        lambda cmd, **_kwargs: captured.setdefault("cmd", cmd),
    )

    run_training(project_dir, epochs=1, batch_size=8)

    index = captured["cmd"].index("--data.batch_size")
    assert captured["cmd"][index + 1] == "8"


def test_run_training_checks_audio_in_audio_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)
    monkeypatch.setattr(
        "training.train.read_manifest",
        lambda _manifest: [("missing.wav", "text")],
    )
    monkeypatch.setitem(
        sys.modules,
        "app.doctor",
        types.SimpleNamespace(check_manifest=lambda *_args, **_kwargs: {"path_fixed": 0}),
    )

    with pytest.raises(RuntimeError, match="Manifest указывает на отсутствующие WAV"):
        run_training(project_dir, epochs=1)


def test_run_training_uses_basename_from_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)
    monkeypatch.setattr(
        "training.train.read_manifest",
        lambda _manifest: [("recordings/custom/nested/001.wav", "text")],
    )
    monkeypatch.setattr(
        "training.train.resolve_train_base_command",
        lambda: ["python", "-m", "training.piper_train_bootstrap"],
    )
    monkeypatch.setattr("training.train.subprocess.run", lambda *_args, **_kwargs: None)

    run_training(project_dir, epochs=1)


def test_run_training_passes_custom_audio_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)
    custom_audio_dir = project_dir / "audio_alt"
    custom_audio_dir.mkdir()
    (custom_audio_dir / "001.wav").write_bytes(b"RIFF")

    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        "training.train.resolve_train_base_command",
        lambda: ["python", "-m", "training.piper_train_bootstrap"],
    )
    monkeypatch.setattr(
        "training.train.subprocess.run",
        lambda cmd, **_kwargs: captured.setdefault("cmd", cmd),
    )

    run_training(project_dir, epochs=1, audio_dir=custom_audio_dir)

    index = captured["cmd"].index("--data.audio_dir")
    assert captured["cmd"][index + 1] == str(custom_audio_dir)


def test_run_training_warns_when_espeakbridge_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)
    monkeypatch.setattr(
        "training.train.ensure_espeakbridge_import",
        lambda: (_ for _ in ()).throw(RuntimeError("no espeakbridge")),
    )
    monkeypatch.setattr(
        "training.train.resolve_train_base_command",
        lambda: ["python", "-m", "training.piper_train_bootstrap"],
    )
    monkeypatch.setattr("training.train.subprocess.run", lambda *_args, **_kwargs: None)

    run_training(project_dir, epochs=1)

    assert "no espeakbridge" in caplog.text


def test_detect_supported_gpu_or_raise_uses_first_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda idx: ["RTX 5060 Ti", "RTX 3060"][idx],
            get_device_capability=lambda idx: [(12, 0), (8, 6)][idx],
            synchronize=lambda _idx: None,
        ),
        zeros=lambda *_args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("no kernel image is available")
        )
        if kwargs.get("device") == "cuda:0"
        else 0,
    )
    monkeypatch.setattr(train_module, "torch", fake_torch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    env_patch = train_module.detect_supported_gpu_or_raise()

    assert env_patch == {"CUDA_VISIBLE_DEVICES": "1"}


def test_detect_supported_gpu_or_raise_prefers_named_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda idx: ["RTX 4090", "RTX 3060"][idx],
            get_device_capability=lambda idx: [(8, 9), (8, 6)][idx],
            synchronize=lambda _idx: None,
        ),
        zeros=lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(train_module, "torch", fake_torch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    env_patch = train_module.detect_supported_gpu_or_raise(preferred_gpu_name="3060")

    assert env_patch == {"CUDA_VISIBLE_DEVICES": "1"}


def test_detect_supported_gpu_or_raise_supports_force_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    env_patch = train_module.detect_supported_gpu_or_raise(force_cpu=True)

    assert env_patch == {"CUDA_VISIBLE_DEVICES": ""}


def test_detect_supported_gpu_or_raise_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    env_patch = train_module.detect_supported_gpu_or_raise()
    assert env_patch == {}


def test_run_training_writes_audio_dir_and_detected_sample_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    custom_audio_dir = project_dir / "audio_alt"
    custom_audio_dir.mkdir()
    (custom_audio_dir / "001.wav").write_bytes(b"RIFF")

    captured_json: dict[str, object] = {}

    monkeypatch.setattr(
        "training.train.resolve_train_base_command",
        lambda: ["python", "-m", "training.piper_train_bootstrap"],
    )
    monkeypatch.setattr("training.train.subprocess.run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "dataset.audio_tools.inspect_wav",
        lambda _path: types.SimpleNamespace(sample_rate=16000),
    )
    monkeypatch.setattr(
        "training.train.write_json",
        lambda _path, payload: captured_json.setdefault("payload", payload),
    )

    run_training(project_dir, epochs=1, audio_dir=custom_audio_dir)

    payload = captured_json["payload"]
    assert payload["audio_dir"] == str(custom_audio_dir)
    assert payload["sample_rate"] == 16000


def test_run_training_allows_learning_rate_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        "training.train.resolve_train_base_command",
        lambda: ["python", "-m", "training.piper_train_bootstrap"],
    )
    monkeypatch.setattr(
        "training.train.subprocess.run",
        lambda cmd, **_kwargs: captured.setdefault("cmd", cmd),
    )

    run_training(project_dir, epochs=1, learning_rate=0.001)

    index = captured["cmd"].index("--model.learning_rate")
    assert captured["cmd"][index + 1] == "0.001"
