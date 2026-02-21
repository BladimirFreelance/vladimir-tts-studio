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
    monkeypatch.setitem(
        sys.modules,
        "app.doctor",
        types.SimpleNamespace(assert_training_preflight=lambda *_args, **_kwargs: None),
    )
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

    monkeypatch.setattr(
        "training.piper_train_bootstrap.validate_runtime_and_training_imports",
        lambda *_args, **_kwargs: None,
    )

    original_import_module = train_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "piper.espeakbridge":
            return object()
        return original_import_module(name)

    monkeypatch.setattr("training.train.importlib.import_module", fake_import_module)


def test_run_training_uses_bootstrap_module_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **_kwargs):
        captured["cmd"] = cmd

    monkeypatch.delenv("PIPER_TRAIN_CMD", raising=False)
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
        types.SimpleNamespace(
            check_manifest=lambda *_args, **_kwargs: {"path_fixed": 0},
            assert_training_preflight=lambda *_args, **_kwargs: None,
        ),
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


def test_run_training_dry_run_skips_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    called = {"subprocess": False}

    def _fail_if_called(*_args, **_kwargs) -> None:
        called["subprocess"] = True

    monkeypatch.setattr("training.train.subprocess.run", _fail_if_called)

    run_training(project_dir, epochs=1, dry_run=True)

    assert called["subprocess"] is False


def test_run_training_adds_resume_ckpt_flag(
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
    monkeypatch.setattr("training.train.ensure_stable_ckpt_aliases", lambda *_args, **_kwargs: None)

    run_training(project_dir, epochs=1, resume_ckpt="/tmp/resume.ckpt")

    resume_flags = {"--trainer.resume_from_checkpoint", "--ckpt_path"}
    flag = next((item for item in captured["cmd"] if item in resume_flags), None)
    assert flag is not None
    idx = captured["cmd"].index(flag)
    assert captured["cmd"][idx + 1] == "/tmp/resume.ckpt"


def test_run_training_copies_latest_checkpoint_to_output_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    def fake_run(*_args, **_kwargs) -> None:
        ckpt = project_dir / "runs" / "epoch=1-step=10.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("trained", encoding="utf-8")

    monkeypatch.setattr("training.train.subprocess.run", fake_run)
    monkeypatch.setattr("training.train.ensure_stable_ckpt_aliases", lambda *_args, **_kwargs: None)

    output_ckpt = tmp_path / "data" / "models" / "demo" / "demo.ckpt"
    run_training(project_dir, epochs=1, output_ckpt_path=output_ckpt)

    assert output_ckpt.exists()
    assert output_ckpt.read_text(encoding="utf-8") == "trained"


def test_run_training_updates_stable_aliases_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_dependencies(monkeypatch)
    project_dir = _prepare_project(tmp_path)

    captured: dict[str, Path] = {}

    monkeypatch.setattr("training.train.subprocess.run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "training.train.ensure_stable_ckpt_aliases",
        lambda runs_dir: captured.setdefault("runs_dir", runs_dir),
    )

    run_training(project_dir, epochs=1)

    assert captured["runs_dir"] == project_dir / "runs"
