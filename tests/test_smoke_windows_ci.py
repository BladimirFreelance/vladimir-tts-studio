from __future__ import annotations

import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest


def _write_silence(path: Path, sample_rate: int = 22050, duration_seconds: float = 0.1) -> None:
    frame_count = int(sample_rate * duration_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(struct.pack("<h", 0) for _ in range(frame_count)))


def test_smoke_import_piper_train_vits() -> None:
    subprocess.run([sys.executable, "-c", "import piper.train.vits"], check=True)


def test_smoke_minimal_training_with_dummy_data(tmp_path: Path) -> None:
    from training.train import run_training

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "train_default.yaml"
    project_dir = tmp_path / "ci_smoke_project"
    metadata_dir = project_dir / "metadata"
    audio_dir = project_dir / "recordings" / "wav_22050"
    metadata_dir.mkdir(parents=True)

    _write_silence(audio_dir / "001.wav")
    _write_silence(audio_dir / "002.wav")
    (metadata_dir / "train.csv").write_text("001.wav|smoke one\n002.wav|smoke two\n", encoding="utf-8")

    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text(
        """
import pathlib
import sys

args = sys.argv[1:]
root = pathlib.Path(args[args.index('--trainer.default_root_dir') + 1])
root.mkdir(parents=True, exist_ok=True)
(root / 'ci-smoke.ckpt').write_text('ok', encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )

    env_key = "PIPER_TRAIN_CMD"
    previous = os.environ.get(env_key)
    os.environ[env_key] = f"{sys.executable} {fake_trainer}"
    try:
        run_training(project_dir, epochs=1, force_cpu=True, config_path=config_path)
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous

    assert (project_dir / "runs" / "ci-smoke.ckpt").exists()


def test_smoke_export_onnx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from training.export_onnx import export_onnx

    project = "ci-smoke"
    project_dir = tmp_path / "project"
    run_dir = project_dir / "runs"
    run_dir.mkdir(parents=True)
    ckpt_path = run_dir / "ci-smoke.ckpt"
    ckpt_path.write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr("training.export_onnx.ensure_espeakbridge_import", lambda: None)
    monkeypatch.setattr(
        "training.export_onnx.build_onnx_config",
        lambda _project_dir: {
            "num_symbols": 10,
            "num_speakers": 1,
            "phoneme_id_map": {"a": [1]},
            "audio": {"sample_rate": 22050},
            "language": {"code": "ru_RU"},
            "espeak": {"voice": "ru"},
            "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        },
    )

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        calls.append(cmd)
        if "training.piper_export_onnx_compat" in cmd:
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"onnx")
        if "training.infer" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF")

    monkeypatch.setattr("training.export_onnx.subprocess.run", fake_run)

    onnx_path, json_path = export_onnx(project, project_dir, ckpt=ckpt_path)

    assert onnx_path.exists()
    assert json_path.exists()
    assert any("training.piper_export_onnx_compat" in cmd for cmd in calls)
    assert any("training.infer" in cmd for cmd in calls)
