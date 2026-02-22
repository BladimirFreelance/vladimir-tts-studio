from __future__ import annotations

import json
import subprocess
from pathlib import Path

from training.export_onnx import export_onnx


def test_export_prefers_best_alias_when_ckpt_not_provided(monkeypatch, tmp_path: Path) -> None:
    project = "demo"
    project_dir = tmp_path / "project"
    runs = project_dir / "runs"
    runs.mkdir(parents=True)
    best_alias = runs / "best.ckpt"
    latest = runs / "epoch=9.ckpt"
    best_alias.write_text("best", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")

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

    export_onnx(project, project_dir, ckpt=None)

    export_call = next(cmd for cmd in calls if "training.piper_export_onnx_compat" in cmd)
    checkpoint_value = export_call[export_call.index("--checkpoint") + 1]
    assert checkpoint_value == str(best_alias)


def test_export_uses_template_when_phonemize_missing(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    (project_dir / "runs").mkdir(parents=True)
    ckpt = project_dir / "runs" / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "ru_RU-dmitri-medium.onnx.json").write_text(
        json.dumps(
            {
                "num_symbols": 12,
                "num_speakers": 1,
                "phoneme_id_map": {"a": [1]},
                "audio": {"sample_rate": 16000},
                "language": {"code": "en_US"},
                "espeak": {"voice": "en"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("training.export_onnx.ensure_espeakbridge_import", lambda: None)
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd: list[str], check: bool) -> None:
        if "training.piper_export_onnx_compat" in cmd:
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"onnx")

    monkeypatch.setattr("training.export_onnx.subprocess.run", fake_run)

    _, json_path = export_onnx("demo", project_dir, ckpt=ckpt)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["num_symbols"] == 12
    assert payload["phoneme_id_map"] == {"a": [1]}


def test_export_smoke_test_failure_is_warning(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    runs = project_dir / "runs"
    runs.mkdir(parents=True)
    ckpt = runs / "best.ckpt"
    ckpt.write_text("x", encoding="utf-8")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "ru_RU-dmitri-medium.onnx.json").write_text(
        json.dumps(
            {
                "num_symbols": 12,
                "num_speakers": 1,
                "phoneme_id_map": {"a": [1]},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("training.export_onnx.ensure_espeakbridge_import", lambda: None)
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd: list[str], check: bool) -> None:
        if "training.piper_export_onnx_compat" in cmd:
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"onnx")
            return
        if "training.infer" in cmd:
            raise subprocess.CalledProcessError(1, cmd, "boom")

    monkeypatch.setattr("training.export_onnx.subprocess.run", fake_run)

    _, json_path = export_onnx("demo", project_dir, ckpt=ckpt)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "warning" in payload["status"]["smoke_test"]
