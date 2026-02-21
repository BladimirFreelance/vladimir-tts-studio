from __future__ import annotations

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
