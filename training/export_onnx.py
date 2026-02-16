from __future__ import annotations

import json
import subprocess

from training.utils import ensure_espeakbridge_import
from pathlib import Path


def find_latest_ckpt(project_dir: Path) -> Path:
    ckpts = sorted(project_dir.glob("runs/**/*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError("No ckpt found in project runs folder")
    return ckpts[0]


def export_onnx(project: str, project_dir: Path, ckpt: Path | None = None) -> tuple[Path, Path]:
    ensure_espeakbridge_import()
    chosen = ckpt or find_latest_ckpt(project_dir)
    out_dir = Path("voices_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"ru_RU-{project}-medium.onnx"

    cmd = ["python", "-m", "piper.export_onnx", "--checkpoint", str(chosen), "--output", str(onnx_path)]
    subprocess.run(cmd, check=True)

    json_path = onnx_path.with_suffix(".onnx.json")
    payload = {
        "audio": {"sample_rate": 22050},
        "language": {"code": "ru_RU"},
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        "phoneme_type": "espeak",
        "espeak": {"voice": "ru"},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    smoke = out_dir / "export_smoke_test.wav"
    subprocess.run(
        [
            "python",
            "-m",
            "training.infer",
            "--model",
            str(onnx_path),
            "--text",
            "Проверка экспорта завершена успешно.",
            "--out",
            str(smoke),
        ],
        check=True,
    )
    return onnx_path, json_path
