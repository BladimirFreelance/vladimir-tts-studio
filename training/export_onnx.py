from __future__ import annotations

import json
import subprocess
import sys

from training.checkpoints import find_best_ckpt
from training.utils import ensure_espeakbridge_import
from pathlib import Path


def export_onnx(
    project: str, project_dir: Path, ckpt: Path | None = None
) -> tuple[Path, Path]:
    ensure_espeakbridge_import()
    runs_dir = project_dir / "runs"
    best_alias = runs_dir / "best.ckpt"
    chosen = ckpt or (best_alias if best_alias.exists() else find_best_ckpt(runs_dir))
    out_dir = Path("voices_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"ru_RU-{project}-medium.onnx"

    cmd = [
        sys.executable,
        "-m",
        "training.piper_export_onnx_compat",
        "--checkpoint",
        str(chosen),
        "--output",
        str(onnx_path),
    ]
    subprocess.run(cmd, check=True)

    json_path = onnx_path.with_suffix(".onnx.json")
    payload = {
        "audio": {"sample_rate": 22050},
        "language": {"code": "ru_RU"},
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        "phoneme_type": "espeak",
        "espeak": {"voice": "ru"},
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    smoke = out_dir / "export_smoke_test.wav"
    subprocess.run(
        [
            sys.executable,
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
