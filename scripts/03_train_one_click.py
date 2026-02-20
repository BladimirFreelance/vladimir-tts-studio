from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _choose_device(force_cpu: bool, preferred_gpu_name: str | None) -> dict[str, str]:
    from training.train import detect_supported_gpu_or_raise

    return detect_supported_gpu_or_raise(
        force_cpu=force_cpu,
        preferred_gpu_name=preferred_gpu_name,
    )


def _check_project_layout(project_dir: Path) -> None:
    required = [
        project_dir,
        project_dir / "metadata",
        project_dir / "metadata" / "train.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Проект не готов к обучению. Отсутствуют пути: " + ", ".join(missing)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click обучение: doctor, выбор GPU/CPU, train, опционально export/test"
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--gpu-name", dest="preferred_gpu_name")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--test-text")
    parser.add_argument("--test-out", default="voices_out/test_voice.wav")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_dir = Path("data/projects") / args.project
    _check_project_layout(project_dir)

    python = sys.executable
    _run([python, "-m", "app.main", "doctor", "--project", args.project])

    env = os.environ.copy()
    env.update(_choose_device(args.force_cpu, args.preferred_gpu_name))

    train_cmd = [
        python,
        "-m",
        "app.main",
        "train",
        "--project",
        args.project,
        "--epochs",
        str(args.epochs),
    ]
    if args.batch_size:
        train_cmd += ["--batch-size", str(args.batch_size)]
    if args.audio_dir:
        train_cmd += ["--audio-dir", str(args.audio_dir)]
    if args.force_cpu:
        train_cmd += ["--force_cpu"]
    if args.preferred_gpu_name:
        train_cmd += ["--gpu-name", args.preferred_gpu_name]

    _run(train_cmd, env=env)

    if args.export_onnx:
        _run([python, "-m", "app.main", "export", "--project", args.project], env=env)

    if args.test_text:
        model_path = project_dir / "export" / f"{args.project}.onnx"
        _run(
            [
                python,
                "-m",
                "app.main",
                "test",
                "--model",
                str(model_path),
                "--text",
                args.test_text,
                "--out",
                args.test_out,
                "--mode",
                "espeak",
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
