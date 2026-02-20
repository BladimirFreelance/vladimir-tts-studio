from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_venv_python(repo_root: Path) -> Path:
    candidates = [
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Не найден Python внутри .venv. Ожидались пути: "
        + ", ".join(str(path) for path in candidates)
    )


def _choose_device_env(*, force_cpu: bool) -> dict[str, str]:
    if force_cpu:
        print("[gpu] --force_cpu включён, запускаю на CPU")
        return {"CUDA_VISIBLE_DEVICES": ""}

    try:
        import torch
    except ImportError:
        print("[gpu] torch не установлен, запускаю обучение на CPU")
        return {"CUDA_VISIBLE_DEVICES": ""}

    if not torch.cuda.is_available():
        print("[gpu] CUDA недоступна, запускаю обучение на CPU")
        return {"CUDA_VISIBLE_DEVICES": ""}

    for index in range(torch.cuda.device_count()):
        try:
            torch.zeros(1, device=f"cuda:{index}")
            torch.cuda.synchronize(index)
            name = torch.cuda.get_device_name(index)
            if index == 0:
                print(f"[gpu] выбран GPU #0: {name}")
                return {}

            print(
                f"[gpu] GPU #0 пропущен, выбран GPU #{index}: {name}; "
                f"ставлю CUDA_VISIBLE_DEVICES={index}"
            )
            return {"CUDA_VISIBLE_DEVICES": str(index)}
        except RuntimeError as exc:
            name = torch.cuda.get_device_name(index)
            print(f"[gpu] пропуск GPU #{index} ({name}): {exc}")

    print("[gpu] не найден совместимый GPU, запускаю обучение на CPU")
    return {"CUDA_VISIBLE_DEVICES": ""}


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
        description=(
            "One-click обучение: doctor, выбор GPU/CPU, запуск train через .venv"
        )
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--force_cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = _resolve_repo_root()
    project_dir = repo_root / "data" / "projects" / args.project
    _check_project_layout(project_dir)

    python = _resolve_venv_python(repo_root)

    _run([str(python), "-m", "app.main", "doctor", "--project", args.project])

    env = os.environ.copy()
    env.update(_choose_device_env(force_cpu=args.force_cpu))

    train_cmd = [
        str(python),
        "-m",
        "app.main",
        "train",
        "--project",
        args.project,
        "--epochs",
        str(args.max_epochs),
    ]

    if args.batch_size:
        train_cmd += ["--batch-size", str(args.batch_size)]
    if args.lr is not None:
        train_cmd += ["--lr", str(args.lr)]
    if args.force_cpu:
        train_cmd += ["--force_cpu"]

    _run(train_cmd, env=env)


if __name__ == "__main__":
    main()
