from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def resolve_pip(venv_dir: Path) -> list[str]:
    if os.name == "nt":
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        pip_path = venv_dir / "bin" / "pip"

    if pip_path.exists():
        return [str(pip_path)]
    return [sys.executable, "-m", "pip"]


def ensure_venv(venv_dir: Path) -> list[str]:
    if not venv_dir.exists():
        print(f"Создаю виртуальное окружение: {venv_dir}")
        run([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        print(f"Виртуальное окружение уже существует: {venv_dir}")
    return resolve_pip(venv_dir)


def install_torch(pip_cmd: list[str], mode: str) -> None:
    if mode == "skip":
        print("Пропускаю установку PyTorch (--torch skip)")
        return

    if mode == "cpu":
        run(pip_cmd + ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
        return

    if mode == "cu121":
        run(pip_cmd + ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
        return

    # auto
    is_windows_or_linux = platform.system().lower() in {"windows", "linux"}
    if is_windows_or_linux:
        try:
            run(
                pip_cmd
                + [
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu121",
                ]
            )
            return
        except subprocess.CalledProcessError:
            print("CUDA-сборка PyTorch не установилась, пробую CPU-сборку...")

    run(pip_cmd + ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])


def print_system_hints() -> None:
    if shutil.which("ffmpeg") is None:
        print("\n[!] ffmpeg не найден в PATH. Для авто-исправления аудио doctor-ом установите ffmpeg.")
    if shutil.which("espeak-ng") is None:
        print("[!] espeak-ng не найден в PATH. Для piper/espeakbridge установите espeak-ng.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Автоматическая установка зависимостей проекта vladimir-piper-voice-lab"
    )
    parser.add_argument("--venv", default=".venv", help="Папка виртуального окружения (по умолчанию .venv)")
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Не создавать venv, ставить зависимости в текущее окружение",
    )
    parser.add_argument(
        "--torch",
        choices=["auto", "cpu", "cu121", "skip"],
        default="cpu",
        help="Как устанавливать PyTorch (по умолчанию cpu)",
    )
    parser.add_argument(
        "--extras",
        nargs="*",
        default=["piper-tts"],
        help="Дополнительные pip-пакеты для установки (по умолчанию: piper-tts)",
    )
    return parser.parse_args()



def main() -> int:
    if sys.version_info < (3, 11):
        print("Нужен Python >= 3.11")
        return 2

    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    if args.no_venv:
        pip_cmd = [sys.executable, "-m", "pip"]
        print("Использую текущее окружение Python")
    else:
        pip_cmd = ensure_venv(Path(args.venv))

    run(pip_cmd + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    run(pip_cmd + ["install", "-r", "requirements.txt"])
    run(pip_cmd + ["install", "-e", "."])

    install_torch(pip_cmd, args.torch)

    if args.extras:
        run(pip_cmd + ["install", *args.extras])

    print_system_hints()

    print("\nГотово. Рекомендуется проверить окружение:")
    print("python scripts/06_doctor.py --project <project_name> --auto-fix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
