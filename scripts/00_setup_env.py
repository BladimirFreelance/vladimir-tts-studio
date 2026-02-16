from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def run_capture(cmd: list[str]) -> str | None:
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


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

    if mode == "cu124":
        run(pip_cmd + ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"])
        return

    if mode == "rocm":
        run(pip_cmd + ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/rocm6.1"])
        return

    if mode == "directml":
        run(pip_cmd + ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
        run(pip_cmd + ["install", "torch-directml"])
        return

    # auto
    torch_mode = auto_detect_torch_mode()
    print(f"[auto] Выбран режим установки PyTorch: {torch_mode}")
    try:
        install_torch(pip_cmd, torch_mode)
    except subprocess.CalledProcessError:
        print(f"[auto] Режим '{torch_mode}' не установился, fallback на CPU...")
        install_torch(pip_cmd, "cpu")


def detect_nvidia_cuda_version() -> tuple[int, int] | None:
    output = run_capture(["nvidia-smi"])
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_windows_gpu_vendor() -> str | None:
    if os.name != "nt":
        return None
    output = run_capture(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]
    )
    if not output:
        return None
    names = output.lower()
    if "nvidia" in names:
        return "nvidia"
    if "amd" in names or "radeon" in names:
        return "amd"
    if "intel" in names:
        return "intel"
    return None


def auto_detect_torch_mode() -> str:
    system = platform.system().lower()
    cuda = detect_nvidia_cuda_version()
    if cuda:
        major, minor = cuda
        if major > 12 or (major == 12 and minor >= 4):
            return "cu124"
        if major >= 12:
            return "cu121"

    if system == "windows":
        vendor = detect_windows_gpu_vendor()
        if vendor in {"amd", "intel"}:
            return "directml"
        return "cpu"

    if system == "linux" and Path("/opt/rocm").exists():
        return "rocm"

    return "cpu"


def print_system_hints() -> None:
    if shutil.which("ffmpeg") is None:
        print("\n[!] ffmpeg не найден в PATH. Для авто-исправления аудио doctor-ом установите ffmpeg.")
    if shutil.which("espeak-ng") is None:
        print("[!] espeak-ng не найден в PATH. Для piper/espeakbridge установите espeak-ng.")
    if os.name == "nt":
        print("[i] Для обучения на Windows без WSL рекомендуется установить свежий драйвер GPU (NVIDIA/AMD/Intel).")



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
        choices=["auto", "cpu", "cu121", "cu124", "rocm", "directml", "skip"],
        default="auto",
        help="Как устанавливать PyTorch (по умолчанию auto)",
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
