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
    """
    Возвращает команду для pip, привязанного к указанному venv.

    На Windows вызов `pip.exe` из одного окружения, когда активировано другое,
    может падать с ошибкой вида "To modify pip, please run ... python -m pip".
    Поэтому приоритетно вызываем pip через python интерпретатор самого venv.
    """
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists():
        return [str(venv_python), "-m", "pip"]

    # fallback, если структура venv неожиданная
    return [sys.executable, "-m", "pip"]


def resolve_python(venv_dir: Path) -> list[str]:
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists():
        return [str(venv_python)]

    return [sys.executable]


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
    install_torch_auto(pip_cmd)


def install_torch_auto(pip_cmd: list[str]) -> None:
    torch_mode = auto_detect_torch_mode()
    attempts: list[str] = [torch_mode]

    if torch_mode == "cu124":
        attempts.extend(["cu121", "cpu"])
    elif torch_mode == "cu121":
        attempts.append("cpu")
    elif torch_mode == "rocm":
        attempts.append("cpu")
    elif torch_mode == "directml":
        attempts.append("cpu")
    else:
        attempts.append("cpu")

    if platform.system().lower() == "windows" and "directml" not in attempts:
        attempts.insert(-1, "directml")

    tried: list[str] = []
    for candidate in attempts:
        if candidate in tried:
            continue
        tried.append(candidate)
        try:
            print(f"[auto] Пробую установить PyTorch в режиме: {candidate}")
            install_torch(pip_cmd, candidate)
            print(f"[auto] Установка PyTorch завершена в режиме: {candidate}")
            return
        except subprocess.CalledProcessError:
            print(f"[auto] Режим '{candidate}' не установился.")

    raise subprocess.CalledProcessError(returncode=1, cmd=[*pip_cmd, "install", "torch"])


def detect_nvidia_cuda_version() -> tuple[int, int] | None:
    output = run_capture(["nvidia-smi"])
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_nvidia_gpu_name() -> str | None:
    output = run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not output:
        return None
    return output.splitlines()[0].strip().lower()


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
        gpu_name = detect_nvidia_gpu_name() or ""
        major, minor = cuda
        # Ampere (например RTX 3060) обычно стабильнее на cu121 wheel,
        # а при более новых драйверах можно пробовать cu124.
        if "rtx 3060" in gpu_name:
            return "cu121"
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


def module_available(python_cmd: list[str], module_name: str) -> bool:
    script = f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    completed = subprocess.run(python_cmd + ["-c", script], check=False)
    return completed.returncode == 0


def install_piper_training(pip_cmd: list[str], python_cmd: list[str], source: str) -> None:
    print("[i] Устанавливаю Piper training-модули...")
    attempts = [
        source,
        "piper-train",
    ]

    for candidate in attempts:
        try:
            print(f"[i] Попытка установки Piper training из: {candidate}")
            run(pip_cmd + ["install", candidate])
        except subprocess.CalledProcessError:
            print(f"[!] Не удалось установить источник: {candidate}")

        if module_available(python_cmd, "piper.train.vits") or module_available(python_cmd, "piper_train"):
            print("[i] Piper training-модули доступны (piper.train.vits или piper_train).")
            return

    raise RuntimeError("Не удалось установить Piper training-модули (piper.train.vits/piper_train)")


def install_piper_runtime(pip_cmd: list[str], python_cmd: list[str]) -> None:
    print("[i] Устанавливаю Piper runtime (piper-tts)...")
    run(pip_cmd + ["install", "piper-tts"])
    if not module_available(python_cmd, "piper.espeakbridge"):
        raise RuntimeError("Piper runtime установлен некорректно: модуль piper.espeakbridge не найден")

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
        default=[],
        help="Дополнительные pip-пакеты для установки",
    )
    parser.add_argument(
        "--without-piper-training",
        action="store_true",
        help="Не устанавливать training-модули Piper",
    )
    parser.add_argument(
        "--piper-training-source",
        default="git+https://github.com/rhasspy/piper.git#subdirectory=src/python",
        help="Источник для установки training-модулей Piper",
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
        python_cmd = [sys.executable]
        print("Использую текущее окружение Python")
    else:
        venv_dir = Path(args.venv)
        pip_cmd = ensure_venv(venv_dir)
        python_cmd = resolve_python(venv_dir)

    run(pip_cmd + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    run(pip_cmd + ["install", "-r", "requirements.txt"])
    run(pip_cmd + ["install", "-e", "."])

    install_torch(pip_cmd, args.torch)
    install_piper_runtime(pip_cmd, python_cmd)

    if args.extras:
        run(pip_cmd + ["install", *args.extras])

    if not args.without_piper_training:
        install_piper_training(pip_cmd, python_cmd, args.piper_training_source)

    print_system_hints()

    print("\nГотово. Рекомендуется проверить окружение:")
    print("python scripts/06_doctor.py --project <project_name> --auto-fix")
    if args.without_piper_training:
        print("[i] Для обучения Piper при необходимости: python scripts/00_setup_env.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
