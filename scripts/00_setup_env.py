from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PIPER_GPL_REPO = "https://github.com/OHF-Voice/piper1-gpl.git"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n>>> {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def run_result(
    cmd: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    print(f"\n>>> {shlex.join(cmd)}")
    return subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)


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
        run(
            pip_cmd
            + [
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        return

    if mode == "cu121":
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

    if mode == "cu124":
        run(
            pip_cmd
            + [
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ]
        )
        return

    if mode == "rocm":
        run(
            pip_cmd
            + [
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/rocm6.1",
            ]
        )
        return

    if mode == "directml":
        run(
            pip_cmd
            + [
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
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

    raise subprocess.CalledProcessError(
        returncode=1, cmd=[*pip_cmd, "install", "torch"]
    )


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
    cuda = detect_nvidia_cuda_version()
    gpu_name = detect_nvidia_gpu_name() or ""
    if cuda:
        major, minor = cuda
        if (major, minor) >= (12, 4):
            print("[auto] Обнаружен NVIDIA GPU (CUDA >= 12.4): выбираю cu124")
            return "cu124"
        if (major, minor) >= (12, 1):
            print("[auto] Обнаружен NVIDIA GPU (CUDA >= 12.1): выбираю cu121")
            return "cu121"
        print("[auto] Обнаружен NVIDIA GPU, но версия CUDA < 12.1: выбираю CPU")
        return "cpu"

    if "nvidia" in gpu_name:
        print("[auto] Обнаружен NVIDIA GPU: выбираю cu121")
        return "cu121"

    vendor = detect_windows_gpu_vendor()
    if vendor in {"amd", "intel"}:
        print(f"[auto] Обнаружен {vendor.upper()} GPU на Windows: выбираю directml")
        return "directml"

    print("[auto] GPU не обнаружен: выбираю CPU-сборку PyTorch")
    return "cpu"


def print_torch_summary(python_cmd: list[str]) -> None:
    summary_code = (
        "import torch;"
        "print(f'torch={torch.__version__}');"
        "print(f'cuda_available={torch.cuda.is_available()}');"
        "print(f'torch_cuda_build={getattr(torch.version, \"cuda\", None)}')"
    )
    completed = run_result(python_cmd + ["-c", summary_code])
    if completed.returncode != 0:
        print("[!] Не удалось проверить установленный torch")
        if completed.stderr:
            print(completed.stderr)


def module_available(python_cmd: list[str], module_name: str) -> bool:
    script = f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    completed = subprocess.run(python_cmd + ["-c", script], check=False)
    return completed.returncode == 0


def clone_or_update_piper_training_repo(target_dir: Path) -> None:
    if target_dir.exists():
        print(f"[i] Обновляю репозиторий Piper training: {target_dir}")
        run(["git", "-C", str(target_dir), "fetch", "origin", "main"])
        run(["git", "-C", str(target_dir), "reset", "--hard", "origin/main"])
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[i] Клонирую Piper training в: {target_dir}")
    run(["git", "clone", PIPER_GPL_REPO, str(target_dir)])


def verify_piper_training_install(python_cmd: list[str]) -> bool:
    probe = (
        "from pathlib import Path;"
        "import piper;"
        "train_path = Path('third_party/piper1-gpl/src/piper').resolve();"
        "piper.__path__.append(str(train_path)) if str(train_path) not in piper.__path__ else None;"
        "import piper.train.vits as v;"
        "print('OK', v.__file__)"
    )
    completed = run_result(python_cmd + ["-c", probe])
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr)
        return False
    print(completed.stdout.strip())
    return True


def install_piper_training(
    python_cmd: list[str],
    repo_root: Path,
    *,
    allow_missing: bool = False,
) -> bool:
    print("[i] Подготавливаю Piper training-модули...")
    repo_dir = repo_root / "third_party" / "piper1-gpl"

    try:
        clone_or_update_piper_training_repo(repo_dir)
    except subprocess.CalledProcessError as error:
        print(f"[!] Не удалось подготовить локальный clone piper1-gpl: {error}")
        if allow_missing:
            print("[WARN] Piper training missing")
            return False
        raise RuntimeError("Не удалось клонировать/обновить third_party/piper1-gpl") from error

    if verify_piper_training_install(python_cmd):
        print("[OK] Piper training available: piper.train.vits")
        return True

    msg = "Не удалось подготовить Piper training-модули (требуется piper.train.vits)."
    if allow_missing:
        print(f"[WARN] {msg}")
        print(
            "[!] Продолжаю без training-модулей. Синтез (piper-tts) будет работать, обучение — нет."
        )
        print(
            "[!] Для строгой проверки запустите: python scripts/00_setup_env.py --require-piper-training"
        )
        return False

    raise RuntimeError(f"""[FAIL] Piper training missing
{msg}
Команды ручного восстановления:
git clone {PIPER_GPL_REPO} third_party/piper1-gpl
{shlex.join(python_cmd)} -c \"import piper; from pathlib import Path; p=Path('third_party/piper1-gpl/src/piper').resolve(); piper.__path__.append(str(p)); import piper.train.vits as v; print('OK', v.__file__)\"""")


def install_piper_runtime(pip_cmd: list[str], python_cmd: list[str]) -> None:
    print("[i] Устанавливаю Piper runtime (piper-tts)...")
    run(pip_cmd + ["install", "piper-tts"])
    if not module_available(python_cmd, "piper.espeakbridge"):
        raise RuntimeError(
            "Piper runtime установлен некорректно: модуль piper.espeakbridge не найден"
        )


def print_system_hints() -> None:
    if shutil.which("ffmpeg") is None:
        print(
            "\n[!] ffmpeg не найден в PATH. Для авто-исправления аудио doctor-ом установите ffmpeg."
        )
    if shutil.which("espeak-ng") is None:
        print(
            "[!] espeak-ng не найден в PATH. Для piper/espeakbridge установите espeak-ng."
        )
    if os.name == "nt":
        print(
            "[i] Для обучения на Windows без WSL рекомендуется установить свежий драйвер GPU (NVIDIA/AMD/Intel)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Автоматическая установка зависимостей проекта vladimir-piper-voice-lab"
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Папка виртуального окружения (по умолчанию .venv)",
    )
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
        "--with-piper-training",
        action="store_true",
        help="Явно включить установку training-модулей Piper (алиас к поведению по умолчанию)",
    )
    parser.add_argument(
        "--piper-training-source",
        default="third_party/piper1-gpl",
        help="Устаревший аргумент (игнорируется, training берётся из third_party/piper1-gpl через bootstrap)",
    )
    parser.add_argument(
        "--require-piper-training",
        action="store_true",
        help="Завершать setup с ошибкой, если training-модули Piper не удалось установить",
    )
    return parser.parse_args()


def main() -> int:
    if sys.version_info < (3, 11):
        print("Нужен Python >= 3.11")
        return 2

    args = parse_args()
    if args.with_piper_training and args.without_piper_training:
        raise SystemExit("Нельзя использовать одновременно --with-piper-training и --without-piper-training")

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
    print_torch_summary(python_cmd)
    install_piper_runtime(pip_cmd, python_cmd)

    if args.extras:
        run(pip_cmd + ["install", *args.extras])

    install_training = args.with_piper_training or not args.without_piper_training
    if install_training:
        try:
            install_piper_training(
                python_cmd,
                repo_root,
                allow_missing=not args.require_piper_training,
            )
        except RuntimeError as error:
            print(error)
            return 1
    elif not verify_piper_training_install(python_cmd):
        if args.require_piper_training:
            print("[FAIL] Piper training missing (указан --require-piper-training)")
            return 1
        print("[WARN] Piper training missing")

    if args.require_piper_training and not verify_piper_training_install(python_cmd):
        print("[FAIL] Не удалось импортировать piper.train.vits после установки.")
        print("[HINT] Запустите из активированного .venv:")
        print("python scripts/00_setup_env.py --require-piper-training")
        return 1

    print_system_hints()

    print("\nГотово. Рекомендуется проверить окружение:")
    print("python scripts/06_doctor.py --project PROJECT_NAME --auto-fix")
    if not install_training:
        print(
            "[i] Для обучения Piper при необходимости: python scripts/00_setup_env.py --with-piper-training"
        )
    print(
        "[i] Проверка training после установки: python -m training.piper_train_bootstrap --help"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
