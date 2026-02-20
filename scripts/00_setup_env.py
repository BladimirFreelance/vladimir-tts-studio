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


def detect_windows_shell() -> str:
    if os.name != "nt":
        return "unknown"
    if os.environ.get("PSModulePath") or os.environ.get("POWERSHELL_DISTRIBUTION_CHANNEL"):
        return "powershell"
    return "cmd"


def format_venv_python_for_help() -> str:
    if os.name == "nt":
        return r".\.venv\Scripts\python.exe"
    return "./.venv/bin/python"


def format_install_command_for_help(*args: str) -> str:
    python_exec = format_venv_python_for_help()
    if os.name == "nt" and detect_windows_shell() == "powershell":
        return f"& {python_exec} {' '.join(args)}"
    return f"{python_exec} {' '.join(args)}"


def resolve_pip(venv_dir: Path) -> list[str]:
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists():
        return [str(venv_python), "-m", "pip"]

    raise FileNotFoundError(f"Не найден python внутри venv: {venv_dir}")


def run_install_with_espeakbridge_tolerance(pip_cmd: list[str], install_args: list[str]) -> None:
    cmd = pip_cmd + install_args
    completed = run_result(cmd)
    if completed.returncode == 0:
        return

    combined_output = f"{completed.stdout}\n{completed.stderr}".lower()
    if "espeakbridge" in combined_output:
        print("[WARN] Установка упала на espeakbridge. Повторяю без зависимостей (--no-deps).")
        run(pip_cmd + install_args + ["--no-deps"])
        return

    raise subprocess.CalledProcessError(returncode=completed.returncode, cmd=cmd)


def resolve_python(venv_dir: Path) -> list[str]:
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists():
        return [str(venv_python)]

    raise FileNotFoundError(f"Не найден python внутри venv: {venv_dir}")


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

    install_torch_auto(pip_cmd)


def install_torch_auto(pip_cmd: list[str]) -> None:
    torch_mode = auto_detect_torch_mode()
    attempts: list[str] = [torch_mode]

    if torch_mode == "cu124":
        attempts.extend(["cu121", "cpu"])
    elif torch_mode in {"cu121", "rocm", "directml"}:
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
    probe = (
        "import importlib.util,sys;"
        f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
    )
    completed = subprocess.run(python_cmd + ["-c", probe], check=False)
    return completed.returncode == 0


def check_piper_modules(python_cmd: list[str]) -> tuple[bool, bool]:
    has_base = module_available(python_cmd, "piper")

    probe_training = (
        "from training.piper_train_bootstrap import validate_runtime_and_training_imports;"
        "validate_runtime_and_training_imports()"
    )
    completed = run_result(python_cmd + ["-c", probe_training])
    has_training = completed.returncode == 0
    return has_base, has_training


def warn_if_espeakbridge_missing(python_cmd: list[str]) -> None:
    probe = "import importlib; importlib.import_module('piper.espeakbridge'); print('[OK] piper.espeakbridge importable')"
    completed = run_result(python_cmd + ["-c", probe])
    if completed.returncode != 0:
        print("[WARN] piper.espeakbridge недоступен")


def _windows_build_tools_hints() -> list[str]:
    hints: list[str] = []
    if os.name != "nt":
        return hints

    if shutil.which("cl") is None:
        hints.append(
            "- (optional advanced) Visual Studio 2022 Build Tools (C++ workload): https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        )
    if shutil.which("cmake") is None:
        hints.append("- (optional advanced) CMake: https://cmake.org/download/")
    if shutil.which("ninja") is None:
        hints.append("- (optional advanced) Ninja: pip install ninja")
    if shutil.which("espeak-ng") is None:
        hints.append("- Установите espeak-ng и добавьте в PATH: https://github.com/espeak-ng/espeak-ng/releases")
    return hints


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
    has_base, has_training = check_piper_modules(python_cmd)
    install_hint = format_install_command_for_help(
        "-m", "pip", "install", "piper-tts==1.4.1"
    )
    if not has_base:
        print(f"[FAIL] Не найден runtime пакет piper (piper-tts). Выполните: {install_hint}")
        return False
    if not has_training:
        print("[FAIL] Не удалось импортировать piper.train.vits через training bootstrap.")
        print(
            "[HINT] Проверьте исходники third_party/piper1-gpl и повторите: "
            + format_install_command_for_help("scripts/00_setup_env.py", "--require-piper-training")
        )
        return False
    print("[OK] Piper runtime + training bootstrap доступны")
    return True


def install_piper_training(pip_cmd: list[str], python_cmd: list[str], repo_root: Path, *, allow_missing: bool = False) -> bool:
    print("[i] Подготавливаю Piper training-модули...")

    run(pip_cmd + ["install", "piper-tts==1.4.1"])

    if verify_piper_training_install(python_cmd):
        return True

    msg = "Не удалось подготовить Piper training-модули (runtime piper-tts + piper.train.vits через bootstrap)."
    if allow_missing:
        print(f"[WARN] {msg}")
        print("[!] Продолжаю без training-модулей. Синтез (piper-tts) будет работать, обучение — нет.")
        print(
            "[!] Для строгой проверки запустите: "
            + format_install_command_for_help("scripts/00_setup_env.py", "--require-piper-training")
        )
        return False

    raise RuntimeError(
        f"""[FAIL] Piper training missing
{msg}
Команды ручного восстановления:
git clone {PIPER_GPL_REPO} third_party/piper1-gpl
{format_install_command_for_help('-m', 'pip', 'install', 'piper-tts==1.4.1')}"""
    )


def install_piper_runtime(pip_cmd: list[str], python_cmd: list[str], repo_root: Path) -> None:
    _ = (pip_cmd, repo_root)
    print("[i] Проверяю Piper runtime (piper-tts)...")

    has_base, _ = check_piper_modules(python_cmd)
    if not has_base:
        raise RuntimeError("[FAIL] Базовый модуль piper не найден после установки runtime-зависимостей.")
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
    parser.add_argument(
        "--mode",
        choices=["runtime", "training"],
        default="training",
        help="Режим установки зависимостей: runtime (синтез) или training (обучение)",
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Устаревший аргумент. Всегда используется .venv",
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
        help="Устаревший аргумент (игнорируется, всегда используется third_party/piper1-gpl)",
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

    if args.venv != ".venv":
        print(f"[WARN] Аргумент --venv={args.venv} игнорируется. Установка всегда выполняется в .venv")

    venv_dir = Path(".venv")
    pip_cmd = ensure_venv(venv_dir)
    python_cmd = resolve_python(venv_dir)

    active_venv = os.environ.get("VIRTUAL_ENV")
    if active_venv and Path(active_venv).resolve() != venv_dir.resolve():
        print(
            f"[WARN] Активировано другое окружение: {active_venv}. "
            f"Установка будет выполняться в {venv_dir.resolve()}"
        )

    run(pip_cmd + ["install", "--upgrade", "pip", "setuptools", "wheel"])

    install_training = args.mode == "training"
    if args.with_piper_training:
        install_training = True
    if args.without_piper_training:
        install_training = False

    if install_training:
        try:
            clone_or_update_piper_training_repo(repo_root / "third_party" / "piper1-gpl")
        except subprocess.CalledProcessError as error:
            print(f"[!] Не удалось подготовить локальный clone piper1-gpl: {error}")
            if args.require_piper_training:
                return 1

    install_torch(pip_cmd, args.torch)
    print_torch_summary(python_cmd)

    requirements_path = "requirements/train.txt" if install_training else "requirements/runtime.txt"
    run(pip_cmd + ["install", "-r", requirements_path])
    run(pip_cmd + ["install", "-e", "."])

    if args.extras:
        run(pip_cmd + ["install", *args.extras])

    if install_training:
        try:
            install_piper_training(pip_cmd, python_cmd, repo_root, allow_missing=not args.require_piper_training)
        except RuntimeError as error:
            print(error)
            return 1
        warn_if_espeakbridge_missing(python_cmd)
        run(
            python_cmd
            + [
                "-c",
                "from training.piper_train_bootstrap import validate_runtime_and_training_imports; validate_runtime_and_training_imports()",
            ]
        )
    elif not verify_piper_training_install(python_cmd):
        install_piper_runtime(pip_cmd, python_cmd, repo_root)
        warn_if_espeakbridge_missing(python_cmd)
        if args.require_piper_training:
            print("[FAIL] Piper training missing (указан --require-piper-training)")
            return 1
        print("[WARN] Piper training missing")
    else:
        warn_if_espeakbridge_missing(python_cmd)

    if args.require_piper_training and not verify_piper_training_install(python_cmd):
        print("[FAIL] Не удалось импортировать piper.train.vits после установки.")
        print("[HINT] Запустите:")
        print(format_install_command_for_help("scripts/00_setup_env.py", "--require-piper-training"))
        return 1

    print_system_hints()

    print("\nГотово. Рекомендуется проверить окружение:")
    print(format_install_command_for_help("scripts/06_doctor.py", "--auto-fix"))
    if not install_training:
        print(
            "[i] Для обучения Piper при необходимости: "
            + format_install_command_for_help("scripts/00_setup_env.py", "--with-piper-training")
        )
    print(
        "[i] Проверка training после установки: "
        + format_install_command_for_help("-m", "training.piper_train_bootstrap", "--help")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
