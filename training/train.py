from __future__ import annotations

import importlib.util
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for pre-flight diagnostics
    torch = None

from dataset.manifest import read_manifest, resolve_audio_path
from training.utils import ensure_espeakbridge_import
from utils import load_yaml, write_json

LOGGER = logging.getLogger(__name__)


def discover_warmstart() -> str | None:
    possible = [
        Path("models") / "ru_RU-dmitri-medium.ckpt",
        Path.home() / ".local" / "share" / "piper" / "ru_RU-dmitri-medium.ckpt",
    ]

    appdata = os.getenv("APPDATA")
    local_appdata = os.getenv("LOCALAPPDATA")
    if appdata:
        possible.append(Path(appdata) / "piper" / "ru_RU-dmitri-medium.ckpt")
    if local_appdata:
        possible.append(Path(local_appdata) / "piper" / "ru_RU-dmitri-medium.ckpt")

    for item in possible:
        if item.exists():
            return str(item)
    return None


def resolve_train_base_command() -> list[str]:
    """Resolve Piper training entrypoint for the current environment."""
    train_cmd = os.getenv("PIPER_TRAIN_CMD")
    if train_cmd:
        return shlex.split(train_cmd)

    module_candidates = [
        ("piper.train.vits", "piper.train"),
        ("piper_train", "piper_train"),
    ]
    for probe_module, cli_module in module_candidates:
        if importlib.util.find_spec(probe_module) is not None:
            return [sys.executable, "-m", cli_module]

    raise RuntimeError(
        "Не найден модуль обучения Piper. Текущее окружение содержит runtime piper-tts, "
        "но не training-компоненты. Установите training-сборку Piper или укажите команду "
        "через PIPER_TRAIN_CMD (пример: python -m piper_train)."
    )


def detect_supported_gpu_or_raise() -> None:
    if torch is None or not torch.cuda.is_available():
        return

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)

    try:
        torch.zeros(1, device=f"cuda:{device_index}")
        torch.cuda.synchronize(device_index)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "no kernel image is available" in message or "not compatible with the current pytorch installation" in message:
            capability = torch.cuda.get_device_capability(device_index)
            cuda_version = getattr(torch.version, "cuda", "unknown")
            raise RuntimeError(
                "Обнаружен GPU '%s' (compute capability %s.%s), но текущая сборка PyTorch/CUDA его не поддерживает. "
                "Установите совместимую версию PyTorch или используйте другой GPU. (torch CUDA=%s)"
                % (device_name, capability[0], capability[1], cuda_version)
            ) from exc
        raise


def run_training(
    project_dir: Path,
    epochs: int,
    base_ckpt: str | None = None,
    config_path: Path | None = None,
    batch_size: int | None = None,
) -> None:
    detect_supported_gpu_or_raise()
    ensure_espeakbridge_import()
    cfg = load_yaml(config_path or Path("configs/train_default.yaml"))
    manifest = project_dir / "metadata" / "train.csv"
    rows = read_manifest(manifest)
    missing = [audio for audio, _ in rows if not resolve_audio_path(project_dir, audio).exists()]
    existing = len(rows) - len(missing)
    if existing <= 0:
        raise RuntimeError(
            "Обнаружено 0 utterances. Проверьте запись в студии и что WAV-файлы существуют в recordings/wav_22050. "
            f"Запустите scripts/06_doctor.py --project {project_dir.name} --auto-fix для диагностики."
        )

    data_config = project_dir / "metadata" / "data_config.json"
    write_json(
        data_config,
        {
            "dataset": "local",
            "manifest": str(manifest),
            "sample_rate": cfg["project_defaults"]["sample_rate"],
            "phoneme_type": cfg["training"]["phoneme_type"],
            "espeak_voice": cfg["training"]["espeak_voice"],
        },
    )

    runs_dir = project_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cmd = resolve_train_base_command() + ["fit"]
    cmd += ["--data.config_path", str(data_config)]
    cmd += ["--data.dataset", "jsonl"]
    cmd += ["--max_epochs", str(epochs)]
    cmd += ["--trainer.default_root_dir", str(runs_dir)]
    cmd += ["--data.phoneme_type", cfg["training"]["phoneme_type"]]
    cmd += ["--data.espeak_voice", cfg["training"]["espeak_voice"]]
    resolved_batch_size = batch_size or int(cfg["training"]["batch_size"])
    cmd += ["--model.batch_size", str(resolved_batch_size)]
    cmd += ["--model.learning_rate", str(cfg["training"]["learning_rate"])]

    selected_ckpt = base_ckpt or (discover_warmstart() if cfg["project_defaults"].get("use_warmstart_if_found") else None)
    if selected_ckpt:
        cmd += ["--model.vocoder_warmstart_ckpt", selected_ckpt]

    command_path = runs_dir / "run_command.txt"
    command_path.write_text(" ".join(shlex.quote(part) for part in cmd), encoding="utf-8")

    LOGGER.info("Launching training: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path.cwd(), env=os.environ.copy())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training(Path(sys.argv[1]), epochs=50)
