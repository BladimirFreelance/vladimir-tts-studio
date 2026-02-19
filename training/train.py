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

from dataset.manifest import read_manifest
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

    try:
        if importlib.util.find_spec("training.piper_train_bootstrap") is not None:
            return [sys.executable, "-m", "training.piper_train_bootstrap"]
    except ModuleNotFoundError:
        pass

    raise RuntimeError(
        "Не найден bootstrap для обучения Piper (training.piper_train_bootstrap). "
        "Запустите scripts/00_setup_env.py --require-piper-training, "
        "или задайте PowerShell-переменную: $env:PIPER_TRAIN_CMD=\"python -m training.piper_train_bootstrap\"."
    )


def detect_supported_gpu_or_raise() -> dict[str, str]:
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        LOGGER.info("CUDA_VISIBLE_DEVICES уже задан, использую его без изменений")
        return {}

    if torch is None or not torch.cuda.is_available():
        LOGGER.warning("CUDA недоступна, запускаю обучение на CPU")
        return {}

    for index in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        try:
            torch.zeros(1, device=f"cuda:{index}")
            torch.cuda.synchronize(index)
            if index == 0:
                LOGGER.info(
                    "Выбран GPU #%s: %s (sm_%s%s)",
                    index,
                    device_name,
                    capability[0],
                    capability[1],
                )
                return {}

            LOGGER.info(
                "GPU #0 пропущен, выбран совместимый GPU #%s: %s (sm_%s%s). "
                "Для процесса обучения установлен CUDA_VISIBLE_DEVICES=%s",
                index,
                device_name,
                capability[0],
                capability[1],
                index,
            )
            return {"CUDA_VISIBLE_DEVICES": str(index)}
        except RuntimeError as exc:
            LOGGER.warning(
                "Пропускаю GPU #%s (%s, sm_%s%s): %s",
                index,
                device_name,
                capability[0],
                capability[1],
                exc,
            )

    LOGGER.warning("Не найден совместимый CUDA GPU. Продолжаю обучение на CPU")
    return {"CUDA_VISIBLE_DEVICES": ""}


def _assert_manifest_audio(project_dir: Path, rows: list[tuple[str, str]]) -> int:
    audio_dir = project_dir / "recordings" / "wav_22050"
    missing: list[str] = []
    for audio, _text in rows:
        filename = Path(audio).name
        if not filename:
            missing.append(audio)
            continue
        if not (audio_dir / filename).exists():
            missing.append(filename)

    if missing:
        preview = ", ".join(missing[:3])
        raise RuntimeError(
            "Manifest указывает на отсутствующие WAV в recordings/wav_22050. "
            f"Отсутствует: {len(missing)} (примеры: {preview})."
        )

    return len(rows)


def run_training(
    project_dir: Path,
    epochs: int,
    vocoder_warmstart_ckpt: str | None = None,
    config_path: Path | None = None,
    batch_size: int | None = None,
    *,
    base_ckpt: str | None = None,
) -> None:
    train_env_patch = detect_supported_gpu_or_raise()
    ensure_espeakbridge_import()
    cfg = load_yaml(config_path or Path("configs/train_default.yaml"))
    manifest = project_dir / "metadata" / "train.csv"
    rows = read_manifest(manifest)
    existing = _assert_manifest_audio(project_dir, rows)
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
    cmd += ["--data.dataset_type", "text"]
    cmd += ["--data.voice_name", project_dir.name]
    cmd += ["--data.csv_path", str(manifest)]
    cmd += ["--data.audio_dir", str(project_dir / "recordings" / "wav_22050")]
    cmd += ["--data.cache_dir", str(project_dir / "cache")]
    cmd += ["--trainer.max_epochs", str(epochs)]
    cmd += ["--trainer.default_root_dir", str(runs_dir)]
    cmd += ["--data.phoneme_type", cfg["training"]["phoneme_type"]]
    cmd += ["--data.espeak_voice", cfg["training"]["espeak_voice"]]
    resolved_batch_size = batch_size or int(cfg["training"]["batch_size"])
    cmd += ["--data.batch_size", str(resolved_batch_size)]
    cmd += ["--model.learning_rate", str(cfg["training"]["learning_rate"])]

    selected_ckpt = vocoder_warmstart_ckpt or base_ckpt or (
        discover_warmstart()
        if cfg["project_defaults"].get("use_warmstart_if_found")
        else None
    )
    if selected_ckpt:
        cmd += ["--model.vocoder_warmstart_ckpt", selected_ckpt]

    command_path = runs_dir / "run_command.txt"
    command_path.write_text(
        " ".join(shlex.quote(part) for part in cmd), encoding="utf-8"
    )

    LOGGER.info("Launching training: %s", " ".join(cmd))
    train_env = os.environ.copy()
    train_env.update(train_env_patch)
    subprocess.run(cmd, check=True, cwd=Path.cwd(), env=train_env)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training(Path(sys.argv[1]), epochs=50)
