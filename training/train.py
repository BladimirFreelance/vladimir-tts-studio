from __future__ import annotations

import importlib.util
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dataset.manifest import read_manifest
from training.utils import ensure_espeakbridge_import
from utils import load_yaml, write_json

LOGGER = logging.getLogger(__name__)


def discover_warmstart() -> str | None:
    possible = [
        Path("models") / "ru_RU-dmitri-medium.ckpt",
        Path.home() / ".local" / "share" / "piper" / "ru_RU-dmitri-medium.ckpt",
    ]
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


def run_training(project_dir: Path, epochs: int, base_ckpt: str | None = None, config_path: Path | None = None) -> None:
    ensure_espeakbridge_import()
    cfg = load_yaml(config_path or Path("configs/train_default.yaml"))
    manifest = project_dir / "metadata" / "train.csv"
    rows = read_manifest(manifest)
    missing = [audio for audio, _ in rows if not (project_dir / audio).exists()]
    existing = len(rows) - len(missing)
    if existing <= 0:
        raise RuntimeError(
            "Обнаружено 0 utterances. Проверьте запись в студии и что WAV-файлы существуют в recordings/wav_22050. "
            "Запустите scripts/06_doctor.py --project <name> для диагностики."
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

    cmd = [*resolve_train_base_command(), "fit"]
    cmd += ["--data.config_path", str(data_config)]
    cmd += ["--data.dataset", "jsonl"]
    cmd += ["--max_epochs", str(epochs)]
    cmd += ["--trainer.default_root_dir", str(runs_dir)]
    cmd += ["--data.phoneme_type", cfg["training"]["phoneme_type"]]
    cmd += ["--data.espeak_voice", cfg["training"]["espeak_voice"]]
    cmd += ["--model.batch_size", str(cfg["training"]["batch_size"])]
    cmd += ["--model.learning_rate", str(cfg["training"]["learning_rate"])]

    selected_ckpt = base_ckpt or (discover_warmstart() if cfg["project_defaults"].get("use_warmstart_if_found") else None)
    if selected_ckpt:
        cmd += ["--ckpt_path", selected_ckpt]

    command_path = runs_dir / "run_command.txt"
    command_path.write_text(" ".join(shlex.quote(part) for part in cmd), encoding="utf-8")

    LOGGER.info("Launching training: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path.cwd(), env=os.environ.copy())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training(Path(sys.argv[1]), epochs=50)
