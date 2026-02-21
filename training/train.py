from __future__ import annotations

import logging
import os
import argparse
import importlib
import shlex
import subprocess
import sys
import shutil
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for pre-flight diagnostics
    torch = None

from dataset.manifest import read_manifest
from training.checkpoints import ensure_stable_ckpt_aliases
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

    return [sys.executable, "-m", "training.piper_train_bootstrap"]


def _resolve_venv_path() -> str:
    if os.getenv("VIRTUAL_ENV"):
        return os.environ["VIRTUAL_ENV"]

    if sys.prefix != sys.base_prefix:
        return sys.prefix

    return "<system>"


def _detect_phonemizer_backend(phoneme_type: str) -> str:
    if phoneme_type != "espeak":
        return f"{phoneme_type} (configured)"

    try:
        importlib.import_module("piper.espeakbridge")
        return "piper.espeakbridge"
    except Exception:
        return "missing (runtime piper-tts is not available)"



def _resolve_resume_flag() -> str:
    train_main = (
        Path(__file__).resolve().parent.parent
        / "third_party"
        / "piper1-gpl"
        / "src"
        / "piper"
        / "train"
        / "__main__.py"
    )
    if train_main.exists():
        source = train_main.read_text(encoding="utf-8")
        if "--ckpt_path" in source or "ckpt_path" in source:
            return "--ckpt_path"
        if "resume_from_checkpoint" in source:
            return "--trainer.resume_from_checkpoint"
    return "--ckpt_path"


def _assert_training_runtime_preflight(phoneme_type: str) -> None:
    train_cmd_override = os.getenv("PIPER_TRAIN_CMD")
    runtime_and_training_ok = bool(train_cmd_override)

    if train_cmd_override:
        LOGGER.info(
            "[train] preflight skipped piper bootstrap check: custom trainer command configured via PIPER_TRAIN_CMD"
        )
    else:
        try:
            from training.piper_train_bootstrap import validate_runtime_and_training_imports

            validate_runtime_and_training_imports()
            runtime_and_training_ok = True
        except SystemExit:
            runtime_and_training_ok = False
        except Exception:
            runtime_and_training_ok = False

    phonemizer_backend = _detect_phonemizer_backend(phoneme_type)

    LOGGER.info("[train] preflight python: %s", Path(sys.executable).resolve())
    LOGGER.info("[train] preflight phonemizer backend: %s", phonemizer_backend)
    LOGGER.info("[train] preflight training module: %s", runtime_and_training_ok)

    if not runtime_and_training_ok:
        raise RuntimeError(
            "Training preflight failed before subprocess launch:\n"
            "- Не удалось проверить piper.espeakbridge и piper.train.vits через training bootstrap:\n"
            "Как починить:\n"
            "  * python -m pip install piper-tts==1.4.1\n"
            "  * pip install -r requirements/train.txt"
        )


def _log_training_runtime_info(
    *,
    project_dir: Path,
    audio_dir: Path,
    train_cmd: list[str],
    cfg: dict,
    dry_run: bool,
) -> None:
    piper_backend = (
        "custom command via PIPER_TRAIN_CMD"
        if os.getenv("PIPER_TRAIN_CMD")
        else "training.piper_train_bootstrap"
    )
    phoneme_type = str(cfg["training"]["phoneme_type"])
    phonemizer_backend = _detect_phonemizer_backend(phoneme_type)
    LOGGER.info("[train] mode: %s", "check/dry-run" if dry_run else "fit")
    LOGGER.info("[train] project dir: %s", project_dir)
    LOGGER.info("[train] audio dir: %s", audio_dir)
    LOGGER.info("[train] python executable: %s", Path(sys.executable).resolve())
    LOGGER.info("[train] python venv: %s", _resolve_venv_path())
    LOGGER.info("[train] piper backend: %s (%s)", piper_backend, " ".join(train_cmd))
    LOGGER.info(
        "[train] phonemizer: %s (phoneme_type=%s, espeak_voice=%s)",
        phonemizer_backend,
        phoneme_type,
        cfg["training"]["espeak_voice"],
    )


def detect_supported_gpu_or_raise(
    *, force_cpu: bool = False, preferred_gpu_name: str | None = None
) -> dict[str, str]:
    if force_cpu:
        LOGGER.info("Выбран принудительный запуск на CPU (--force_cpu)")
        return {"CUDA_VISIBLE_DEVICES": ""}

    if os.getenv("CUDA_VISIBLE_DEVICES"):
        LOGGER.info("CUDA_VISIBLE_DEVICES уже задан, использую его без изменений")
        return {}

    if torch is None or not torch.cuda.is_available():
        LOGGER.warning("CUDA недоступна, запускаю обучение на CPU")
        return {}

    preferred = (preferred_gpu_name or "").strip().lower()
    preferred_candidates: list[int] = []
    fallback_candidates: list[int] = []

    for index in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(index)
        if preferred and preferred in device_name.lower():
            preferred_candidates.append(index)
        else:
            fallback_candidates.append(index)

    if preferred and not preferred_candidates:
        LOGGER.warning(
            "GPU с именем '%s' не найден. Продолжаю обычный автоподбор.",
            preferred_gpu_name,
        )

    candidate_indexes = preferred_candidates + fallback_candidates

    for index in candidate_indexes:
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




def _detect_sample_rate(audio_dir: Path, rows: list[tuple[str, str]], fallback: int) -> int:
    for audio, _text in rows:
        audio_name = Path(audio).name.strip()
        if not audio_name:
            continue

        wav_path = audio_dir / audio_name
        if not wav_path.exists():
            continue

        try:
            from dataset.audio_tools import inspect_wav

            return inspect_wav(wav_path).sample_rate
        except Exception:
            continue

    return fallback

def _assert_manifest_audio(audio_dir: Path, rows: list[tuple[str, str]]) -> int:
    missing: list[str] = []
    for audio, _text in rows:
        audio_name = Path(audio).name.strip()
        if not audio_name:
            missing.append(audio)
            continue

        wav_path = audio_dir / audio_name
        if not wav_path.exists():
            missing.append(audio_name)

    if missing:
        preview = ", ".join(missing[:3])
        raise RuntimeError(
            "Manifest указывает на отсутствующие WAV. "
            f"Отсутствует: {len(missing)} (примеры: {preview})."
        )

    return len(rows)


def run_training(
    project_dir: Path,
    epochs: int,
    dry_run: bool = False,
    vocoder_warmstart_ckpt: str | None = None,
    config_path: Path | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    audio_dir: Path | None = None,
    *,
    base_ckpt: str | None = None,
    resume_ckpt: str | Path | None = None,
    output_ckpt_path: str | Path | None = None,
    force_cpu: bool = False,
    preferred_gpu_name: str | None = None,
) -> None:
    resolved_audio_dir = audio_dir or project_dir / "recordings" / "wav_22050"
    from app.doctor import assert_training_preflight

    assert_training_preflight(project_dir, audio_dir=resolved_audio_dir)

    train_env_patch = detect_supported_gpu_or_raise(
        force_cpu=force_cpu,
        preferred_gpu_name=preferred_gpu_name,
    )

    cfg = load_yaml(config_path or Path("configs/train_default.yaml"))
    _assert_training_runtime_preflight(str(cfg["training"]["phoneme_type"]))
    manifest = project_dir / "metadata" / "train.csv"
    rows = read_manifest(manifest)

    try:
        existing = _assert_manifest_audio(resolved_audio_dir, rows)
    except RuntimeError:
        from app.doctor import check_manifest

        stats = check_manifest(project_dir, auto_fix=True, audio_dir=resolved_audio_dir)
        if stats.get("path_fixed", 0) > 0:
            LOGGER.info(
                "doctor auto-fix обновил manifest paths: %s. Повторяю проверку.",
                stats["path_fixed"],
            )
            rows = read_manifest(manifest)
        existing = _assert_manifest_audio(resolved_audio_dir, rows)

    if existing <= 0:
        raise RuntimeError(
            "Обнаружено 0 utterances. Проверьте запись в студии и что WAV-файлы существуют в recordings/wav_22050. "
            f"Запустите python -m app.main doctor --project {project_dir.name} --auto-fix для диагностики."
        )

    detected_sample_rate = _detect_sample_rate(
        resolved_audio_dir, rows, int(cfg["project_defaults"]["sample_rate"])
    )

    data_config = project_dir / "metadata" / "data_config.json"
    write_json(
        data_config,
        {
            "dataset": "local",
            "manifest": str(manifest),
            "audio_dir": str(resolved_audio_dir),
            "sample_rate": detected_sample_rate,
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
    cmd += ["--data.audio_dir", str(resolved_audio_dir)]
    cmd += ["--data.cache_dir", str(project_dir / "cache")]
    cmd += ["--trainer.max_epochs", str(epochs)]
    cmd += ["--trainer.default_root_dir", str(runs_dir)]
    cmd += ["--data.phoneme_type", cfg["training"]["phoneme_type"]]
    cmd += ["--data.espeak_voice", cfg["training"]["espeak_voice"]]
    resolved_batch_size = batch_size or int(cfg["training"]["batch_size"])
    cmd += ["--data.batch_size", str(resolved_batch_size)]
    cmd += ["--data.num_workers", "0"]
    resolved_learning_rate = learning_rate or float(cfg["training"]["learning_rate"])
    cmd += ["--model.learning_rate", str(resolved_learning_rate)]

    selected_ckpt = vocoder_warmstart_ckpt or base_ckpt or (
        discover_warmstart()
        if cfg["project_defaults"].get("use_warmstart_if_found")
        else None
    )
    if selected_ckpt:
        cmd += ["--model.vocoder_warmstart_ckpt", selected_ckpt]

    if resume_ckpt:
        cmd += [_resolve_resume_flag(), str(resume_ckpt)]

    command_path = runs_dir / "run_command.txt"
    command_path.write_text(
        " ".join(shlex.quote(part) for part in cmd), encoding="utf-8"
    )

    _log_training_runtime_info(
        project_dir=project_dir,
        audio_dir=resolved_audio_dir,
        train_cmd=cmd,
        cfg=cfg,
        dry_run=dry_run,
    )

    if dry_run:
        LOGGER.info(
            "[train] Dry-run completed: окружение, пути и данные валидны. Эпохи не запускались."
        )
        return

    LOGGER.info("Launching training: %s", " ".join(cmd))
    train_env = os.environ.copy()
    train_env.update(train_env_patch)
    subprocess.run(cmd, check=True, cwd=Path.cwd(), env=train_env)
    ensure_stable_ckpt_aliases(runs_dir)

    if output_ckpt_path:
        output_ckpt = Path(output_ckpt_path).expanduser()
        latest_ckpt = max(
            (path for path in runs_dir.glob("**/*.ckpt") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            default=None,
        )
        if latest_ckpt is not None:
            output_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_ckpt, output_ckpt)
            LOGGER.info("Saved latest checkpoint to %s", output_ckpt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Запуск локального обучения Piper")
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--gpu-name", dest="preferred_gpu_name")
    args = parser.parse_args()

    run_training(
        args.project_dir,
        epochs=args.epochs,
        force_cpu=args.force_cpu,
        preferred_gpu_name=args.preferred_gpu_name,
    )
