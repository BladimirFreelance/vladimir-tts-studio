from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_PIPER_KEYS = (
    "num_symbols",
    "num_speakers",
    "phoneme_id_map",
)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_template_config(project_dir: Path) -> Path | None:
    repo_root = Path.cwd()
    candidates = [
        repo_root / "models" / "ru_RU-dmitri-medium.onnx.json",
        repo_root / "ru_RU-dmitri-medium.onnx.json",
    ]
    found = _first_existing(candidates)
    if found is not None:
        return found

    model_templates = sorted((repo_root / "models").glob("*.onnx.json"))
    if model_templates:
        return model_templates[0]

    project_templates = sorted(project_dir.glob("*.onnx.json"))
    if project_templates:
        return project_templates[0]
    return None


def load_template_config(project_dir: Path) -> dict[str, Any]:
    template_path = find_template_config(project_dir)
    if template_path is None:
        raise RuntimeError(
            "Не найден шаблон ONNX JSON: положите базовый *.onnx.json в models/"
        )

    return json.loads(template_path.read_text(encoding="utf-8"))


def build_onnx_config(project_dir: Path, project_name: str) -> dict[str, Any]:
    template = load_template_config(project_dir)

    config: dict[str, Any] = dict(template)
    config.setdefault("speaker_id_map", {})
    config.setdefault("phoneme_map", {})
    config.setdefault("phoneme_type", "espeak")

    audio = dict(config.get("audio") or {})
    audio["sample_rate"] = 22050
    config["audio"] = audio

    language = dict(config.get("language") or {})
    language["code"] = "ru_RU"
    config["language"] = language

    espeak = dict(config.get("espeak") or {})
    espeak["voice"] = "ru"
    config["espeak"] = espeak

    config["dataset"] = project_name

    inference = dict(config.get("inference") or {})
    inference.update({"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8})
    config["inference"] = inference

    missing = [key for key in REQUIRED_PIPER_KEYS if key not in config]
    if missing:
        raise RuntimeError(
            "Шаблон ONNX JSON неполный. Отсутствуют ключи: " + ", ".join(missing)
        )

    return config
