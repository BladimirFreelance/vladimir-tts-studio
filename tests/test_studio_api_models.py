from __future__ import annotations

import os
import time
from pathlib import Path

from studio.api import (
    build_default_base_ckpt_path,
    discover_checkpoint_models,
    discover_onnx_models,
    discover_prompts,
    normalize_user_path,
)


def _touch(path: Path, *, offset_seconds: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub", encoding="utf-8")
    ts = time.time() + offset_seconds
    os.utime(path, (ts, ts))


def test_discover_checkpoint_models_returns_latest_first(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    _touch(project_dir / "runs" / "epoch=1.ckpt", offset_seconds=-10)
    _touch(project_dir / "runs" / "best.ckpt", offset_seconds=0)

    found = discover_checkpoint_models(project_dir)

    assert [path.name for path in found] == ["best.ckpt", "epoch=1.ckpt"]


def test_discover_onnx_models_prefers_current_project(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "demo"
    _touch(tmp_path / "voices_out" / "other-voice.onnx", offset_seconds=-10)
    _touch(tmp_path / "voices_out" / "demo-fast.onnx", offset_seconds=0)

    found = discover_onnx_models(project_dir)

    assert [path.name for path in found] == ["demo-fast.onnx"]


def test_build_default_base_ckpt_path_uses_standard_location(tmp_path: Path) -> None:
    project_dir = tmp_path / "my_project"

    result = build_default_base_ckpt_path(project_dir)

    assert result.parent == Path("data") / "models" / "my_project"
    assert result.name.startswith("my_project-")
    assert result.suffix == ".ckpt"


def test_discover_prompts_uses_manifest_when_segments_missing(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts" / "segments.txt"
    manifest_path = tmp_path / "metadata" / "train.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "recordings/wav_22050/demo_0001.wav|Пример\n"
        "recordings/wav_22050/demo_0002|Вторая строка\n",
        encoding="utf-8",
    )

    prompts = discover_prompts(prompts_file, manifest_path)

    assert prompts == ["demo_0001|Пример", "demo_0002|Вторая строка"]


def test_discover_prompts_prefers_existing_segments_file(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts" / "segments.txt"
    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    prompts_file.write_text("seg_0001|Из prompts\n", encoding="utf-8")

    manifest_path = tmp_path / "metadata" / "train.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("recordings/x.wav|Из manifest\n", encoding="utf-8")

    prompts = discover_prompts(prompts_file, manifest_path)

    assert prompts == ["seg_0001|Из prompts"]


def test_normalize_user_path_handles_windows_slashes() -> None:
    path = normalize_user_path(r"data\models\demo\demo.ckpt")

    assert path == Path("data/models/demo/demo.ckpt")
