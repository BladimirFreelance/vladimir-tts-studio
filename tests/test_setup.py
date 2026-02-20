from __future__ import annotations

import importlib.util
import types
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "00_setup_env.py"


def _load_setup_module(tmp_path: Path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    temp_script = scripts_dir / "00_setup_env.py"
    temp_script.write_text(MODULE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("setup_env_test_module", temp_script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_require_piper_training_fails_when_modules_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_setup_module(tmp_path)
    previous_cwd = Path.cwd()

    monkeypatch.setattr(module, "ensure_venv", lambda _venv_dir: [sys.executable, "-m", "pip"])
    monkeypatch.setattr(module, "resolve_python", lambda _venv_dir: [sys.executable])
    monkeypatch.setattr(module, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "print_torch_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "install_torch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "install_piper_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "install_piper_training", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(module, "verify_piper_training_install", lambda *_args, **_kwargs: False)

    monkeypatch.setattr(sys, "argv", ["00_setup_env.py", "--require-piper-training"])

    try:
        result = module.main()
    finally:
        monkeypatch.chdir(previous_cwd)

    assert result == 1


def test_setup_training_mode_skips_runtime_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_setup_module(tmp_path)
    previous_cwd = Path.cwd()
    calls: list[str] = []

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: types.SimpleNamespace(
            venv=".venv",
            torch="skip",
            extras=[],
            without_piper_training=False,
            with_piper_training=False,
            piper_training_source="third_party/piper1-gpl",
            require_piper_training=False,
        ),
    )
    monkeypatch.setattr(module, "ensure_venv", lambda _venv_dir: [sys.executable, "-m", "pip"])
    monkeypatch.setattr(module, "resolve_python", lambda _venv_dir: [sys.executable])
    monkeypatch.setattr(module, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "print_torch_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "install_torch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "warn_if_espeakbridge_missing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "print_system_hints", lambda: None)
    monkeypatch.setattr(module, "verify_piper_training_install", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        module,
        "install_piper_runtime",
        lambda *_args, **_kwargs: calls.append("runtime"),
    )
    monkeypatch.setattr(
        module,
        "install_piper_training",
        lambda *_args, **_kwargs: calls.append("training") or True,
    )

    try:
        result = module.main()
    finally:
        monkeypatch.chdir(previous_cwd)

    assert result == 0
    assert "training" in calls
    assert "runtime" not in calls
