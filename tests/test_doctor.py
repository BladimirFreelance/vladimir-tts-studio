from __future__ import annotations

import types
import wave
from pathlib import Path

import pytest

from app.doctor import (
    assert_training_preflight,
    build_training_preflight_report,
    check_imports,
    check_manifest,
    run_doctor,
)


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 22050,
    channels: int = 1,
    sample_width: int = 2,
    frames: int = 22050,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frames * channels * sample_width)


def test_check_imports_reports_missing_training_modules(monkeypatch):
    def fake_import(name: str):
        if name == "piper.espeakbridge":
            return object()
        if name == "torch":
            return types.SimpleNamespace(
                __version__="0", cuda=types.SimpleNamespace(is_available=lambda: False)
            )
        raise ImportError(name)

    monkeypatch.setattr("app.doctor.importlib.import_module", fake_import)
    monkeypatch.setattr("app.doctor.importlib.util.find_spec", lambda _name: None)
    monkeypatch.setattr("app.doctor.shutil.which", lambda _name: "/usr/bin/espeak-ng")

    issues = check_imports()

    assert any("Piper training module not found" in issue for issue in issues)


def test_check_manifest_handles_missing_manifest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    stats = check_manifest(tmp_path / "<name>")

    assert stats == {
        "rows": 0,
        "ok": 0,
        "missing": 0,
        "fixed": 0,
        "invalid_paths": 0,
        "sample_rate_mismatch": 0,
        "error": "manifest_missing",
    }
    assert "Project name looks like a placeholder" in caplog.text
    assert "Manifest not found" in caplog.text


def test_check_manifest_reports_missing_files_invalid_paths_and_sample_rate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    project_dir = tmp_path / "demo"
    manifest_path = project_dir / "metadata" / "train.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "recordings/wav_22050/good|ok\n"
        "recordings/wav_22050/missing|missing\n"
        "../outside|bad path\n"
        "recordings/wav_22050/wrong_sr|wrong sr\n",
        encoding="utf-8",
    )

    _write_wav(project_dir / "recordings" / "wav_22050" / "good.wav", sample_rate=22050)
    _write_wav(
        project_dir / "recordings" / "wav_22050" / "wrong_sr.wav", sample_rate=16000
    )

    stats = check_manifest(project_dir)

    assert stats["rows"] == 4
    assert stats["ok"] == 2
    assert stats["missing"] == 1
    assert stats["invalid_paths"] == 1
    assert stats["sample_rate_mismatch"] == 1
    assert "Missing file referenced in manifest" in caplog.text
    assert "Invalid manifest audio path" in caplog.text
    assert "Sample rate mismatch" in caplog.text


def test_run_doctor_returns_error_code_when_manifest_missing(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr("app.doctor.check_imports", lambda: [])

    code = run_doctor(tmp_path / "demo", require_audio=True)

    assert code == 2


def test_run_doctor_logs_prepare_hint_when_manifest_missing(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr("app.doctor.check_imports", lambda: [])

    _code = run_doctor(tmp_path / "demo", require_audio=True)

    assert "No manifest found. Run prepare first" in caplog.text
    assert "0 utterances ready for training" not in caplog.text


def test_run_doctor_auto_fix_training_invokes_setup(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "app.doctor.check_imports",
        lambda: ["Piper training module not found"]
    )
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], check: bool = False):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("app.doctor.subprocess.run", fake_run)
    monkeypatch.setattr(
        "app.doctor.check_manifest",
        lambda _project_dir, auto_fix=False, audio_dir=None: {
            "rows": 0,
            "ok": 1,
            "missing": 0,
            "fixed": 0,
            "path_fixed": 0,
            "invalid_paths": 0,
            "sample_rate_mismatch": 0,
            "duration_min": 0.0,
            "error": "",
        },
    )

    run_doctor(tmp_path / "demo", auto_fix=True, require_audio=False)

    assert captured["cmd"][1:] == [
        "scripts/00_setup_env.py",
        "--require-piper-training",
    ]


def test_check_manifest_uses_custom_audio_dir(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    manifest_path = project_dir / "metadata" / "train.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("001.wav|ok\n", encoding="utf-8")

    audio_dir = project_dir / "recordings" / "wav_16000"
    _write_wav(audio_dir / "001.wav", sample_rate=16000)

    stats = check_manifest(project_dir, audio_dir=audio_dir)

    assert stats["ok"] == 1
    assert stats["missing"] == 0


def test_check_manifest_auto_fix_rewrites_paths_to_filenames(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo"
    manifest_path = project_dir / "metadata" / "train.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "recordings/wav_22050/001.wav|ok\n"
        "../recordings/wav_22050/002.wav|ok2\n",
        encoding="utf-8",
    )

    _write_wav(project_dir / "recordings" / "wav_22050" / "001.wav")
    _write_wav(project_dir / "recordings" / "wav_22050" / "002.wav")

    stats = check_manifest(project_dir, auto_fix=True)

    assert stats["path_fixed"] == 2
    assert manifest_path.read_text(encoding="utf-8") == "001.wav|ok\n002.wav|ok2\n"


def test_run_doctor_auto_fix_creates_audio_and_cache_dirs(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("app.doctor.check_imports", lambda: [])
    monkeypatch.setattr(
        "app.doctor.check_manifest",
        lambda _project_dir, auto_fix=False, audio_dir=None: {
            "rows": 0,
            "ok": 1,
            "missing": 0,
            "fixed": 0,
            "path_fixed": 0,
            "invalid_paths": 0,
            "sample_rate_mismatch": 0,
            "duration_min": 0.0,
            "error": "",
        },
    )

    project_dir = tmp_path / "demo"
    code = run_doctor(project_dir, auto_fix=True, require_audio=False)

    assert code == 0
    assert (project_dir / "recordings" / "wav_22050").is_dir()
    assert (project_dir / "cache").is_dir()


def test_build_training_preflight_report_returns_fixes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "app.doctor.check_imports",
        lambda: [
            "Piper training module not found",
            "PyTorch not importable (training unavailable)",
        ],
    )
    monkeypatch.setattr(
        "app.doctor.check_manifest",
        lambda *_args, **_kwargs: {
            "ok": 0,
            "missing": 1,
            "error": "manifest_missing",
        },
    )

    blocking, remediation = build_training_preflight_report(tmp_path / "demo")

    assert any("Piper training" in item for item in blocking)
    assert any("PyTorch" in item for item in blocking)
    assert any("metadata/train.csv" in item for item in blocking)
    assert any("00_setup_env.py --require-piper-training" in item for item in remediation)


def test_assert_training_preflight_raises_with_hints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "app.doctor.build_training_preflight_report",
        lambda *_args, **_kwargs: (
            ["Нет готовых записей"],
            ["python scripts/06_doctor.py --project demo --auto-fix"],
        ),
    )

    with pytest.raises(RuntimeError, match="Training preflight failed"):
        assert_training_preflight(tmp_path / "demo")
