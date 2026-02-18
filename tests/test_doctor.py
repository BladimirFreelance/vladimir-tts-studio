from __future__ import annotations

import types
from pathlib import Path

from app.doctor import check_imports, check_manifest, run_doctor


def test_check_imports_reports_missing_training_modules(monkeypatch):
    def fake_import(name: str):
        if name == "piper.espeakbridge":
            return object()
        if name == "torch":
            return types.SimpleNamespace(__version__="0", cuda=types.SimpleNamespace(is_available=lambda: False))
        raise ImportError(name)

    monkeypatch.setattr("app.doctor.importlib.import_module", fake_import)
    monkeypatch.setattr("app.doctor.importlib.util.find_spec", lambda _name: None)
    monkeypatch.setattr("app.doctor.shutil.which", lambda _name: "/usr/bin/espeak-ng")

    issues = check_imports()

    assert any("Piper training modules not found" in issue for issue in issues)


def test_check_manifest_handles_missing_manifest(tmp_path: Path, caplog):
    stats = check_manifest(tmp_path / "<name>")

    assert stats == {"rows": 0, "ok": 0, "missing": 0, "fixed": 0, "error": "manifest_missing"}
    assert "Project name looks like a placeholder" in caplog.text
    assert "Manifest not found" in caplog.text


def test_run_doctor_returns_error_code_when_manifest_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.doctor.check_imports", lambda: [])

    code = run_doctor(tmp_path / "demo", require_audio=True)

    assert code == 2


def test_run_doctor_logs_prepare_hint_when_manifest_missing(monkeypatch, tmp_path: Path, caplog):
    monkeypatch.setattr("app.doctor.check_imports", lambda: [])

    _code = run_doctor(tmp_path / "demo", require_audio=True)

    assert "No manifest found. Run prepare first" in caplog.text
    assert "0 utterances ready for training" not in caplog.text
