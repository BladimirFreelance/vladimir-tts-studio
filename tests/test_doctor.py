from __future__ import annotations

import types

from app.doctor import check_imports


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
