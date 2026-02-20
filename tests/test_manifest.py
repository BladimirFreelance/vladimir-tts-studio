from pathlib import Path

import pytest

from dataset.manifest import read_manifest, resolve_audio_path, write_manifest


def test_write_and_read_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata" / "train.csv"
    rows = [
        ("recordings/a.wav", " Первая строка\nс переносом "),
        ("recordings/b.wav", "Вторая строка"),
    ]

    write_manifest(manifest, rows)

    assert manifest.exists()
    assert read_manifest(manifest) == [
        ("recordings/a.wav", "Первая строка с переносом"),
        ("recordings/b.wav", "Вторая строка"),
    ]


def test_read_manifest_rejects_missing_delimiter(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text("recordings/a.wav без разделителя\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing '\\|' delimiter"):
        read_manifest(manifest)


def test_read_manifest_rejects_header_row(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text("audio|text\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain CSV header"):
        read_manifest(manifest)


def test_resolve_audio_path_supports_custom_audio_dir(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    custom_audio_dir = project_dir / "recordings" / "wav_16000"

    resolved = resolve_audio_path(project_dir, "001.wav", audio_dir=custom_audio_dir)

    assert resolved == custom_audio_dir / "001.wav"
