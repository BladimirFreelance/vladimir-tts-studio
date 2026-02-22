from pathlib import Path

import numpy as np
from scipy.io import wavfile

from app.workflows import prepare_dataset
from dataset.audio_tools import inspect_wav


def _write_test_config(root: Path) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "train_default.yaml").write_text(
        """
text:
  replacements: {}
  expand_common_abbreviations: true
  split_regex: "[.!?]+"
  max_chars: 300
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_prepare_dataset_creates_manifest_with_leaf_wav_filename(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _write_test_config(tmp_path)

    source_text = tmp_path / "input.txt"
    source_text.write_text("Первая реплика. Вторая реплика.", encoding="utf-8")

    project = "demo"
    audio_id = f"{project}_00001"
    recordings_dir = tmp_path / "data" / "projects" / project / "recordings" / "wav_22050"
    recordings_dir.mkdir(parents=True, exist_ok=True)

    stereo = np.zeros((44100, 2), dtype=np.int16)
    stereo[:, 0] = 500
    stereo[:, 1] = -500
    wavfile.write(recordings_dir / f"{audio_id}.wav", 44100, stereo)

    prepare_dataset(source_text, project)

    manifest = tmp_path / "data" / "projects" / project / "metadata" / "train.csv"
    rows = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith(f"{audio_id}.wav|")

    converted = recordings_dir / f"{audio_id}.wav"
    info = inspect_wav(converted)
    assert info.sample_rate == 22050
    assert info.channels == 1
    assert info.bits_per_sample == 16


def test_prepare_dataset_raises_clear_error_for_missing_text(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _write_test_config(tmp_path)

    missing_text = tmp_path / "missing.txt"

    try:
        prepare_dataset(missing_text, "demo")
    except FileNotFoundError as exc:
        assert "Текстовый файл не найден" in str(exc)
        assert "--text" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing text file")


def test_prepare_dataset_csv_manifest_uses_leaf_filename(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _write_test_config(tmp_path)

    source_text = tmp_path / "input.csv"
    source_text.write_text("audio,text\nfolder/line_001.wav,Первая реплика\n", encoding="utf-8")

    project = "demo_csv"
    prepare_dataset(source_text, project)

    manifest = tmp_path / "data" / "projects" / project / "metadata" / "train.csv"
    first_audio = manifest.read_text(encoding="utf-8").strip().split("|", maxsplit=1)[0]
    assert "/" not in first_audio
    assert "\\" not in first_audio


def test_prepare_dataset_tsv_manifest_uses_leaf_filename(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _write_test_config(tmp_path)

    source_text = tmp_path / "input.tsv"
    source_text.write_text("audio\ttext\nfolder\\line_001.wav\tПервая реплика\n", encoding="utf-8")

    project = "demo_tsv"
    prepare_dataset(source_text, project)

    manifest = tmp_path / "data" / "projects" / project / "metadata" / "train.csv"
    first_audio = manifest.read_text(encoding="utf-8").strip().split("|", maxsplit=1)[0]
    assert "/" not in first_audio
    assert "\\" not in first_audio


def test_prepare_append_does_not_overwrite(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _write_test_config(tmp_path)

    project = "ddn_vladimir"
    project_dir = tmp_path / "data" / "projects" / project
    manifest = project_dir / "metadata" / "train.csv"
    recordings_dir = project_dir / "recordings" / "wav_22050"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    recordings_dir.mkdir(parents=True, exist_ok=True)

    old_wav = recordings_dir / "ddn_vladimir_00001.wav"
    old_bytes = b"RIFF_old_audio_payload"
    old_wav.write_bytes(old_bytes)
    manifest.write_text("ddn_vladimir_00001.wav|Старая фраза\n", encoding="utf-8")

    source_text = tmp_path / "append.txt"
    source_text.write_text("Новая фраза", encoding="utf-8")

    prepare_dataset(source_text, project, mode="append")

    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "ddn_vladimir_00001.wav|Старая фраза"
    assert rows[1].startswith("ddn_vladimir_00002.wav|")
    assert old_wav.read_bytes() == old_bytes


def test_prepare_overwrite_requires_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _write_test_config(tmp_path)

    source_text = tmp_path / "input.txt"
    source_text.write_text("Текст", encoding="utf-8")

    try:
        prepare_dataset(source_text, "demo", mode="overwrite", overwrite_confirmed=False)
    except ValueError as exc:
        assert "requires explicit confirmation" in str(exc)
    else:
        raise AssertionError("Expected ValueError for overwrite without confirmation")


def test_prepare_append_keeps_nonstandard_wavs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _write_test_config(tmp_path)

    project = "ddn_vladimir"
    project_dir = tmp_path / "data" / "projects" / project
    manifest = project_dir / "metadata" / "train.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("9000000001.wav|legacy\n", encoding="utf-8")

    source_text = tmp_path / "append_nonstandard.txt"
    source_text.write_text("Новая фраза", encoding="utf-8")

    prepare_dataset(source_text, project, mode="append")

    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "9000000001.wav|legacy"
    assert rows[1].startswith("ddn_vladimir_00001.wav|")
