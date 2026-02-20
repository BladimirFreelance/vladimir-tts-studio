from __future__ import annotations

import json
import os
import struct
import wave
from pathlib import Path

from training.train import run_training


def _write_tone(path: Path, sample_rate: int = 22050, duration_seconds: float = 0.1) -> None:
    frame_count = int(sample_rate * duration_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(struct.pack("<h", 0) for _ in range(frame_count)))


def test_training_creates_checkpoint_with_minimal_dataset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "train_default.yaml"
    project_dir = tmp_path / "demo_project"
    metadata_dir = project_dir / "metadata"
    audio_dir = project_dir / "recordings" / "wav_22050"
    metadata_dir.mkdir(parents=True)

    _write_tone(audio_dir / "001.wav")
    _write_tone(audio_dir / "002.wav")
    (metadata_dir / "train.csv").write_text("001.wav|test one\n002.wav|test two\n", encoding="utf-8")

    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text(
        """
import json
import pathlib
import sys

args = sys.argv[1:]
root_dir = pathlib.Path(args[args.index('--trainer.default_root_dir') + 1])
root_dir.mkdir(parents=True, exist_ok=True)
checkpoint = root_dir / 'epoch=0-step=1.ckpt'
checkpoint.write_text('ok', encoding='utf-8')
(root_dir / 'args.json').write_text(json.dumps(args), encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )

    env_key = "PIPER_TRAIN_CMD"
    previous = os.environ.get(env_key)
    os.environ[env_key] = f"{os.fspath(Path(os.sys.executable))} {fake_trainer}"
    try:
        run_training(project_dir, epochs=1, force_cpu=True, config_path=config_path)
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous

    checkpoints = list((project_dir / "runs").glob("*.ckpt"))
    assert checkpoints

    args_payload = json.loads((project_dir / "runs" / "args.json").read_text(encoding="utf-8"))
    assert "--trainer.max_epochs" in args_payload
