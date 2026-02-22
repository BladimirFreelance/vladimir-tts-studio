from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

from training.infer import synth_with_piper


class _Chunk:
    def __init__(self, data: bytes, sample_rate: int = 24000, sample_width: int = 2, sample_channels: int = 1):
        self.audio_int16_bytes = data
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.sample_channels = sample_channels


def _install_voice(monkeypatch, voice_cls) -> None:
    module = types.ModuleType("piper.voice")
    module.PiperVoice = voice_cls
    package = types.ModuleType("piper")
    package.voice = module
    monkeypatch.setitem(sys.modules, "piper", package)
    monkeypatch.setitem(sys.modules, "piper.voice", module)


def test_infer_uses_synthesize_wav(monkeypatch, tmp_path: Path) -> None:
    class Voice:
        @staticmethod
        def load(_path: str):
            return Voice()

        def synthesize_wav(self, _text: str, wave_file) -> None:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(22050)
            wave_file.writeframes(b"\x00\x00" * 10)

    _install_voice(monkeypatch, Voice)
    monkeypatch.setattr("training.infer.ensure_espeakbridge_import", lambda: None)

    out = tmp_path / "out.wav"
    synth_with_piper(tmp_path / "model.onnx", "Привет", out)

    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 22050


def test_infer_uses_chunk_synthesis(monkeypatch, tmp_path: Path) -> None:
    class Voice:
        @staticmethod
        def load(_path: str):
            return Voice()

        def synthesize(self, _text: str):
            return [_Chunk(b"\x01\x00" * 5, sample_rate=16000)]

    _install_voice(monkeypatch, Voice)
    monkeypatch.setattr("training.infer.ensure_espeakbridge_import", lambda: None)

    out = tmp_path / "chunk.wav"
    synth_with_piper(tmp_path / "model.onnx", "Привет", out)

    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
