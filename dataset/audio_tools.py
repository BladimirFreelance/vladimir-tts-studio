from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile


@dataclass
class WavInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    duration_sec: float


def inspect_wav(path: Path) -> WavInfo:
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        frames = w.getnframes()
    bits = sample_width * 8
    duration = frames / float(sample_rate) if sample_rate else 0.0
    return WavInfo(sample_rate=sample_rate, channels=channels, bits_per_sample=bits, duration_sec=duration)


def analyze_quality(path: Path) -> dict[str, float | bool]:
    sr, data = wavfile.read(path)
    samples = data.astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    clipping = bool(np.any(np.abs(samples) >= np.iinfo(data.dtype).max)) if samples.size else False
    noise_floor = float(np.percentile(np.abs(samples), 10)) if samples.size else 0.0
    return {
        "sample_rate": float(sr),
        "peak": peak,
        "rms": rms,
        "noise_floor": noise_floor,
        "clipping": clipping,
    }


def convert_wav_python(source: Path, target: Path, sample_rate: int = 22050) -> None:
    src_rate, data = wavfile.read(source)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if src_rate != sample_rate:
        ratio = sample_rate / src_rate
        new_len = int(round(len(data) * ratio))
        xp = np.linspace(0, len(data) - 1, num=len(data), dtype=np.float32)
        xq = np.linspace(0, len(data) - 1, num=new_len, dtype=np.float32)
        data = np.interp(xq, xp, data)
    max_val = np.max(np.abs(data)) if data.size else 1.0
    if max_val > 0:
        data = data / max_val * 0.95
    pcm = (data * np.iinfo(np.int16).max).astype(np.int16)
    target.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(target, sample_rate, pcm)


def convert_wav(source: Path, target: Path, sample_rate: int = 22050) -> str:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(target),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return "ffmpeg"
    except Exception:
        convert_wav_python(source, target, sample_rate=sample_rate)
        return "python"
