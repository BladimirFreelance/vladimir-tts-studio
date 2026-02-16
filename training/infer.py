from __future__ import annotations

import argparse
import wave

from training.utils import ensure_espeakbridge_import
from pathlib import Path


def synth_with_piper(model_path: Path, text: str, out_path: Path) -> None:
    ensure_espeakbridge_import()
    try:
        from piper.voice import PiperVoice
    except Exception as exc:
        raise RuntimeError("Не удалось импортировать piper.voice. Установите piper-tts") from exc

    voice = PiperVoice.load(str(model_path))
    audio_bytes = b"".join(voice.synthesize_stream_raw(text))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    synth_with_piper(Path(args.model), args.text, Path(args.out))
    print(f"Saved test audio: {args.out}")


if __name__ == "__main__":
    main()
