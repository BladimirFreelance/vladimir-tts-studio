from __future__ import annotations

import argparse
import json
import subprocess
import wave

from training.utils import ensure_espeakbridge_import
from pathlib import Path


def _resolve_onnx_config_path(
    model_path: Path, config_path: Path | None = None
) -> Path:
    if config_path is not None:
        return config_path
    return model_path.with_suffix(".onnx.json")


def _load_onnx_config(model_path: Path, config_path: Path | None = None) -> dict:
    config_path = _resolve_onnx_config_path(model_path, config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Не найден конфиг ONNX: {config_path}")

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Некорректный JSON в {config_path}") from exc


def _extract_phoneme_type(config: dict) -> str | None:
    if isinstance(config.get("phoneme_type"), str):
        return config["phoneme_type"]

    if isinstance(config.get("phonemize"), dict):
        phonemize = config["phonemize"]
        if isinstance(phonemize.get("phoneme_type"), str):
            return phonemize["phoneme_type"]

    return None


def _to_espeak_phonemes(text: str, voice: str = "ru") -> str:
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "--ipa=3", "-v", voice, text],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("espeak-ng не найден в PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"espeak-ng завершился с кодом {exc.returncode}: {exc.stderr.strip()}"
        ) from exc

    phonemes = " ".join(result.stdout.split())
    if not phonemes:
        raise RuntimeError("espeak-ng вернул пустую фонемную строку")
    return phonemes


def _prepare_input(
    model_path: Path, text: str, mode: str, config_path: Path | None = None
) -> str:
    if mode == "text":
        return text

    if mode != "espeak":
        raise ValueError(f"Неизвестный режим синтеза: {mode}")

    config = _load_onnx_config(model_path, config_path)
    phoneme_type = _extract_phoneme_type(config)
    if phoneme_type != "espeak":
        raise RuntimeError(
            f"Режим 'espeak' требует phoneme_type='espeak' в ONNX JSON (получено: {phoneme_type!r})"
        )

    phonemes = _to_espeak_phonemes(text)
    return f"[[{phonemes}]]"


def synth_with_piper(
    model_path: Path,
    text: str,
    out_path: Path,
    mode: str = "text",
    config_path: Path | None = None,
) -> None:
    ensure_espeakbridge_import()
    try:
        from piper.voice import PiperVoice
    except Exception as exc:
        raise RuntimeError(
            "Не удалось импортировать piper.voice. Установите piper-tts"
        ) from exc

    input_text = _prepare_input(model_path, text, mode, config_path=config_path)

    voice = PiperVoice.load(str(model_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(voice, "synthesize_wav"):
        with wave.open(str(out_path), "wb") as wf:
            voice.synthesize_wav(input_text, wf)
        return

    if hasattr(voice, "synthesize"):
        chunks = voice.synthesize(input_text)
        audio_bytes = b""
        sample_rate, sample_width, sample_channels = 22050, 2, 1
        for chunk in chunks:
            audio_bytes += getattr(chunk, "audio_int16_bytes", b"")
            sample_rate = int(getattr(chunk, "sample_rate", sample_rate))
            sample_width = int(getattr(chunk, "sample_width", sample_width))
            sample_channels = int(getattr(chunk, "sample_channels", sample_channels))

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(sample_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        return

    if hasattr(voice, "synthesize_stream_raw"):
        audio_bytes = b"".join(voice.synthesize_stream_raw(input_text))
    elif hasattr(voice, "synthesize_raw"):
        audio_bytes = voice.synthesize_raw(input_text)
    else:
        raise RuntimeError("PiperVoice API не поддерживает доступные способы синтеза")

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config", help="Путь к ONNX JSON (по умолчанию <model>.onnx.json)"
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", default="voices_out/test_voice.wav")
    parser.add_argument("--mode", choices=["text", "espeak"], default="text")
    args = parser.parse_args()

    synth_with_piper(
        Path(args.model),
        args.text,
        Path(args.out),
        mode=args.mode,
        config_path=Path(args.config) if args.config else None,
    )
    print(f"Saved test audio: {args.out}")


if __name__ == "__main__":
    main()
