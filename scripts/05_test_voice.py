from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.infer import synth_with_piper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Синтез тестового аудио через Piper. Режим 'text' ожидает IPA-фонемы, "
            "режим 'espeak' сначала прогоняет русский текст через espeak-ng --ipa=3."
        )
    )
    parser.add_argument("--model", required=True, help="Путь к ONNX модели Piper")
    parser.add_argument(
        "--config",
        help="Путь к ONNX JSON (по умолчанию <model>.onnx.json)",
    )
    parser.add_argument(
        "--text",
        required=True,
        help=(
            "Входная строка: IPA-фонемы в режиме text или обычный русский текст "
            "в режиме espeak"
        ),
    )
    parser.add_argument("--out", default="voices_out/test_voice.wav")
    parser.add_argument(
        "--mode",
        choices=["text", "espeak"],
        default="text",
        help="text = вход уже IPA; espeak = конвертация текста в IPA через espeak-ng",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    synth_with_piper(
        Path(args.model),
        args.text,
        Path(args.out),
        mode=args.mode,
        config_path=Path(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
