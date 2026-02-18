from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Поддержка запуска как `python app/main.py ...` без конфликта с внешним
    # пакетом `app`, установленным в окружении.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.doctor import run_doctor
from studio.server import run_server
from training.export_onnx import export_onnx
from training.infer import synth_with_piper
from training.train import run_training
from app.workflows import prepare_dataset
from utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("vladimir-voice-lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_prepare = sub.add_parser("prepare")
    s_prepare.add_argument("--text", required=True)
    s_prepare.add_argument("--project", required=True)

    s_record = sub.add_parser("record")
    s_record.add_argument("--project", required=True)
    s_record.add_argument("--port", type=int, default=8765)

    s_train = sub.add_parser("train")
    s_train.add_argument("--project", required=True)
    s_train.add_argument("--base_ckpt")
    s_train.add_argument("--epochs", type=int, default=50)
    s_train.add_argument("--batch-size", type=int, dest="batch_size")

    s_export = sub.add_parser("export")
    s_export.add_argument("--project", required=True)
    s_export.add_argument("--ckpt")

    s_test = sub.add_parser("test")
    s_test.add_argument("--model", required=True)
    s_test.add_argument("--text", required=True)
    s_test.add_argument("--out", required=True)
    s_test.add_argument("--mode", choices=["text", "espeak"], default="text")

    s_doctor = sub.add_parser("doctor")
    s_doctor.add_argument("--project", required=True)
    s_doctor.add_argument("--auto-fix", action="store_true")
    return p


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare_dataset(Path(args.text), args.project)
        code = run_doctor(Path("data/projects") / args.project, auto_fix=False, require_audio=False)
        if code != 0:
            logging.warning("Prepare finished with doctor warnings. Continue to recording step.")
    elif args.cmd == "record":
        run_server(Path("data/projects") / args.project, port=args.port)
    elif args.cmd == "train":
        run_training(
            Path("data/projects") / args.project,
            epochs=args.epochs,
            base_ckpt=args.base_ckpt,
            batch_size=args.batch_size,
        )
    elif args.cmd == "export":
        onnx, cfg = export_onnx(args.project, Path("data/projects") / args.project, Path(args.ckpt) if args.ckpt else None)
        logging.info("Exported %s and %s", onnx, cfg)
    elif args.cmd == "test":
        synth_with_piper(Path(args.model), args.text, Path(args.out), mode=args.mode)
    elif args.cmd == "doctor":
        code = run_doctor(Path("data/projects") / args.project, auto_fix=args.auto_fix)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
