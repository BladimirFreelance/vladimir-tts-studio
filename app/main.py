from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Поддержка запуска как `python app/main.py ...` без конфликта с внешним
    # пакетом `app`, установленным в окружении.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



def ensure_project_venv_for_critical_commands(cmd: str) -> None:
    if cmd not in {"train", "doctor"}:
        return

    repo_root = Path(__file__).resolve().parent.parent
    project_venv = repo_root / ".venv"
    if not project_venv.exists():
        return

    current_python = Path(sys.executable).resolve()
    in_project_venv = project_venv.resolve() in current_python.parents
    if in_project_venv:
        return

    print(
        "Вы сейчас запускаете проект не из .venv. "
        "Активируйте .\\.venv\\Scripts\\Activate.ps1\n"
        "или выполните: . .\\scripts\\00_bootstrap.ps1"
    )
    raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("vladimir-voice-lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_prepare = sub.add_parser("prepare")
    s_prepare.add_argument("--text", required=True)
    s_prepare.add_argument("--project", required=True)

    s_record = sub.add_parser(
        "record",
        help=(
            "Запустить локальный FastAPI-сервер со студией записи "
            "(сохранение WAV в recordings/wav_22050/)"
        ),
    )
    s_record.add_argument("--project", required=True)
    s_record.add_argument("--port", type=int, default=8765)

    s_train = sub.add_parser("train")
    s_train.add_argument("--project", required=True)
    s_train.add_argument(
        "--model.vocoder_warmstart_ckpt",
        dest="vocoder_warmstart_ckpt",
        help=(
            "Путь к базовому Piper checkpoint для warmstart без восстановления "
            "состояния тренера/эпох"
        ),
    )
    s_train.add_argument("--epochs", type=int, default=50)
    s_train.add_argument("--batch-size", type=int, dest="batch_size")

    s_export = sub.add_parser("export")
    s_export.add_argument("--project", required=True)
    s_export.add_argument("--ckpt")

    s_test = sub.add_parser("test")
    s_test.add_argument("--model", required=True)
    s_test.add_argument("--config")
    s_test.add_argument("--text", required=True)
    s_test.add_argument("--out", default="voices_out/test_voice.wav")
    s_test.add_argument("--mode", choices=["text", "espeak"], default="text")

    s_doctor = sub.add_parser("doctor")
    s_doctor.add_argument("--project", required=True)
    s_doctor.add_argument("--auto-fix", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ensure_project_venv_for_critical_commands(args.cmd)

    from utils import setup_logging

    setup_logging()

    if args.cmd == "prepare":
        from app.workflows import prepare_dataset
        from app.doctor import run_doctor

        prepare_dataset(Path(args.text), args.project)
        code = run_doctor(
            Path("data/projects") / args.project, auto_fix=False, require_audio=False
        )
        if code != 0:
            logging.warning(
                "Prepare finished with doctor warnings. Continue to recording step."
            )
    elif args.cmd == "record":
        from studio.server import run_server

        logging.info(
            "Starting studio UI for project '%s' on port %s "
            "(recordings -> data/projects/%s/recordings/wav_22050/)",
            args.project,
            args.port,
            args.project,
        )
        run_server(Path("data/projects") / args.project, port=args.port)
    elif args.cmd == "train":
        from training.train import run_training

        run_training(
            Path("data/projects") / args.project,
            epochs=args.epochs,
            vocoder_warmstart_ckpt=args.vocoder_warmstart_ckpt,
            batch_size=args.batch_size,
        )
    elif args.cmd == "export":
        from training.export_onnx import export_onnx

        onnx, cfg = export_onnx(
            args.project,
            Path("data/projects") / args.project,
            Path(args.ckpt) if args.ckpt else None,
        )
        logging.info("Exported %s and %s", onnx, cfg)
    elif args.cmd == "test":
        from training.infer import synth_with_piper

        synth_with_piper(
            Path(args.model),
            args.text,
            Path(args.out),
            mode=args.mode,
            config_path=Path(args.config) if args.config else None,
        )
    elif args.cmd == "doctor":
        from app.doctor import run_doctor

        code = run_doctor(Path("data/projects") / args.project, auto_fix=args.auto_fix)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
