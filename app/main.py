from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.doctor import run_doctor
from dataset.manifest import write_manifest
from dataset.segmenter import indexed_segments, split_to_segments
from dataset.text_cleaner import normalize_text
from studio.server import run_server
from training.export_onnx import export_onnx
from training.infer import synth_with_piper
from training.train import run_training
from utils import load_yaml, setup_logging


def prepare_dataset(text_file: Path, project: str) -> None:
    cfg = load_yaml(Path("configs/train_default.yaml"))
    project_dir = Path("data/projects") / project
    (project_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (project_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (project_dir / "recordings" / "wav_22050").mkdir(parents=True, exist_ok=True)

    raw = text_file.read_text(encoding="utf-8")
    cleaned = normalize_text(
        raw,
        replacements=cfg["text"]["replacements"],
        expand_abbreviations=cfg["text"].get("expand_common_abbreviations", True),
    )
    segments = split_to_segments(cleaned, cfg["text"]["split_regex"], cfg["text"]["max_chars"])
    indexed = indexed_segments(segments, prefix=project)

    prompts_path = project_dir / "prompts" / "segments.txt"
    prompts_path.write_text("\n".join(f"{audio_id}|{text}" for audio_id, text in indexed) + "\n", encoding="utf-8")
    test20 = project_dir / "prompts" / "test_script_20.txt"
    test20.write_text("\n".join(text for _id, text in indexed[:20]) + "\n", encoding="utf-8")

    manifest_rows = [(f"recordings/wav_22050/{audio_id}.wav", text) for audio_id, text in indexed]
    write_manifest(project_dir / "metadata" / "train.csv", manifest_rows)
    print(f"Prepared {len(indexed)} prompts at {prompts_path}")


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

    s_export = sub.add_parser("export")
    s_export.add_argument("--project", required=True)
    s_export.add_argument("--ckpt")

    s_test = sub.add_parser("test")
    s_test.add_argument("--model", required=True)
    s_test.add_argument("--text", required=True)
    s_test.add_argument("--out", required=True)

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
        run_training(Path("data/projects") / args.project, epochs=args.epochs, base_ckpt=args.base_ckpt)
    elif args.cmd == "export":
        onnx, cfg = export_onnx(args.project, Path("data/projects") / args.project, Path(args.ckpt) if args.ckpt else None)
        logging.info("Exported %s and %s", onnx, cfg)
    elif args.cmd == "test":
        synth_with_piper(Path(args.model), args.text, Path(args.out))
    elif args.cmd == "doctor":
        code = run_doctor(Path("data/projects") / args.project, auto_fix=args.auto_fix)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
