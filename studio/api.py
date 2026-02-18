from __future__ import annotations

import io
import json
import logging
import threading
import traceback
import wave
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.doctor import run_doctor
from app.workflows import prepare_dataset
from dataset.manifest import read_manifest
from training.export_onnx import export_onnx
from training.infer import synth_with_piper
from training.train import run_training

LOGGER = logging.getLogger(__name__)


def build_router(project_dir: Path) -> APIRouter:
    router = APIRouter()
    prompts_file = project_dir / "prompts" / "segments.txt"
    manifest_path = project_dir / "metadata" / "train.csv"
    bad_path = project_dir / "metadata" / "bad_lines.txt"
    index_path = project_dir / "metadata" / ".index_state.json"
    recordings_dir = project_dir / "recordings" / "wav_22050"

    task_state: dict[str, Any] = {
        "running": False,
        "name": "",
        "status": "idle",
        "message": "",
        "error": "",
        "last_result": {},
    }
    task_lock = threading.Lock()

    def safe_prompts() -> list[str]:
        if not prompts_file.exists():
            return []
        return [
            line.strip()
            for line in prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def load_idx() -> int:
        if not index_path.exists():
            return 0
        return int(json.loads(index_path.read_text(encoding="utf-8")).get("index", 0))

    def save_idx(index: int) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps({"index": index}, ensure_ascii=False), encoding="utf-8"
        )

    def parse_prompt(idx: int) -> tuple[str, str]:
        prompts = safe_prompts()
        if idx >= len(prompts):
            return "", ""
        audio_id, text = prompts[idx].split("|", maxsplit=1)
        return audio_id, text

    def run_task(name: str, worker: Callable[[], dict[str, Any]]) -> bool:
        with task_lock:
            if task_state["running"]:
                return False
            task_state.update(
                {
                    "running": True,
                    "name": name,
                    "status": "running",
                    "message": "",
                    "error": "",
                    "last_result": {},
                }
            )

        def _thread_runner() -> None:
            try:
                result = worker() or {}
                with task_lock:
                    task_state.update(
                        {
                            "running": False,
                            "status": "done",
                            "message": "Готово",
                            "last_result": result,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Task %s failed", name)
                with task_lock:
                    task_state.update(
                        {
                            "running": False,
                            "status": "error",
                            "error": f"{exc}\n\n{traceback.format_exc()}",
                            "message": "Задача завершилась с ошибкой",
                        }
                    )

        threading.Thread(target=_thread_runner, daemon=True).start()
        return True

    @router.get("/api/status")
    def status() -> JSONResponse:
        prompts = safe_prompts()
        idx = load_idx()
        with task_lock:
            task = dict(task_state)
        return JSONResponse(
            {
                "project": project_dir.name,
                "project_dir": str(project_dir),
                "prepared": prompts_file.exists(),
                "recorded": idx,
                "total": len(prompts),
                "task": task,
            }
        )

    @router.get("/api/next")
    def next_line() -> JSONResponse:
        prompts = safe_prompts()
        idx = load_idx()
        audio_id, text = parse_prompt(idx)
        return JSONResponse(
            {
                "audio_id": audio_id,
                "text": text,
                "index": idx + 1,
                "total": len(prompts),
                "done": idx >= len(prompts),
            }
        )

    @router.post("/api/prepare")
    def prepare(payload: dict[str, Any]) -> JSONResponse:
        text_path = Path(payload.get("text_path", "")).expanduser()
        if not text_path.exists():
            raise HTTPException(status_code=400, detail=f"Файл не найден: {text_path}")

        def worker() -> dict[str, Any]:
            prepare_dataset(text_path, project_dir.name)
            code = run_doctor(project_dir, auto_fix=False, require_audio=False)
            return {"doctor_code": code}

        if not run_task("prepare", worker):
            raise HTTPException(status_code=409, detail="Уже выполняется другая задача")
        return JSONResponse({"ok": True})

    @router.post("/api/prepare/upload")
    async def upload_prepare_text(file: UploadFile) -> JSONResponse:
        filename = Path(file.filename or "").name
        if not filename:
            raise HTTPException(status_code=400, detail="Имя файла отсутствует")
        if Path(filename).suffix.lower() != ".txt":
            raise HTTPException(status_code=400, detail="Разрешены только .txt файлы")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Файл пустой")

        input_dir = project_dir / "input_texts"
        input_dir.mkdir(parents=True, exist_ok=True)
        dst = input_dir / filename
        dst.write_bytes(data)
        return JSONResponse({"ok": True, "text_path": str(dst)})

    @router.post("/api/train")
    def train(payload: dict[str, Any]) -> JSONResponse:
        epochs = int(payload.get("epochs", 50))
        base_ckpt = payload.get("base_ckpt") or None

        def worker() -> dict[str, Any]:
            run_training(project_dir, epochs=epochs, base_ckpt=base_ckpt)
            return {"epochs": epochs, "base_ckpt": base_ckpt}

        if not run_task("train", worker):
            raise HTTPException(status_code=409, detail="Уже выполняется другая задача")
        return JSONResponse({"ok": True})

    @router.post("/api/export")
    def onnx_export(payload: dict[str, Any]) -> JSONResponse:
        ckpt_raw = payload.get("ckpt") or None

        def worker() -> dict[str, Any]:
            onnx, cfg = export_onnx(
                project_dir.name, project_dir, Path(ckpt_raw) if ckpt_raw else None
            )
            return {"onnx": str(onnx), "config": str(cfg)}

        if not run_task("export", worker):
            raise HTTPException(status_code=409, detail="Уже выполняется другая задача")
        return JSONResponse({"ok": True})

    @router.post("/api/test")
    def test_voice(payload: dict[str, Any]) -> JSONResponse:
        model = Path(payload.get("model", "")).expanduser()
        text = str(payload.get("text", "")).strip()
        out = Path(payload.get("out", "")).expanduser()
        if not model.exists():
            raise HTTPException(status_code=400, detail=f"Модель не найдена: {model}")
        if not text:
            raise HTTPException(status_code=400, detail="Введите текст для синтеза")

        def worker() -> dict[str, Any]:
            synth_with_piper(model, text, out)
            return {"out": str(out)}

        if not run_task("test", worker):
            raise HTTPException(status_code=409, detail="Уже выполняется другая задача")
        return JSONResponse({"ok": True})

    @router.post("/api/doctor")
    def doctor(payload: dict[str, Any]) -> JSONResponse:
        auto_fix = bool(payload.get("auto_fix", False))

        def worker() -> dict[str, Any]:
            code = run_doctor(project_dir, auto_fix=auto_fix)
            return {"code": code, "auto_fix": auto_fix}

        if not run_task("doctor", worker):
            raise HTTPException(status_code=409, detail="Уже выполняется другая задача")
        return JSONResponse({"ok": True})

    @router.get("/api/progress")
    def progress() -> JSONResponse:
        prompts = safe_prompts()
        idx = load_idx()
        return JSONResponse({"recorded": idx, "total": len(prompts)})

    @router.post("/api/repeat")
    def repeat() -> JSONResponse:
        prompts = safe_prompts()
        idx = load_idx()
        audio_id, text = parse_prompt(idx)
        return JSONResponse(
            {
                "audio_id": audio_id,
                "text": text,
                "index": idx + 1,
                "total": len(prompts),
            }
        )

    @router.post("/api/back")
    def back() -> JSONResponse:
        prompts = safe_prompts()
        idx = max(load_idx() - 1, 0)
        save_idx(idx)
        audio_id, text = parse_prompt(idx)
        return JSONResponse(
            {
                "audio_id": audio_id,
                "text": text,
                "index": idx + 1,
                "total": len(prompts),
            }
        )

    @router.post("/api/bad")
    def mark_bad(payload: dict[str, Any]) -> JSONResponse:
        prompts = safe_prompts()
        idx = load_idx()
        audio_id = payload.get("audio_id", "")
        text = payload.get("text", "")
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        with bad_path.open("a", encoding="utf-8") as f:
            f.write(f"{audio_id}|{text}\n")
        if idx < len(prompts):
            save_idx(idx + 1)
        return JSONResponse({"ok": True})

    @router.post("/api/save")
    async def save(
        audio_id: str = Form(...), text: str = Form(...), file: UploadFile = File(...)
    ) -> JSONResponse:
        prompts = safe_prompts()
        if not audio_id:
            raise HTTPException(status_code=400, detail="audio_id is required")
        blob = await file.read()
        if not blob:
            raise HTTPException(status_code=400, detail="empty blob")

        recordings_dir.mkdir(parents=True, exist_ok=True)
        wav_path = recordings_dir / f"{audio_id}.wav"

        with wave.open(io.BytesIO(blob), "rb") as src:
            channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            framerate = src.getframerate()
            frames = src.readframes(src.getnframes())

        with wave.open(str(wav_path), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(sampwidth)
            out.setframerate(framerate)
            out.writeframes(frames)

        idx = load_idx()
        save_idx(min(idx + 1, len(prompts)))

        rows = read_manifest(manifest_path) if manifest_path.exists() else []
        existing = {audio: txt for audio, txt in rows}
        existing[f"recordings/wav_22050/{audio_id}"] = text
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            for audio_rel, text_row in existing.items():
                f.write(f"{audio_rel}|{text_row}\n")

        return JSONResponse({"ok": True, "path": str(wav_path)})

    return router
