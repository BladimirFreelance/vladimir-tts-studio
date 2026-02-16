from __future__ import annotations

import io
import json
import wave
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from dataset.manifest import read_manifest


def build_router(project_dir: Path) -> APIRouter:
    router = APIRouter()
    prompts_file = project_dir / "prompts" / "segments.txt"
    manifest_path = project_dir / "metadata" / "train.csv"
    bad_path = project_dir / "metadata" / "bad_lines.txt"
    index_path = project_dir / "metadata" / ".index_state.json"
    recordings_dir = project_dir / "recordings" / "wav_22050"

    prompts = [line.strip() for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_idx() -> int:
        if not index_path.exists():
            return 0
        return int(json.loads(index_path.read_text(encoding="utf-8")).get("index", 0))

    def save_idx(index: int) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps({"index": index}, ensure_ascii=False), encoding="utf-8")

    def parse_prompt(idx: int) -> tuple[str, str]:
        if idx >= len(prompts):
            return "", ""
        audio_id, text = prompts[idx].split("|", maxsplit=1)
        return audio_id, text

    @router.get("/api/next")
    def next_line() -> JSONResponse:
        idx = load_idx()
        audio_id, text = parse_prompt(idx)
        return JSONResponse({"audio_id": audio_id, "text": text, "index": idx + 1, "total": len(prompts), "done": idx >= len(prompts)})

    @router.get("/api/progress")
    def progress() -> JSONResponse:
        idx = load_idx()
        return JSONResponse({"recorded": idx, "total": len(prompts)})

    @router.post("/api/repeat")
    def repeat() -> JSONResponse:
        idx = load_idx()
        audio_id, text = parse_prompt(idx)
        return JSONResponse({"audio_id": audio_id, "text": text, "index": idx + 1, "total": len(prompts)})

    @router.post("/api/back")
    def back() -> JSONResponse:
        idx = max(load_idx() - 1, 0)
        save_idx(idx)
        audio_id, text = parse_prompt(idx)
        return JSONResponse({"audio_id": audio_id, "text": text, "index": idx + 1, "total": len(prompts)})

    @router.post("/api/bad")
    def mark_bad(payload: dict) -> JSONResponse:
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
    async def save(audio_id: str, text: str, file: UploadFile) -> JSONResponse:
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

        # keep manifest valid and up to date
        rows = read_manifest(manifest_path)
        existing = {audio: txt for audio, txt in rows}
        existing[f"recordings/wav_22050/{audio_id}.wav"] = text
        with manifest_path.open("w", encoding="utf-8") as f:
            for audio_rel, text_row in existing.items():
                f.write(f"{audio_rel}|{text_row}\n")

        return JSONResponse({"ok": True, "path": str(wav_path)})

    return router
