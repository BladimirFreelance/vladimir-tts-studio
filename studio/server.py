from __future__ import annotations

import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from studio.api import build_router


def run_server(project_dir: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    app = FastAPI(title="Vladimir Piper Voice Studio")
    app.include_router(build_router(project_dir))

    web_root = Path(__file__).parent / "web"
    app.mount("/", StaticFiles(directory=str(web_root), html=True), name="web")

    if open_browser:
        webbrowser.open(f"http://{host}:{port}")

    uvicorn.run(app, host=host, port=port)
