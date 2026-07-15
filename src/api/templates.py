from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def mount_frontend(app: FastAPI) -> None:
    static_dir = _FRONTEND_DIR / "static"
    index_html = _FRONTEND_DIR / "index.html"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    if index_html.is_file():

        @app.get("/", include_in_schema=False)
        def _index() -> FileResponse:
            return FileResponse(index_html, media_type="text/html")
