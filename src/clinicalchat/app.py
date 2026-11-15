"""ASGI app: FastAPI (health + logo) with the Gradio chat UI mounted at `/` behind persona login."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse

from clinicalchat.config import get_settings
from clinicalchat.llm import describe_llm
from clinicalchat.personas import authenticate
from clinicalchat.ui import CSS, build_ui, login_message

LOGO = Path(__file__).parent / "assets" / "logo.svg"


def create_app() -> FastAPI:
    settings = get_settings()
    auth_enabled = os.getenv("UI_AUTH", "on").lower() not in {"0", "off", "false", "no"}
    app = FastAPI(title="ClinicalChat", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def force_light_theme(request: Request, call_next):
        if request.method == "GET" and request.url.path == "/" and "__theme" not in request.query_params:
            query = str(request.url.query)
            return RedirectResponse(url="/?" + (query + "&" if query else "") + "__theme=light", status_code=307)
        return await call_next(request)

    @app.get("/brand/logo.svg", include_in_schema=False)
    def logo():
        return FileResponse(LOGO, media_type="image/svg+xml")

    @app.get("/health")
    def health():
        return {"status": "ok", "llm": describe_llm()}

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate")
    return gr.mount_gradio_app(
        app, build_ui(auth_enabled=auth_enabled), path="/",
        auth=authenticate if auth_enabled else None,
        auth_message=login_message() if auth_enabled else None,
        css=CSS, theme=theme, favicon_path=str(LOGO), footer_links=[], show_error=True,
    )


def serve() -> None:
    import uvicorn

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    settings = get_settings()
    from clinicalchat.engine import get_engine

    get_engine()  # warm up: parse corpus + vector index before accepting traffic
    uvicorn.run(create_app(), host=settings.host, port=settings.port)
