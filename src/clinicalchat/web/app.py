"""ASGI app: FastAPI (health, logo, sign-up) with the Gradio chat UI mounted at `/` behind persona login."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import gradio as gr
from fastapi import FastAPI, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from clinicalchat.container import Container, get_container
from clinicalchat.infrastructure.llm.models import describe_llm, warm_up
from clinicalchat.web.api import api_router
from clinicalchat.web.content import login_message
from clinicalchat.web.controller import UIController
from clinicalchat.web.layout import build_ui
from clinicalchat.web.signup_page import render_done, render_form
from clinicalchat.web.theme import CSS

LOGO = Path(__file__).parent / "assets" / "logo.svg"
OFF = {"0", "off", "false", "no"}


def _llm_label(info: dict) -> str:
    return f"{info.get('provider')} {info.get('model', '')}".strip() if info.get("provider") != "unavailable" else "LLM unavailable"


def create_app(container: Optional[Container] = None) -> FastAPI:
    container = container or get_container()
    settings = container.settings
    auth_enabled = os.getenv("UI_AUTH", "on").lower() not in OFF

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        # graceful stop (systemd / Ctrl+C): finish background care-team insights, then send the last traces
        try:
            await run_in_threadpool(container.insight.drain, 60)
        except Exception as exc:  # a failed background job must not block the shutdown
            logging.getLogger(__name__).warning("Background work did not finish cleanly: %s", exc)
        container.tracer.shutdown()

    app = FastAPI(title="ClinicalChat API", lifespan=lifespan, docs_url="/docs", redoc_url=None,
                  description="Multi-agent clinical decision support. Log in with the same username / password as the UI.")
    app.include_router(api_router(container))

    @app.middleware("http")
    async def force_light_theme(request: Request, call_next):
        if request.method == "GET" and request.url.path == "/" and "__theme" not in request.query_params:
            query = str(request.url.query)
            return RedirectResponse(url="/?" + (query + "&" if query else "") + "__theme=light", status_code=307)
        return await call_next(request)

    @app.get("/brand/logo.svg", include_in_schema=False)
    def logo():
        return FileResponse(LOGO, media_type="image/svg+xml")

    @app.get("/signup", include_in_schema=False)
    def signup_form():
        return HTMLResponse(render_form())

    @app.post("/signup", include_in_schema=False)
    async def signup_submit(full_name: str = Form(""), age: str = Form(""), gender: str = Form(""),
                            username: str = Form(""), password: str = Form(""), confirm: str = Form(""),
                            allergies: str = Form(""), medications: str = Form("")):
        ok, msg = await run_in_threadpool(container.accounts.signup, username, password, confirm, full_name, age,
                                          gender, allergies, medications)
        if ok:
            return HTMLResponse(render_done(username.strip().lower()))
        values = dict(full_name=full_name, age=age, gender=gender, username=username, allergies=allergies,
                      medications=medications)
        return HTMLResponse(render_form(values, msg), status_code=400)

    @app.get("/health")
    def health():
        return {"status": "ok", "llm": describe_llm(settings),
                "web_search": "tavily" if container.web.enabled else "off",
                "tracing": "langfuse" if settings.langfuse_enabled else "off"}

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate")
    ui = build_ui(UIController(container), _llm_label(describe_llm(settings)), auth_enabled=auth_enabled)
    return gr.mount_gradio_app(
        app, ui, path="/",
        auth=container.accounts.authenticate if auth_enabled else None,
        auth_message=login_message() if auth_enabled else None,
        css=CSS, theme=theme, favicon_path=str(LOGO), footer_links=[], show_error=True,
    )


def serve() -> None:
    import uvicorn

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    container = get_container()  # warm up: parse corpus + vector index before accepting traffic
    threading.Thread(target=warm_up, args=(container.llm, container.fast_llm), name="llm-warm-up", daemon=True).start()
    uvicorn.run(create_app(container), host=container.settings.host, port=container.settings.port)
