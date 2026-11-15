"""Audit trail: always-on JSONL log + optional Langfuse tracing of every LLM call (LangChain callback)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from clinicalchat.config import Settings, get_settings

log = logging.getLogger(__name__)


class Auditor:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._lock = threading.Lock()
        self._handler = None
        self.langfuse_error = ""
        if self.settings.langfuse_enabled:
            try:
                from langfuse import Langfuse
                from langfuse.langchain import CallbackHandler

                Langfuse(public_key=self.settings.langfuse_public_key, secret_key=self.settings.langfuse_secret_key,
                         host=self.settings.langfuse_host)
                self._handler = CallbackHandler(public_key=self.settings.langfuse_public_key)
            except Exception as exc:
                self.langfuse_error = f"{type(exc).__name__}: {exc}"
                log.warning("Langfuse disabled: %s", self.langfuse_error)

    @property
    def langfuse_enabled(self) -> bool:
        return self._handler is not None

    def callbacks(self) -> List[Any]:
        return [self._handler] if self._handler else []

    @property
    def path(self) -> Path:
        return self.settings.audit_dir / f"audit-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"

    def record(self, event: Dict[str, Any]) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(l) for l in lines if l.strip()]
