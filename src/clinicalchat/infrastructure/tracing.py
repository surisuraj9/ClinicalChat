"""Optional Langfuse tracing: every LLM call and web search becomes a span tagged with the user's role and session."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from clinicalchat.config import Settings
from clinicalchat.domain.conversation import ChatSession

log = logging.getLogger(__name__)


class Tracer:
    def __init__(self, settings: Settings):
        self._handler = None
        self.error = ""
        if not settings.langfuse_enabled:
            return
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                     host=settings.langfuse_host)
            self._handler = CallbackHandler(public_key=settings.langfuse_public_key)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("Langfuse disabled: %s", self.error)

    @property
    def enabled(self) -> bool:
        return self._handler is not None

    def callbacks(self) -> List[Any]:
        return [self._handler] if self._handler else []

    def config_for(self, session: ChatSession) -> Dict[str, Any]:
        """LangChain run config for one chat turn."""
        return {"callbacks": self.callbacks(), "tags": [session.role.value],
                "metadata": {"user": session.username, "session_id": session.session_id}}
