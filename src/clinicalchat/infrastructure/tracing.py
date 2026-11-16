"""Langfuse observability (optional — everything is a no-op without keys).

Every chat turn, report upload, care-team action and index build is ONE Langfuse trace (tagged with the user's
role, user id and chat session). Inside it each agent step is a nested observation:

    chat-turn (trace: user, session, role tags)
      ├─ router                          span
      ├─ data-extraction                 agent
      ├─ clinical-analysis               agent
      │    ├─ knowledge-graph-safety     tool
      │    └─ lab / radiology / allergy  agent → LLM generation (tokens, cost, latency)
      ├─ risk-detection                  agent  (+ risk_level score)
      ├─ retrieval                       retriever → embedding (OpenAI text-embedding-3-small)
      ├─ insight-generation              agent → LLM generation (+ confidence score)
      ├─ web-search                      tool (Tavily)
      └─ patient-communication           agent → LLM generation, guardrail

Why explicit parents instead of "current span" context managers: the chat is a Python generator that Gradio
resumes on different worker threads, so thread-local "current span" state cannot be trusted across ``yield``.
Each :class:`Span` therefore creates its children directly. :func:`activate` sets a short-lived "current parent"
only around synchronous blocks (e.g. a vector search) so deep infrastructure code (embeddings) can nest under it.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from clinicalchat.config import Settings

log = logging.getLogger(__name__)

_ACTIVE: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar("clinicalchat_active_span", default=None)


def _short(value: Any, limit: int = 4000) -> Any:
    """Keep trace payloads small (and avoid shipping whole records)."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"… [{len(value) - limit} more chars]"
    return value


class Span:
    """A Langfuse observation, or a silent no-op when tracing is off (Null Object — callers never check)."""

    def __init__(self, obs: Any = None, tracer: Optional["Tracer"] = None):
        self._obs = obs
        self._tracer = tracer

    @property
    def enabled(self) -> bool:
        return self._obs is not None

    @property
    def trace_id(self) -> str:
        return getattr(self._obs, "trace_id", "") or ""

    @property
    def id(self) -> str:
        return getattr(self._obs, "id", "") or ""

    def child(self, name: str, as_type: str = "span", input: Any = None, metadata: Optional[dict] = None,
              **kwargs: Any) -> "Span":
        if self._obs is None:
            return NULL_SPAN
        try:
            return Span(self._obs.start_observation(name=name, as_type=as_type, input=_short(input),
                                                    metadata=metadata, **kwargs), self._tracer)
        except Exception as exc:  # tracing must never break the app
            log.debug("Langfuse child span failed: %s", exc)
            return NULL_SPAN

    def end(self, output: Any = None, *, error: str = "", metadata: Optional[dict] = None, **kwargs: Any) -> None:
        if self._obs is None:
            return
        try:
            update: Dict[str, Any] = dict(kwargs)
            if output is not None:
                update["output"] = _short(output)
            if metadata:
                update["metadata"] = metadata
            if error:
                update.update(level="ERROR", status_message=error[:500])
            if update:
                self._obs.update(**update)
            self._obs.end()
        except Exception as exc:
            log.debug("Langfuse span end failed: %s", exc)

    def update(self, **kwargs: Any) -> None:
        if self._obs is not None:
            try:
                self._obs.update(**kwargs)
            except Exception as exc:
                log.debug("Langfuse span update failed: %s", exc)

    @contextmanager
    def step(self, name: str, as_type: str = "span", input: Any = None, metadata: Optional[dict] = None
             ) -> Iterator["Span"]:
        """A child observation ended automatically (ERROR level if the block raises)."""
        span = self.child(name, as_type, input=input, metadata=metadata)
        try:
            yield span
        except Exception as exc:
            span.end(error=f"{type(exc).__name__}: {exc}")
            raise
        span.end()

    def score(self, name: str, value: Any, comment: str = "", data_type: Optional[str] = None) -> None:
        """Attach an evaluation score to the whole trace (e.g. risk level, AI confidence)."""
        if self._obs is None:
            return
        try:
            kwargs = {"data_type": data_type} if data_type else {}
            self._obs.score_trace(name=name, value=value, comment=comment or None, **kwargs)
        except Exception as exc:
            log.debug("Langfuse score failed: %s", exc)

    def langchain_config(self, tags: Optional[List[str]] = None, metadata: Optional[dict] = None) -> Dict[str, Any]:
        """LangChain run config whose LLM calls appear as generations nested under this observation."""
        cfg: Dict[str, Any] = {"tags": list(tags or []), "metadata": dict(metadata or {})}
        handler = self._tracer.langchain_handler(self) if self._tracer and self._obs is not None else None
        cfg["callbacks"] = [handler] if handler else []
        return cfg


NULL_SPAN = Span()


@contextmanager
def activate(span: Span) -> Iterator[Span]:
    """Make ``span`` the parent for infrastructure calls inside a synchronous block (never across ``yield``)."""
    token = _ACTIVE.set(span)
    try:
        yield span
    finally:
        _ACTIVE.reset(token)


def active_span() -> Span:
    return _ACTIVE.get() or NULL_SPAN


class Tracer:
    """Creates root traces and LangChain callback handlers bound to a parent observation."""

    def __init__(self, settings: Optional[Settings] = None):
        self._client = None
        self._public_key = settings.langfuse_public_key if settings else ""
        self.error = ""
        if settings is None or not settings.langfuse_enabled:
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                                    host=settings.langfuse_host)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("Langfuse disabled: %s", self.error)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def trace(self, name: str, *, user_id: str = "", session_id: str = "", tags: Optional[List[str]] = None,
              input: Any = None, metadata: Optional[dict] = None, as_type: str = "span") -> Span:
        """Start a new trace; its first observation carries the user / session / tags shown in Langfuse."""
        if self._client is None:
            return NULL_SPAN
        try:
            from langfuse import propagate_attributes

            attrs: Dict[str, Any] = {"trace_name": name, "tags": list(tags or [])}
            if user_id:
                attrs["user_id"] = user_id
            if session_id:
                attrs["session_id"] = session_id
            with propagate_attributes(**attrs):
                obs = self._client.start_observation(name=name, as_type=as_type, input=_short(input),
                                                     metadata=metadata)
            return Span(obs, self)
        except Exception as exc:
            log.debug("Langfuse trace failed: %s", exc)
            return NULL_SPAN

    def langchain_handler(self, parent: Span) -> Any:
        try:
            from langfuse.langchain import CallbackHandler

            return CallbackHandler(public_key=self._public_key,
                                   trace_context={"trace_id": parent.trace_id, "parent_span_id": parent.id})
        except Exception as exc:
            log.debug("Langfuse LangChain handler failed: %s", exc)
            return None

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:
                log.debug("Langfuse flush failed: %s", exc)

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.shutdown()
            except Exception as exc:
                log.debug("Langfuse shutdown failed: %s", exc)


NULL_TRACER = Tracer()
