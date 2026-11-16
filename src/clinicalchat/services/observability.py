"""Small helpers that record agent results in the turn's Langfuse trace (no-ops when tracing is off)."""

from __future__ import annotations

from typing import List

from clinicalchat.agents.guard import blocking
from clinicalchat.agents.llm.base import AgentResult
from clinicalchat.infrastructure.tracing import Span


def end_agent_span(span: Span, result: AgentResult) -> None:
    """Close an AI agent's span with its structured output, cache hit, latency and any error."""
    span.end(result.output, error=result.error,
             metadata={"parsed": result.parsed, "cache_hit": result.cache_hit, "latency_ms": result.latency_ms})
    confidence = (result.output or {}).get("confidence")
    if isinstance(confidence, (int, float)) and not result.error:
        span.score(f"{result.agent.value}_confidence", float(confidence), data_type="NUMERIC")


def guard_span(parent: Span, text: str, issues: List[str]) -> None:
    """The deterministic guard on patient-facing text, recorded as a Langfuse guardrail observation."""
    span = parent.child("patient-text-guard", as_type="guardrail", input={"text": text})
    held = blocking(issues)
    span.end({"issues": issues, "held_for_clinician": bool(held)}, level="WARNING" if held else None)
