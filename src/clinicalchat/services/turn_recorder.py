"""Closes a chat turn: adds it to the session history and writes one audit-log line (who asked what, which agents
ran, what was blocked, risk, AI calls, cache hits, web searches, latency)."""

from __future__ import annotations

import time

from clinicalchat.infrastructure.audit_log import AuditLog
from clinicalchat.services.turn import TurnContext


class TurnRecorder:
    def __init__(self, audit: AuditLog):
        self.audit = audit

    def finish(self, ctx: TurnContext) -> None:
        session, turn, plan, run = ctx.session, ctx.turn, ctx.plan, ctx.run
        turn.agents = [a.value for a in plan.agents]
        session.turns.append(turn)
        if plan.patient_key:
            session.last_patient = plan.patient_key
        self.audit.record({
            "session": session.session_id, "user": session.username, "role": session.role.value,
            "patient": plan.patient_key, "question": turn.question, "mode": plan.mode, "agents": turn.agents,
            "denied_domains": [d.value for d in plan.denied_domains], "blocked": plan.blocked,
            "drugs": plan.drugs, "symptoms": plan.symptoms, "safety_shown": bool(turn.safety),
            "risk_level": run.risk.risk_level if run.risk else None, "escalate": bool(run.risk and run.risk.escalate),
            "review_reasons": run.review_reasons, "case_id": turn.case_id,
            "llm_calls": sum(1 for r in run.llm if not r.cache_hit and not r.error),
            "cache_hits": sum(1 for r in run.llm if r.cache_hit),
            "errors": [r.error for r in run.llm if r.error],
            "web_searches": [w.audit() for w in run.web],
            "latency_ms": int((time.perf_counter() - ctx.started) * 1000),
        })
