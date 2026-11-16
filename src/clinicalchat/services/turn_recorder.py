"""Opens and closes a chat turn: starts its Langfuse trace, then adds the turn to the session history, writes one
audit-log line (who asked what, which agents ran, what was blocked, risk, AI calls, cache hits, web searches,
latency, trace id) and closes the trace with the answer and scores."""

from __future__ import annotations

import time

from clinicalchat.domain.conversation import ChatSession
from clinicalchat.infrastructure.audit_log import AuditLog
from clinicalchat.infrastructure.tracing import Span, Tracer
from clinicalchat.services.turn import TurnContext


class TurnRecorder:
    def __init__(self, audit: AuditLog, tracer: Tracer):
        self.audit = audit
        self.tracer = tracer

    def start(self, question: str, session: ChatSession, name: str = "chat-turn") -> Span:
        return self.tracer.trace(name, user_id=session.username, session_id=session.session_id,
                                 tags=[name, session.role.value], input={"question": question},
                                 metadata={"role": session.role.value}, as_type="chain")

    def finish(self, ctx: TurnContext) -> None:
        session, turn, plan, run = ctx.session, ctx.turn, ctx.plan, ctx.run
        turn.agents = [a.value for a in plan.agents]
        turn.trace_id = ctx.trace.trace_id
        session.turns.append(turn)
        if plan.patient_key:
            session.last_patient = plan.patient_key
        latency_ms = int((time.perf_counter() - ctx.started) * 1000)
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
            "latency_ms": latency_ms, "trace_id": ctx.trace.trace_id or None,
        })
        self._close_trace(ctx, latency_ms)

    @staticmethod
    def _close_trace(ctx: TurnContext, latency_ms: int) -> None:
        turn, plan, run, trace = ctx.turn, ctx.plan, ctx.run, ctx.trace
        if not trace.enabled:
            return
        answer = turn.blocked or "\n\n".join(a.get("summary") or a.get("markdown", "") for a in turn.answers)
        trace.end(output={"answer": answer, "risk_level": run.risk.risk_level if run.risk else None,
                          "case_id": turn.case_id, "review_reasons": run.review_reasons},
                  metadata={"mode": plan.mode, "patient": plan.patient_key, "agents": turn.agents,
                            "denied_domains": [d.value for d in plan.denied_domains], "latency_ms": latency_ms,
                            "llm_calls": sum(1 for r in run.llm if not r.cache_hit and not r.error)})
        if run.risk:
            trace.score("risk_level", run.risk.risk_level, data_type="CATEGORICAL",
                        comment="; ".join(run.risk.escalation_reasons[:5]))
        if plan.blocked:
            trace.score("rbac_blocked", 1, data_type="BOOLEAN", comment=plan.blocked[:200])
        if run.review_reasons:
            trace.score("review_required", 1, data_type="BOOLEAN", comment="; ".join(run.review_reasons))
