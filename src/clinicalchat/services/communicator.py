"""Step 5 — 💬 Patient Communication for a case: writes the plain-language answer, checks it with the guard, then either
files / updates the patient's care-team case or just answers the doubt. For staff it is a patient-friendly draft."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator

from clinicalchat.agents.communication import communication_markdown
from clinicalchat.agents.guard import check_patient_text
from clinicalchat.domain.conversation import Event
from clinicalchat.domain.risk import PATIENT_GUIDANCE
from clinicalchat.domain.roles import AGENT_LABELS, AgentName
from clinicalchat.presenters.answers import patient_assessment_markdown, staff_draft_markdown
from clinicalchat.services.answer_writer import AnswerWriter
from clinicalchat.services.cases import CaseFiler
from clinicalchat.services.observability import end_agent_span, guard_span
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.turn import TurnContext

LABEL = AGENT_LABELS[AgentName.COMMUNICATION]


class Communicator:
    def __init__(self, writer: AnswerWriter, cases: CaseFiler, patients: PatientDirectory):
        self.writer = writer
        self.cases = cases
        self.patients = patients

    def communicate(self, ctx: TurnContext, insight: Dict[str, Any], risk_level: str, abnormal: bool,
                    question: str = "") -> Iterator[Event]:
        question = question or ctx.question
        yield Event("status", f"{LABEL} — writing", "Writing a plain-language explanation (AI)…", "pending",
                    "agent-communication")
        span = ctx.trace.child("patient-communication", as_type="agent", input={"question": question},
                               metadata={"risk_level": risk_level, "for_patient": ctx.is_patient})
        com = yield from self.writer.write(
            "💬 Writing your answer…" if ctx.is_patient else f"{LABEL} — writing…", ctx.llm_config(span), abnormal,
            question=question, patient=self.patients.summary(ctx.patient_key),
            risk=f"{risk_level}: {PATIENT_GUIDANCE[risk_level]}", insight=json.dumps(insight, ensure_ascii=False)[:6000])
        ctx.run.llm.append(com)
        end_agent_span(span, com)
        yield Event("status", f"{LABEL} — {'failed' if com.error else 'done'}", com.timing(), "done", "agent-communication")
        for_patient = ctx.is_patient and ctx.patient_key
        if com.error:
            if for_patient:
                yield from self.cases.file_without_ai(ctx, insight)
            else:
                yield Event("answer", LABEL, f"_Could not write the explanation right now._ {com.error}",
                            data={"agent": "communication"})
            return
        draft = communication_markdown(com.output or {})
        issues = check_patient_text([draft], abnormal)
        guard_span(ctx.trace, draft, issues)
        summary = str((com.output or {}).get("message", ""))
        if not for_patient:
            md = staff_draft_markdown(draft, issues)
            ctx.turn.add_answer(LABEL, md, summary)
            yield Event("answer", f"{LABEL} — patient-friendly draft", md, data={"agent": "communication"})
            return
        level = ctx.run.risk.risk_level if ctx.run.risk else risk_level
        md = patient_assessment_markdown(insight, level, draft, issues, reviewed=ctx.run.file_case)
        ctx.turn.add_answer(LABEL, md, summary)
        if ctx.run.file_case:
            yield from self.cases.file(ctx, insight, level, draft, issues, md)
        else:
            yield from self.cases.answer_doubt(ctx.patient_key, md)
