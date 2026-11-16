"""Opens or updates the patient's ONE care-team case and routes it (🩻 radiologist → 🧭 coordinator → 👩‍⚕️ clinician).
A case is only opened when the care team has something new to look at; simple doubts are just answered."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterator, List, Optional, Sequence

from clinicalchat.domain import workflow as wf
from clinicalchat.domain.conversation import Event
from clinicalchat.domain.roles import AGENT_LABELS, AgentName
from clinicalchat.infrastructure.repositories.cases import CaseRepository
from clinicalchat.infrastructure.repositories.patient_facts import PatientFactRepository
from clinicalchat.infrastructure.repositories.uploads import UploadRepository
from clinicalchat.presenters.answers import rule_based_findings
from clinicalchat.services.catalog import DocumentCatalog
from clinicalchat.services.safety import SafetyChecker
from clinicalchat.services.turn import PipelineRun, TurnContext

AI_BUSY = "AI explanation unavailable (service busy) — rule-based findings only"


class CaseFiler:
    def __init__(self, cases: CaseRepository, uploads: UploadRepository, facts: PatientFactRepository,
                 catalog: DocumentCatalog, safety: SafetyChecker):
        self.cases = cases
        self.uploads = uploads
        self.facts = facts
        self.catalog = catalog
        self.safety = safety

    # ------------------------------------------------------------------ decide
    def evidence_fp(self, key: str, risk_level: str, reads: bool = True) -> str:
        """Fingerprint of the patient's evidence (reports, facts, risk; + radiologist reads for the Insight cache).
        Unchanged fingerprint = nothing new for the Insight agent / nothing the care team hasn't already seen."""
        basis: Dict[str, Any] = {"uploads": sorted(u["doc_id"] for u in self.uploads.all(key)),
                                 "facts": sorted(f"{f['kind']}:{f['value']}" for f in self.facts.all(key)),
                                 "risk": risk_level}
        if reads:
            basis["reads"] = sorted(self.cases.read_imaging_docs(key))
        return hashlib.sha1(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:16]

    def needs_case(self, ctx: TurnContext, upload_doc_ids: Sequence[str]) -> bool:
        """Only when there is something new to review, the patient asks for a doctor, the clinician is waiting for the
        patient's reply, or escalating findings were never put in front of a clinician."""
        run, key = ctx.run, ctx.patient_key
        if upload_doc_ids or run.new_symptoms or ctx.plan.wants_review:
            return True
        open_ = self.cases.open_case(key)
        if open_:
            return open_["status"] == wf.RETURNED
        if run.risk.escalate:  # safety net: escalating findings that no case has covered yet
            return not any(c["payload"].get("evidence_fp") == run.case_fp for c in self.cases.list(patient_key=key))
        return False

    # ------------------------------------------------------------------ act
    def file(self, ctx: TurnContext, insight: Dict[str, Any], level: str, draft: str, issues: List[str],
             md: str) -> Iterator[Event]:
        """Save the patient's case (reusing the open one) and route it to the care team."""
        run, key = ctx.run, ctx.patient_key
        fields = dict(question=ctx.question, risk_level=level, escalate=bool(run.risk and run.risk.escalate),
                      review_required=bool(run.review_reasons or issues), payload=self._payload(key, run, insight, issues),
                      patient_draft=draft)
        existing = self.cases.open_case(key)
        unread = wf.unread_imaging(fields["payload"]["imaging"], self.cases.read_imaging_docs(key))
        if existing:
            case_id, status = existing["case_id"], wf.status_after_patient_update(existing, unread)
            case = self.cases.update(case_id, wf.patient_update(existing, status=status, **fields))
        else:
            status = wf.initial_status(unread)
            case_id = self.cases.create(patient_key=key, created_by=ctx.session.username, status=status, **fields)
            case = self.cases.require(case_id)
        ctx.turn.case_id = case_id
        yield Event("answer", "🩺 Your health picture — AI assessment", md,
                    data={"agent": "communication", "case_id": case_id})
        yield Event("case", "📁 Your case with the care team", _case_text(case, updated=bool(existing)),
                    data={"case_id": case_id, "updated": bool(existing), "status": status})

    def answer_doubt(self, key: Optional[str], md: str) -> Iterator[Event]:
        """A question about records the care team already has: answer it, no case opened or changed."""
        open_ = self.cases.open_case(key)
        if open_:
            tri = open_["payload"].get("triage") or {}
            who = f" · {tri['clinician_name']}" if tri.get("clinician_name") else ""
            md += (f"\n\n📁 Your case **{open_['case_id']}** is still with the care team "
                   f"({wf.STAGE_LABEL[open_['status']][0]}{who}) — this question did not change it.")
        yield Event("answer", "🩺 Your health picture — AI answer", md, data={"agent": "communication", "case_id": ""})

    def file_without_ai(self, ctx: TurnContext, insight: Optional[Dict[str, Any]] = None) -> Iterator[Event]:
        """The AI is unavailable (e.g. rate-limited): still file and route the case from the rule-based agents'
        findings, so reports — especially critical ones — always reach the care team."""
        run, insight = ctx.run, insight or {}
        level = run.risk.risk_level if run.risk else "MODERATE"
        if AI_BUSY not in run.review_reasons:
            run.review_reasons.append(AI_BUSY)
        conditions = list(insight.get("possible_conditions") or (run.analysis.possible_conditions if run.analysis else []))[:5]
        draft, md = rule_based_findings(conditions, _findings(run), level)
        label = AGENT_LABELS[AgentName.COMMUNICATION]
        if not run.file_case:
            md += ("\n\n_The AI assistant is busy right now, so this is the automatic check only. Ask again in a few "
                   "minutes for a full explanation._\n\n<sub>🤖 Automatic rule-based check — not a diagnosis.</sub>")
            ctx.turn.add_answer(label, md, AI_BUSY)
            yield from self.answer_doubt(ctx.patient_key, md)
            return
        md += ("\n\n_The AI assistant is busy right now, so I couldn't write a full explanation. Your reports have still "
               "been sent to the care team, who will review them. Ask again in a few minutes for the full explanation — "
               "it will update this same case._")
        md += "\n\n<sub>🤖 Automatic rule-based check — not a diagnosis. A clinician will confirm or correct it.</sub>"
        ctx.turn.add_answer(label, md, AI_BUSY)
        fallback = {**insight, "possible_conditions": conditions, "ai_unavailable": True,
                    "answer": insight.get("answer") or "AI unavailable — rule-based findings only.",
                    "confidence": insight.get("confidence", 0)}
        yield from self.file(ctx, fallback, level, draft, [], md)

    # ------------------------------------------------------------------ helpers
    def _payload(self, key: str, run: PipelineRun, insight: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
        return {"insight": insight, "facts": run.analysis.facts_text() if run.analysis else "",
                "risk": run.risk.model_dump() if run.risk else {}, "review_reasons": run.review_reasons,
                "guard_issues": issues, "sources": run.ext.sources if run.ext else [],
                "possible_conditions": insight.get("possible_conditions", []), "evidence_fp": run.case_fp,
                **self.care_team_payload(key, run)}

    def care_team_payload(self, key: str, run: PipelineRun) -> Dict[str, Any]:
        """What the radiologist and the coordinator need: imaging (for the radiology read), allergies with a contrast
        safety check, and the report types in the case."""
        imaging = [{"doc_id": f.section.doc_id, "title": f.section.title, "source": f.source, "date": f.section.report_date,
                    "impression": f.impression, "red_flags": f.red_flags, "abnormal": f.abnormal,
                    "text": f.section.text[:2500]} for f in (run.analysis.imaging if run.analysis else [])]
        domains = {self.catalog.domain_of(s) for s in (run.ext.sections if run.ext else [])}
        return {"imaging": imaging, "allergies": run.analysis.allergies if run.analysis else [],
                "contrast_alerts": self.safety.contrast_alerts(key) if imaging else [],
                "report_types": sorted(d.value for d in domains if d)}


def _findings(run: PipelineRun) -> List[str]:
    an = run.analysis
    if not an:
        return []
    found = [r.line() for r, _ in an.abnormal][:8]
    for f in an.imaging[:4]:
        notes = f.red_flags[:2] or f.abnormal[:2]
        if notes:
            found.append(f"{f.section.title or 'Imaging'}: " + "; ".join(notes))
    return found


def _case_text(case: Dict[str, Any], updated: bool) -> str:
    status = case["status"]
    assigned = (case["payload"].get("triage") or {}).get("clinician_name")
    return (f"📁 **Case {case['case_id']}** — "
            + ("your open case was **updated** with this question (no new case created). " if updated else "created for you. ")
            + f"**Now:** {wf.stage_label(case)[0]}. " + wf.next_step_text(status, assigned)
            + " The confirmed answer will appear here and under **My case updates** — ask *“what is the "
              "status of my case?”* any time. The guidance above applies right now.")
