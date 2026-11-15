"""Orchestrator — the 5-agent pipeline from the course slide:

  📄 Data Extraction → 🔬 Clinical Analysis → 🚨 Risk Detection → 💡 Insight Generation → 💬 Patient Communication
     (deterministic)      (deterministic:        (deterministic       (LLM, 1 call)           (LLM, 1 call, patients;
                           Lab Interpreter,       rules)                                        held for clinician review)
                           Radiology Analyzer,
                           Allergy Safety)

`ask()` / `upload()` are generators of `Event`s so the UI can show live progress. Deterministic safety information
(drug/allergy conflicts, emergency guidance) is shown immediately and is never held back.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from clinicalchat.agents import (AgentResult, InsightGenerationAgent, PatientCommunicationAgent, check_patient_text,
                                 communication_markdown)
from clinicalchat.analysis import AnalysisResult, ClinicalAnalysisAgent
from clinicalchat.audit import Auditor
from clinicalchat.engine import ClinicalEngine, alert_line
from clinicalchat.extraction import (MAX_UPLOAD_FILES, DataExtractionAgent, ExtractionResult, assess_document, foreign_name,
                                     parse_upload)
from clinicalchat.llm import get_chat_model
from clinicalchat.models import Domain
from clinicalchat.rbac import AGENT_LABELS, AgentName, Role, policy
from clinicalchat.risk import PATIENT_GUIDANCE, RiskAssessment, RiskDetectionAgent
from clinicalchat.router import RoutePlan, Router
from clinicalchat.store import Store, get_store

HISTORY_TURNS = 3
REVIEW_CONFIDENCE = 0.6
LEVEL_BADGE = {"LOW": "🟢 LOW", "MODERATE": "🟡 MODERATE", "HIGH": "🟠 HIGH", "CRITICAL": "🔴 CRITICAL"}


@dataclass
class Turn:
    question: str
    role: str
    patient_key: Optional[str]
    agents: List[str] = field(default_factory=list)
    safety: str = ""
    answers: List[Dict[str, Any]] = field(default_factory=list)  # {label, markdown, summary}
    blocked: str = ""
    case_id: str = ""
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))


@dataclass
class ChatSession:
    username: str
    role: Role
    display_name: str = ""
    own_patient: Optional[str] = None
    selected_patient: Optional[str] = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: List[Turn] = field(default_factory=list)
    last_patient: Optional[str] = None
    last_insight: Optional[Dict[str, Any]] = None  # {patient_key, question, output, risk}
    busy: bool = False  # a question is streaming (the UI poller must not touch the chat)

    def history_text(self) -> str:
        lines = []
        for t in self.turns[-HISTORY_TURNS:]:
            answer = " ".join(a["summary"] for a in t.answers)[:400] if t.answers else t.blocked
            lines.append(f"Q: {t.question}\nA: {answer}")
        return "\n".join(lines)


@dataclass
class Event:
    kind: str  # status | step | safety | guidance | answer | case | blocked
    title: str = ""
    text: str = ""
    status: str = "done"  # pending | done
    key: str = ""  # stable id so the UI can update a status line in place
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Run:
    """Everything one pipeline run produced (for audit and case creation)."""
    ext: Optional[ExtractionResult] = None
    analysis: Optional[AnalysisResult] = None
    risk: Optional[RiskAssessment] = None
    llm: List[AgentResult] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, engine: ClinicalEngine, llm: Optional[BaseChatModel] = None, auditor: Optional[Auditor] = None,
                 store: Optional[Store] = None):
        self.engine = engine
        self.router = Router(engine)
        self.llm = llm or get_chat_model()
        self.auditor = auditor or Auditor(engine.settings)
        self.store = store or get_store()
        self.extraction = DataExtractionAgent(engine)
        self.analysis = ClinicalAnalysisAgent(engine)
        self.risk = RiskDetectionAgent(engine.kg)
        self.insight = InsightGenerationAgent(self.llm)
        self.communication = PatientCommunicationAgent(self.llm)

    # ------------------------------------------------------------------ chat question
    def ask(self, question: str, session: ChatSession) -> Iterator[Event]:
        started = time.perf_counter()
        question = question.strip()
        turn = Turn(question=question, role=session.role.value, patient_key=None)
        run = _Run()

        yield Event("status", "🧭 Understanding your question", "Detecting the topic and checking access…", "pending", "route")
        context_patient = session.selected_patient or session.last_patient  # staff follow-ups keep the patient
        current = session.own_patient or context_patient
        has_insight = bool(session.last_insight and session.last_insight.get("patient_key") == current)
        plan = self.router.route(question, session.role, session_patient=context_patient,
                                 own_patient=session.own_patient, has_last_insight=has_insight)
        turn.patient_key = plan.patient_key
        yield Event("status", "🧭 Routed", self._route_text(plan, session), "done", "route", {"plan": _plan_dict(plan)})
        if plan.blocked:
            turn.blocked = plan.blocked
            yield Event("blocked", "🔒 Access", plan.blocked)
            return self._finish(session, turn, plan, run, started)

        if plan.mode == "general":
            yield from self._general(question, session, plan, turn, run)
        elif plan.mode == "explain" and has_insight:
            yield from self._explain_last(question, session, turn, run)
        else:
            if session.role == Role.PATIENT and plan.symptoms and plan.patient_key:
                self._save_symptoms(plan.patient_key, plan.symptoms)
            yield from self._pipeline(question, session, plan, turn, run)
        return self._finish(session, turn, plan, run, started)

    def run(self, question: str, session: ChatSession) -> List[Event]:
        return list(self.ask(question, session))

    # ------------------------------------------------------------------ report upload
    def upload(self, path: Path, session: ChatSession, display_name: Optional[str] = None,
               report_type: str = "auto") -> Iterator[Event]:
        """Single-file upload (kept for the CLI/tests) — see `upload_many`."""
        return self.upload_many([(Path(path), display_name or Path(path).name)], session, report_type=report_type)

    def upload_many(self, files: Sequence[Tuple[Path, str]], session: ChatSession, message: str = "",
                    report_type: str = "auto") -> Iterator[Event]:
        """Upload one or more reports, optionally with a message ("I have fever — here are my reports, is it serious?").
        Each file is read, checked and saved on its own; the 5-agent pipeline then runs ONCE over everything that was
        accepted, answering the message if there is one (same LLM cost as a single report)."""
        message = (message or "").strip()
        started = time.perf_counter()
        pol = policy(session.role)
        key = session.own_patient if pol.own_record_only else session.selected_patient
        names = [n for _, n in files]
        turn = Turn(question=message or f"[upload] {', '.join(names)}", role=session.role.value, patient_key=key)
        plan = RoutePlan(mode="case", patient_key=key, patient_source="session", check_allergies=True,
                         agents=[a for a in AgentName if a in pol.agents
                                 and (a != AgentName.COMMUNICATION or session.role == Role.PATIENT)])
        run = _Run()
        label = AGENT_LABELS[AgentName.EXTRACTION]
        if not pol.can_upload or not key:
            turn.blocked = ("Uploading reports is not available for your role." if not pol.can_upload
                            else "Select a patient first, then upload their report.")
            yield Event("blocked", "🔒 Upload", turn.blocked)
            return self._finish(session, turn, plan, run, started)
        if not files:
            turn.blocked = "No files were received."
            yield Event("blocked", "📄 Upload", turn.blocked)
            return self._finish(session, turn, plan, run, started)
        if len(files) > MAX_UPLOAD_FILES:
            turn.blocked = (f"You selected **{len(files)}** files. Please upload at most **{MAX_UPLOAD_FILES}** reports at a "
                            "time so each one can be checked properly.")
            yield Event("blocked", "📄 Upload", turn.blocked)
            return self._finish(session, turn, plan, run, started)
        if message:
            # the message decides symptoms / drugs / allergens to check; the record stays the session's patient
            yield Event("status", "🧭 Understanding your message", "Detecting symptoms, medicines and topic…", "pending", "route")
            routed = self.router.route(message, session.role, session_patient=key, own_patient=session.own_patient)
            if routed.blocked or (routed.patient_key and routed.patient_key != key):
                turn.blocked = routed.blocked or ("Your message is about a different patient than the one these reports are "
                                                  "being added to. Select that patient first, then attach the reports.")
                yield Event("status", "🧭 Routed", "Blocked", "done", "route")
                yield Event("blocked", "🔒 Access", turn.blocked)
                return self._finish(session, turn, plan, run, started)
            plan.symptoms, plan.drugs, plan.allergens = routed.symptoms, routed.drugs, routed.allergens
            plan.notes = routed.notes
            yield Event("status", "🧭 Routed", self._route_text(plan, session), "done", "route", {"plan": _plan_dict(plan)})

        patient = self.engine.patient(key)
        accepted: List[Tuple[str, Any]] = []  # (display name, ClinicalDocument)
        lines: List[str] = []
        seen: set = set()
        total = len(files)
        for i, (path, name) in enumerate(files, 1):
            prefix = f"({i}/{total}) " if total > 1 else ""
            yield Event("status", f"{label} — reading {prefix}{name}", "Extracting text (OCR for scans), sections and lab values…",
                        "pending", "agent-extraction")
            try:
                doc = parse_upload(Path(path), patient_key=key, patient_name=patient.name, report_type=report_type,
                                   uploaded_by=session.username, uploaded_at=_now(), display_name=name,
                                   cache_dir=self.engine.settings.work_dir / "ocr_cache")
            except Exception as exc:
                lines.append(f"- ❌ **{name}** — could not be read: {str(exc)[:160]}")
                continue
            if doc.doc_id in seen or doc.doc_id in self.engine.docs_by_id:
                lines.append(f"- ↩️ **{name}** — already on your record, skipped")
                continue
            other = foreign_name(doc, patient.name)
            if other:
                lines.append(f"- 🪪 **{name}** — the name on the report is **{other}**, not **{patient.name}**. "
                             "Not added (ask the lab for a corrected copy, or a clinician can add it).")
                continue
            seen.add(doc.doc_id)
            quality = assess_document(doc, patient.name)
            self.store.save_upload(doc.model_dump_json(), quality.model_dump(), key, doc.doc_id, name, session.username)
            self.engine.add_document(doc)
            accepted.append((name, doc))
            labs = sum(len(s.lab_results) for s in doc.sections)
            line = (f"- ✅ **{name}** — {doc.category.replace('_', ' ')} report ({doc.extraction_method}), "
                    f"{labs} lab value(s), extraction confidence **{quality.level}** ({quality.score:.0%})")
            line += "".join(f"\n  - ⚠️ {issue}" for issue in quality.issues[:3])
            if quality.low_confidence:
                line += "\n  - 🚩 Low-confidence extraction — flagged for clinician review"
            lines.append(line)

        head = (f"**{len(accepted)} of {total}** report(s) added to the record." if total > 1 else
                ("Report added to the record." if accepted else "The report was not added."))
        yield Event("step", f"{label} — reports checked", head + "\n" + "\n".join(lines), data={"agent": "extraction"})
        if not accepted:
            turn.blocked = "None of the uploaded files could be added, so there is nothing new to analyse."
            yield Event("status", f"{label} — nothing added", turn.blocked, "done", "agent-extraction")
            yield Event("blocked", "📄 Upload", turn.blocked)
            return self._finish(session, turn, plan, run, started)

        plan.focus = list(dict.fromkeys(d.domain for _, d in accepted if d.domain != Domain.REFERENCE))
        listed = ", ".join(f"'{n}'" for n, _ in accepted)
        word = "report" if len(accepted) == 1 else "reports"
        if message:
            if session.role == Role.PATIENT and plan.symptoms:
                self._save_symptoms(key, plan.symptoms)
            question = f"{message}\n(Attached {word}: {listed})"
        else:
            question = (f"Please review my newly uploaded {word} {listed}." if session.role == Role.PATIENT else
                        f"Review the newly uploaded {word} {listed} for {patient.name}.")
        turn.question = question
        yield from self._pipeline(question, session, plan, turn, run, upload_doc_ids=[d.doc_id for _, d in accepted])
        return self._finish(session, turn, plan, run, started)

    # ------------------------------------------------------------------ clinician review
    def review(self, case_id: str, reviewer: str, approve: bool, note: str = "", edited_message: str = "") -> Dict[str, Any]:
        case = self.store.case(case_id)
        if not case:
            raise ValueError(f"Unknown case {case_id}")
        status = "VALIDATED" if approve else "RETURNED"
        final = (edited_message.strip() or case["patient_draft"]) if approve else ""
        if approve and note.strip():
            final += f"\n\n**Note from your clinician:** {note.strip()}"
        self.store.review_case(case_id, reviewer=reviewer, status=status, note=note.strip(), final_message=final)
        topic = ", ".join(case["payload"].get("possible_conditions", [])[:3]) or case["question"][:80]
        self.store.add_feedback(case_id, reviewer, status, note.strip(), topic)
        self.auditor.record({"event": "review", "case": case_id, "reviewer": reviewer, "status": status,
                             "note": note.strip(), "edited": bool(edited_message.strip())})
        return self.store.case(case_id)

    # ------------------------------------------------------------------ pipeline
    def _pipeline(self, question: str, session: ChatSession, plan: RoutePlan, turn: Turn, run: _Run,
                  upload_doc_ids: Sequence[str] = ()) -> Iterator[Event]:
        e, role = self.engine, session.role
        patient = e.patient(plan.patient_key)
        is_patient = role == Role.PATIENT

        # 1) 📄 Data Extraction
        label = AGENT_LABELS[AgentName.EXTRACTION]
        if not upload_doc_ids:
            yield Event("status", f"{label} — gathering records", "Selecting the relevant reports and checking quality…",
                        "pending", "agent-extraction")
        ext = self.extraction.run(question, role, plan.patient_key, plan.focus, new_symptoms=plan.symptoms,
                                  upload_doc_ids=upload_doc_ids)
        run.ext = ext
        if not upload_doc_ids:
            summary = self._extraction_text(ext, is_patient)
            yield Event("status", f"{label} — done", summary.split("\n")[0], "done", "agent-extraction")
            yield Event("step", label, summary, data={"agent": "extraction"})
        else:
            yield Event("status", f"{label} — done", f"{len(upload_doc_ids)} report(s) processed", "done", "agent-extraction")

        # 2) 🔬 Clinical Analysis (Lab Interpreter, Radiology Analyzer, Allergy Safety, symptom differential)
        label = AGENT_LABELS[AgentName.ANALYSIS]
        yield Event("status", f"{label} — analysing", "Checking values against reference ranges, imaging findings, "
                    "allergies and symptoms…", "pending", "agent-analysis")
        an = self.analysis.run(ext, plan.drugs, plan.allergens, check_allergies=plan.check_allergies)
        run.analysis = an
        summary = self._analysis_text(an, is_patient, patient.medications if patient else ())
        yield Event("status", f"{label} — done", summary.split("\n")[0], "done", "agent-analysis")
        yield Event("step", label, summary, data={"agent": "analysis"})
        safety_md = self.safety_banner(plan)
        if safety_md:
            turn.safety = safety_md
            yield Event("safety", "🛡️ Safety check", safety_md)

        # 3) 🚨 Risk Detection
        label = AGENT_LABELS[AgentName.RISK]
        yield Event("status", f"{label} — scoring", "Applying safety rules…", "pending", "agent-risk")
        risk = self.risk.run(patient=patient, lab_results=an.lab_results,
                             imaging=[(f.section, f.red_flags, f.abnormal) for f in an.imaging], symptoms=ext.symptoms,
                             new_symptoms=ext.new_symptoms, proposed_drug_alerts=an.drug_alerts, medication_alerts=an.medication_alerts,
                             conditions=an.possible_conditions, inconsistencies=an.inconsistencies,
                             low_confidence=ext.quality.low_confidence)
        run.risk = risk
        summary = self._risk_text(risk, is_patient)
        yield Event("status", f"{label} — {risk.risk_level}", summary.split("\n")[0], "done", "agent-risk")
        yield Event("step", label, summary, data={"agent": "risk", "level": risk.risk_level})
        guidance = risk.clinical_level
        if is_patient and (ext.new_symptoms or upload_doc_ids or guidance != "LOW"):
            text = PATIENT_GUIDANCE[guidance]
            if guidance == "LOW" and ext.quality.low_confidence:
                text = ("🔍 **I couldn't read enough from this report to judge it**, so I can't say whether it is normal. "
                        "A clinician will check it. If you feel unwell, don't wait — see a doctor.")
            yield Event("guidance", "What to do now", text, data={"level": guidance})

        # 4) 💡 Insight Generation (LLM)
        config = self._config(session)
        label = AGENT_LABELS[AgentName.INSIGHT]
        yield Event("status", f"{label} — writing", "Consolidating the findings (AI)…" if not is_patient else
                    "Preparing a clinical summary for your clinician (AI)…", "pending", "agent-insight")
        audience = policy(Role.CLINICIAN if is_patient else role).audience
        ins = self.insight.run(question=question, audience=audience, patient=e.patient_summary(plan.patient_key),
                               extraction=self._extraction_brief(ext), facts=an.facts_text(), risk=self._risk_brief(risk),
                               feedback=self._feedback(), context=e.records_context(question, ext.sections, ext.focus),
                               history=session.history_text(), config=config)
        run.llm.append(ins)
        out = ins.output or {}
        run.review_reasons = self._review_reasons(ext, an, risk, ins)
        yield Event("status", f"{label} — {'failed' if ins.error else 'done'}", _timing(ins), "done", "agent-insight")
        session.last_insight = {"patient_key": plan.patient_key, "question": question, "output": out,
                                "risk": risk.risk_level, "facts": an.facts_text()}
        if not is_patient:
            md = render_insight(ins, run.review_reasons, ext.sources)
            turn.answers.append({"label": label, "markdown": md, "summary": str(out.get("answer", ins.error))[:300]})
            yield Event("answer", label, md, data={"agent": "insight", "review": run.review_reasons})
            if plan.mode != "explain":
                return

        # 5) 💬 Patient Communication (LLM) — patients: held for clinician validation
        if ins.error:
            if is_patient:
                yield Event("answer", "💬 Patient Communication", "_The AI assistant is busy right now, so I could not prepare "
                            "a written explanation. The safety guidance above still applies — please try again shortly._",
                            data={"agent": "communication"})
            return
        yield from self._communicate(question, session, plan, turn, run, out, risk.clinical_level, bool(an.abnormal or risk.escalate))

    def _communicate(self, question: str, session: ChatSession, plan: RoutePlan, turn: Turn, run: _Run,
                     insight: Dict[str, Any], risk_level: str, abnormal: bool) -> Iterator[Event]:
        is_patient = session.role == Role.PATIENT
        label = AGENT_LABELS[AgentName.COMMUNICATION]
        yield Event("status", f"{label} — writing", "Writing a plain-language explanation (AI)…", "pending", "agent-communication")
        com = self.communication.run(question=question, patient=self.engine.patient_summary(plan.patient_key),
                                     risk=f"{risk_level}: {PATIENT_GUIDANCE[risk_level]}",
                                     insight=json.dumps(insight, ensure_ascii=False)[:6000], config=self._config(session))
        run.llm.append(com)
        yield Event("status", f"{label} — {'failed' if com.error else 'done'}", _timing(com), "done", "agent-communication")
        if com.error:
            yield Event("answer", label, f"_Could not write the explanation right now._ {com.error}", data={"agent": "communication"})
            return
        draft = communication_markdown(com.output or {})
        issues = check_patient_text([draft], abnormal)
        if is_patient and plan.patient_key:
            level = run.risk.risk_level if run.risk else risk_level
            fields = dict(question=question, risk_level=level, escalate=bool(run.risk and run.risk.escalate),
                          review_required=bool(run.review_reasons or issues),
                          payload={"insight": insight, "facts": run.analysis.facts_text() if run.analysis else "",
                                   "risk": run.risk.model_dump() if run.risk else {}, "review_reasons": run.review_reasons,
                                   "guard_issues": issues, "sources": run.ext.sources if run.ext else [],
                                   "possible_conditions": insight.get("possible_conditions", [])},
                          patient_draft=draft)
            existing = self.store.open_case(plan.patient_key)
            if existing:
                case_id = existing["case_id"]
                self.store.update_case(case_id, **fields)
            else:
                case_id = self.store.create_case(patient_key=plan.patient_key, created_by=session.username, **fields)
            turn.case_id = case_id
            md = patient_assessment_markdown(insight, level, draft, issues)
            turn.answers.append({"label": label, "markdown": md, "summary": str((com.output or {}).get("message", ""))[:300]})
            yield Event("answer", "🩺 Your health picture — AI assessment", md, data={"agent": "communication",
                                                                                      "case_id": case_id})
            case_text = (f"📁 **Case {case_id}** — "
                         + ("your open case was **updated** with this question (no new case created). " if existing
                            else "created for you. ")
                         + "A clinician will check this AI assessment; their confirmed answer will appear here and under "
                           "**My case updates**. The guidance above applies right now.")
            yield Event("case", "⏳ Waiting for clinician confirmation", case_text,
                        data={"case_id": case_id, "updated": bool(existing)})
        else:
            md = draft + (("\n\n**🛡️ Guard check before sending:**\n" + "\n".join(f"- ⚠️ {i}" for i in issues))
                          if issues else "\n\n<sub>🛡️ Guard check passed (no doses, no definitive diagnosis).</sub>")
            turn.answers.append({"label": label, "markdown": md, "summary": str((com.output or {}).get("message", ""))[:300]})
            yield Event("answer", f"{label} — patient-friendly draft", md, data={"agent": "communication"})

    def _explain_last(self, question: str, session: ChatSession, turn: Turn, run: _Run) -> Iterator[Event]:
        last = session.last_insight or {}
        plan = RoutePlan(mode="explain", patient_key=last.get("patient_key"))
        yield from self._communicate(last.get("question", question), session, plan, turn, run, last.get("output", {}),
                                     last.get("risk", "LOW"), "outside reference range" in last.get("facts", "")
                                     or last.get("risk") in ("HIGH", "CRITICAL"))

    def _general(self, question: str, session: ChatSession, plan: RoutePlan, turn: Turn, run: _Run) -> Iterator[Event]:
        """General health question from a patient: informational answer, no personal records, no review needed."""
        safety_md = self.safety_banner(plan)
        if safety_md:
            turn.safety = safety_md
            yield Event("safety", "🛡️ Safety check", safety_md)
        context, sources = self.engine.reference_context(question)
        label = AGENT_LABELS[AgentName.COMMUNICATION]
        yield Event("status", f"{label} — answering", "Looking up reference information and writing an answer (AI)…",
                    "pending", "agent-communication")
        p = self.engine.patient(plan.patient_key)
        allergies = ", ".join(a.allergen for a in p.allergies) if p and p.allergies else "none recorded"
        com = self.communication.run(question=question, patient=f"Documented allergies: {allergies}",
                                     risk="Not applicable (general question).", insight=context, general=True,
                                     config=self._config(session))
        run.llm.append(com)
        yield Event("status", f"{label} — {'failed' if com.error else 'done'}", _timing(com), "done", "agent-communication")
        if com.error:
            md = f"_The AI assistant is busy right now — please try again shortly._ ({com.error[:120]})"
        else:
            md = communication_markdown(com.output or {})
            md += "\n\n<sub>ℹ️ General health information, not personal medical advice."
            md += f" Sources: {', '.join(sources)}</sub>" if sources else "</sub>"
        turn.answers.append({"label": label, "markdown": md, "summary": md[:300]})
        yield Event("answer", label, md, data={"agent": "communication"})

    # ------------------------------------------------------------------ helpers
    def _save_symptoms(self, key: str, symptoms: List[str]) -> None:
        known = set(self.engine.kg.patient_symptoms(key))
        for s in symptoms:
            if s not in known:
                self.store.add_fact(key, "symptom", s)
                self.engine.apply_fact(key, "symptom", s)

    def _feedback(self) -> str:
        notes = self.store.recent_feedback(3)
        return "\n".join(f"- ({n['action'].lower()}) {n['note']}" for n in notes) or "None."

    def _config(self, session: ChatSession) -> dict:
        return {"callbacks": self.auditor.callbacks(), "tags": [session.role.value],
                "metadata": {"user": session.username, "session_id": session.session_id}}

    def _route_text(self, plan: RoutePlan, session: ChatSession) -> str:
        p = self.engine.patient(plan.patient_key)
        who = p.label if p else "no specific patient"
        text = f"**Patient:** {who}" + (" (named in question)" if plan.patient_source == "question" else "")
        return text + (f"\n**Pipeline:** {plan.describe()}" if not plan.blocked else "")

    @staticmethod
    def _review_reasons(ext: ExtractionResult, an: AnalysisResult, risk: RiskAssessment, ins: AgentResult) -> List[str]:
        reasons = []
        conf = (ins.output or {}).get("confidence")
        if ins.error or not ins.parsed:
            reasons.append("AI output could not be fully validated")
        if isinstance(conf, (int, float)) and conf < REVIEW_CONFIDENCE:
            reasons.append(f"Low confidence ({conf:.0%})")
        if ext.quality.low_confidence:
            reasons.append("Low-confidence data extraction")
        if not ext.sufficient:
            reasons.append("Insufficient data")
        if an.inconsistencies:
            reasons.append("Clinical inconsistency")
        if risk.escalate:
            reasons.append(f"Risk {risk.risk_level} — escalated for clinician validation")
        return reasons

    def _extraction_text(self, ext: ExtractionResult, is_patient: bool) -> str:
        if ext.sections:
            what = "your" if is_patient else "the"
            text = f"Selected {len(ext.sections)} section(s) from {what} records: {', '.join(ext.sources[:5])}"
            if not ext.reference_only:
                text += f" · extraction confidence **{ext.quality.level}**"
        else:
            text = "No matching reports on file"
        if ext.new_symptoms:
            text += "\n- 📝 Saved to " + ("your record" if is_patient else "the record") + ": " + ", ".join(ext.new_symptoms)
        text += "".join(f"\n- ℹ️ {m}" for m in ext.missing[:3])
        text += "".join(f"\n- ⚠️ {i}" for i in ext.quality.issues[:3])
        if ext.denied:
            text += "\n- 🔒 Not permitted for your role: " + ", ".join(d.value.replace("_", " ") for d in ext.denied)
        return text

    @staticmethod
    def _extraction_brief(ext: ExtractionResult) -> str:
        lines = [f"Records used: {', '.join(ext.sources) or 'none'}", f"Extraction quality: {ext.quality.level} ({ext.quality.score:.2f})"]
        if ext.symptoms:
            lines.append("Patient-reported symptoms: " + ", ".join(ext.symptoms))
        lines += [f"Missing: {m}" for m in ext.missing] + [f"Quality issue: {i}" for i in ext.quality.issues[:4]]
        return "\n".join(lines)

    @staticmethod
    def _analysis_text(an: AnalysisResult, is_patient: bool, medications: Sequence[str] = ()) -> str:
        out = []
        if an.lab_results:
            line = f"**Lab Interpreter:** {len(an.lab_results)} values checked, {len(an.abnormal)} outside the normal range"
            if an.abnormal:
                line += " — " + ", ".join(f"{r.parameter} {r.value_text} ({r.status.value.lower()})" for r, _ in an.abnormal[:6])
            out.append(line)
        for f in an.imaging[:2]:
            line = f"**Radiology Analyzer:** {f.section.title}"
            if f.red_flags:
                line += " — 🚩 " + "; ".join(h[:90] for h in f.red_flags[:2])
            elif f.abnormal:
                line += " — " + "; ".join(h[:90] for h in f.abnormal[:2])
            else:
                line += " — no abnormal finding flagged"
            out.append(line)
        if an.drug_alerts:
            out.append("**Allergy Safety:** " + "; ".join(f"{a.level} — {a.drug} vs {a.against}" for a in an.drug_alerts))
        elif an.allergies:
            out.append("**Allergy Safety:** documented allergies " + ", ".join(an.allergies))
        if an.differential:
            prefix = "**Symptoms may fit (not a diagnosis):** " if is_patient else "**Symptom differential (KG):** "
            out.append(prefix + ", ".join(str(d["condition"]) for d in an.differential[:4]))
        if an.documented_conditions:
            out.append("**Record review:** documented conditions — " + ", ".join(an.documented_conditions[:5]))
        if medications:
            out.append("**Record review:** medicines on record — " + "; ".join(m.split(" - ")[0] for m in medications[:4]))
        out += [f"⚠️ **Clinical inconsistency:** {i}" for i in an.inconsistencies[:2]]
        return "\n".join(f"- {o}" for o in out) if out else "- Nothing to analyse yet (no reports or symptoms)"

    @staticmethod
    def _risk_text(risk: RiskAssessment, is_patient: bool) -> str:
        if is_patient:
            text = f"Health risk from your symptoms and reports: **{LEVEL_BADGE[risk.clinical_level]}**"
            if any(i.category == "medication" for i in risk.indicators):
                text += " · ⛔ medicine safety alert"
            if risk.escalate:
                text += " · flagged for a clinician to check"
            return text + "".join(f"\n- {i.indicator}" for i in risk.indicators[:5])
        text = f"Risk level **{LEVEL_BADGE[risk.risk_level]}** · urgency {risk.urgency.lower()}"
        if risk.escalate:
            text += " · **escalated for clinician validation**"
        text += "".join(f"\n- {i.indicator}" for i in risk.indicators[:5])
        return text

    @staticmethod
    def _risk_brief(risk: RiskAssessment) -> str:
        lines = [f"Risk level {risk.risk_level}, urgency {risk.urgency}, escalate={risk.escalate}"]
        return "\n".join(lines + [f"- [{i.level}] {i.indicator}" for i in risk.indicators[:10]])

    def safety_banner(self, plan: RoutePlan) -> str:
        p = self.engine.patient(plan.patient_key)
        lines = []
        alerts = self.engine.safety_alerts(plan.patient_key, plan.drugs)
        for a in alerts:
            icon = "⛔" if a.level == "CONTRAINDICATED" else "⚠️"
            lines.append(f"{icon} **{a.level}** — {alert_line(a).split('] ', 1)[1]}")
        if plan.drugs and not alerts:
            lines.append(f"✅ No allergy or interaction conflict found for **{', '.join(plan.drugs)}**"
                         + ("" if p else " (select a patient to check their allergies)"))
        if p and p.allergies and (plan.drugs or plan.allergens or Domain.ALLERGY in plan.focus):
            lines.append("🩺 **Documented allergies:** " + "; ".join(
                f"{a.allergen} ({a.severity.lower()}{', ' + ', '.join(a.reactions[:2]) if a.reactions else ''})" for a in p.allergies))
        return "\n".join(lines)

    def _finish(self, session: ChatSession, turn: Turn, plan: RoutePlan, run: _Run, started: float) -> None:
        turn.agents = [a.value for a in plan.agents]
        session.turns.append(turn)
        if plan.patient_key:
            session.last_patient = plan.patient_key
        self.auditor.record({
            "session": session.session_id, "user": session.username, "role": session.role.value,
            "patient": plan.patient_key, "question": turn.question, "mode": plan.mode, "agents": turn.agents,
            "denied_domains": [d.value for d in plan.denied_domains], "blocked": plan.blocked,
            "drugs": plan.drugs, "symptoms": plan.symptoms, "safety_shown": bool(turn.safety),
            "risk_level": run.risk.risk_level if run.risk else None, "escalate": bool(run.risk and run.risk.escalate),
            "review_reasons": run.review_reasons, "case_id": turn.case_id,
            "llm_calls": sum(1 for r in run.llm if not r.cache_hit and not r.error),
            "cache_hits": sum(1 for r in run.llm if r.cache_hit),
            "errors": [r.error for r in run.llm if r.error],
            "latency_ms": int((time.perf_counter() - started) * 1000),
        })


# ---------------------------------------------------------------------- rendering
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _timing(res: AgentResult) -> str:
    if res.error:
        return res.error[:160]
    return "cached answer" if res.cache_hit else f"{res.latency_ms / 1000:.1f}s"


BLOCKING_GUARD = ("doses", "definitive diagnosis", "changing medication", "Over-reassuring")
RISK_BADGE = {"CRITICAL": "🔴 CRITICAL", "HIGH": "🟠 HIGH", "MODERATE": "🟡 MODERATE", "LOW": "🟢 LOW"}


def patient_assessment_markdown(insight: Dict[str, Any], risk_level: str, draft: str, issues: List[str]) -> str:
    """What the patient sees straight away: the AI's predicted health picture + plain-language explanation.
    Clearly labelled as an AI assessment awaiting clinician confirmation; unsafe wording is never shown."""
    conditions = [str(c) for c in insight.get("possible_conditions") or [] if str(c).strip()][:5]
    try:
        conf = float(insight.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    md = f"**Overall risk:** {RISK_BADGE.get(risk_level, risk_level)} · **AI confidence:** {conf:.0%}"
    if conditions:
        md += ("\n\n**What your reports and symptoms may point to** _(AI prediction — not a diagnosis)_:\n"
               + "\n".join(f"- {c}" for c in conditions))
    blocking = [i for i in issues if any(k in i for k in BLOCKING_GUARD)]
    if blocking:
        md += ("\n\n_The detailed explanation needs a clinician's check before I can show it (it contained wording "
               "that must be confirmed by a doctor). Please discuss these findings with your doctor._")
    elif draft.strip():
        md += "\n\n---\n" + draft
        if not any(w in draft.lower() for w in ("doctor", "clinician")):
            md += "\n\nPlease discuss these results with your doctor."
    md += "\n\n<sub>🤖 AI assessment from your records — not a diagnosis. A clinician will confirm or correct it.</sub>"
    return md


def _plan_dict(plan: RoutePlan) -> Dict[str, Any]:
    return {"mode": plan.mode, "patient_key": plan.patient_key, "agents": [a.value for a in plan.agents],
            "denied_domains": [d.value for d in plan.denied_domains], "symptoms": plan.symptoms, "drugs": plan.drugs}


def _bullets(title: str, items) -> str:
    items = [i for i in (items or []) if str(i).strip()]
    return f"\n\n**{title}**\n" + "\n".join(f"- {i}" for i in items) if items else ""


def render_insight(res: AgentResult, review_reasons: List[str], sources: List[str]) -> str:
    if res.error:
        return f"_The Insight agent could not answer right now._ {res.error}"
    o = res.output or {}
    md = ""
    if review_reasons:
        md += "> 🟠 **Review Required** — " + "; ".join(review_reasons) + "\n\n"
    md += str(o.get("answer", "")).strip()
    md += (_bullets("Key findings", o.get("key_findings")) + _bullets("Possible conditions (to confirm)", o.get("possible_conditions"))
           + _bullets("Recommended next steps", o.get("next_steps")) + _bullets("Additional data needed", o.get("additional_data_needed")))
    conf = o.get("confidence")
    if isinstance(conf, (int, float)):
        md += f"\n\n**Confidence:** {conf:.0%}"
    srcs = list(dict.fromkeys([*map(str, o.get("sources") or []), *sources]))
    if srcs:
        md += f"\n\n<sub>Sources: {', '.join(srcs[:8])}</sub>"
    return md
