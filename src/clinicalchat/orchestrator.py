"""Orchestrator — the 5-agent pipeline from the course slide:

  📄 Data Extraction → 🔬 Clinical Analysis → 🚨 Risk Detection → 💡 Insight Generation → 💬 Patient Communication
     (deterministic)      (deterministic:        (deterministic       (LLM, 1 call)           (LLM, 1 call, patients)
                           Lab Interpreter,       rules)
                           Radiology Analyzer,
                           Allergy Safety)

Patient answers are shown at once as an AI assessment; the case then moves through the care team (workflow.py):
🩻 radiologist (only when there is imaging) → 🧭 care coordinator (priority, assignment) → 👩‍⚕️ clinician (confirms).

`ask()` / `upload()` are generators of `Event`s so the UI can show live progress. Deterministic safety information
(drug/allergy conflicts, emergency guidance) is shown immediately and is never held back.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeout
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
from clinicalchat.websearch import WebSearch, WebSearchOutcome
from clinicalchat.router import RoutePlan, Router
from clinicalchat import workflow as wf
from clinicalchat.store import Store, get_store

HISTORY_TURNS = 2
GENERIC_WORDS = set("""what which when where does should would could their there this that these those with from about
have patient patients condition conditions problem problems issue issues findings diagnosis current currently best
options option next steps step approach plan care give tell explain summary summarize please need needs into
right based results result report reports labs case history them they""".split())
WEB_WAIT_PATIENT = 2.5  # seconds a patient's general answer waits for web sources (then answers without them)
GUIDELINE_ASK = re.compile(
    r"\b(guidelines?|guidance|evidence|literature|research|stud(?:y|ies)|trials?|latest|recommend\w*|protocols?|"
    r"first[- ]line|treat\w*|manag\w*|therap\w*|dos(?:e|es|ing)|standard of care|criteria|work[- ]?up|screening)\b", re.I)
DRUG_CANDIDATE = re.compile(
    r"\b(?:take|taking|took|use|using|give|giving|prescribed|start(?:ed|ing)?|stop(?:ped|ping)?|tablets? of|syrup)\s+"
    r"(?:taking |using )?(?:a |an |the |some |one |two |this |my |his |her )?"
    r"([A-Za-z][A-Za-z0-9-]{2,}(?:[ -](?:\d{2,4}|forte|plus|ds|xr|sr|er|cr|duo|max|mr))?)", re.I)
DRUG_SAFE_ASK = re.compile(r"\b(?:is|are)\s+([A-Za-z][A-Za-z0-9-]{2,}(?:[ -]\d{2,4})?)\s+(?:safe|ok|okay|fine|allowed)\b", re.I)
NOT_DRUGS = set("""medicine medicines medication medications tablet tablets pill pills drug drugs capsule capsules it this that
them these those rest care food foods water alcohol exercise bath shower leave test tests report reports painkiller
painkillers antibiotic antibiotics something anything vaccine vaccines steps action time walk coffee tea milk sugar
juice supplement supplements vitamin vitamins iron calcium injection injections blood more less any dose doses course
diet sleep nap break off over care advice help them treatment surgery holiday day days week weeks it's while place
fever cold cough pain""".split())  # short history keeps prompts (and time-to-first-word) small
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
    new_symptoms: List[str] = field(default_factory=list)
    file_case: bool = False  # patient: open/update a care-team case (new evidence or an explicit review request)
    evidence_fp: str = ""  # Insight cache key
    case_fp: str = ""  # what a case covered (for the escalation safety net)
    web: List[WebSearchOutcome] = field(default_factory=list)


class Orchestrator:
    def __init__(self, engine: ClinicalEngine, llm: Optional[BaseChatModel] = None, auditor: Optional[Auditor] = None,
                 store: Optional[Store] = None):
        self.engine = engine
        self.router = Router(engine)
        self.llm = llm or get_chat_model(purpose="insight")
        self.auditor = auditor or Auditor(engine.settings)
        self.store = store or get_store()
        self.extraction = DataExtractionAgent(engine)
        self.analysis = ClinicalAnalysisAgent(engine)
        self.risk = RiskDetectionAgent(engine.kg)
        self.fast_llm = llm or get_chat_model(purpose="fast")
        self.insight = InsightGenerationAgent(self.llm)
        self.communication = PatientCommunicationAgent(self.fast_llm)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="insight")
        self.web = WebSearch(engine.settings, self.store)
        self._web_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="web")
        self._background: List[Future] = []

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

        if plan.mode != "status":
            yield from self._lookup_unknown_drugs(question, session, plan, run)
        if plan.mode == "status":
            md = self.case_status_markdown(plan.patient_key)
            turn.answers.append({"label": "📁 Case status", "markdown": md, "summary": md[:300]})
            yield Event("answer", "📁 Your case — where it is now", md, data={"agent": "status"})
        elif plan.mode == "general":
            yield from self._general(question, session, plan, turn, run)
        elif plan.mode == "explain" and has_insight:
            yield from self._explain_last(question, session, turn, run)
        else:
            if session.role == Role.PATIENT and plan.symptoms and plan.patient_key:
                run.new_symptoms = self._save_symptoms(plan.patient_key, plan.symptoms)
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
        def read(item: Tuple[Path, str]):
            path, name = item
            try:
                return parse_upload(Path(path), patient_key=key, patient_name=patient.name, report_type=report_type,
                                    uploaded_by=session.username, uploaded_at=_now(), display_name=name,
                                    cache_dir=self.engine.settings.work_dir / "ocr_cache"), None
            except Exception as exc:
                return None, exc

        # read / OCR all files at the same time; then check and add them in the order they were attached
        yield Event("status", f"{label} — reading {total} file{'s' if total > 1 else ''}"
                    + (" in parallel" if total > 1 else ""), "Extracting text (OCR for scans), sections and lab values…",
                    "pending", "agent-extraction")
        results: Dict[int, tuple] = {}
        with ThreadPoolExecutor(max_workers=min(4, total), thread_name_prefix="upload") as pool:
            futures = {pool.submit(read, f): i for i, f in enumerate(files)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                if total > 1:
                    yield Event("status", f"{label} — read {len(results)}/{total}", f"{files[futures[fut]][1]} done",
                                "pending", "agent-extraction")
        for i, (path, name) in enumerate(files):
            doc, exc = results[i]
            if exc is not None:
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
        if case["status"] != wf.CLINICIAN:
            raise ValueError(f"Case {case_id} is not waiting for a clinician")
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

    def radiology_read(self, case_id: str, reader: str, verdict: str, report: str) -> Dict[str, Any]:
        """🩻 Radiologist's read of the case's imaging; the case then goes to the care coordinator (or straight back to
        the assigned clinician if it was triaged already and nothing critical was found)."""
        case = self.store.case(case_id)
        if not case:
            raise ValueError(f"Unknown case {case_id}")
        if verdict not in wf.RADIOLOGY_VERDICTS:
            raise ValueError(f"Unknown verdict {verdict}")
        doc_ids = [i["doc_id"] for i in case["payload"].get("imaging", []) if i.get("doc_id")]
        case = self.store.radiology_read(case_id, reader=reader, verdict=verdict, report=report.strip(), doc_ids=doc_ids)
        self.auditor.record({"event": "radiology_read", "case": case_id, "reader": reader, "verdict": verdict,
                             "next": case["status"]})
        return case

    def triage(self, case_id: str, coordinator: str, priority: str, clinician: str, note: str = "") -> Dict[str, Any]:
        """🧭 Care coordinator sets the priority and assigns the clinician who will confirm the case."""
        from clinicalchat.personas import PERSONAS

        doc = PERSONAS.get(clinician)
        if not doc or doc.role != Role.CLINICIAN:
            raise ValueError(f"{clinician} is not a clinician")
        case = self.store.triage(case_id, coordinator=coordinator, priority=priority, clinician=doc.username,
                                 clinician_name=doc.display_name, note=note.strip())
        self.auditor.record({"event": "triage", "case": case_id, "coordinator": coordinator, "priority": priority,
                             "clinician": doc.username})
        return case

    def case_status_markdown(self, patient_key: Optional[str]) -> str:
        """Deterministic answer to 'what is happening with my case?' (no LLM call)."""
        cases = [c for c in self.store.cases(patient_key=patient_key) if c["status"] != wf.MERGED] if patient_key else []
        if not cases:
            return ("You don't have a case yet. Describe how you feel or attach your reports with **📎 Attach** — the AI "
                    "assessment opens a case for you and the care team takes it from there.")
        open_ = [c for c in cases if c["status"] in wf.OPEN_STATUSES]
        case = open_[0] if open_ else max(cases, key=lambda c: c["reviewed"] or c["created"])
        pay = case["payload"]
        tri = pay.get("triage") or {}
        label = wf.STAGE_LABEL.get(case["status"], (case["status"], ""))[0]
        if case["status"] == wf.CLINICIAN and tri.get("clinician_name"):
            label = f"⏳ With {tri['clinician_name']}"
        md = f"**Case {case['case_id']}** — {label}\n\n"
        md += " → ".join({"done": f"✅ {s}", "now": f"**▶ {s}**", "todo": f"◻️ {s}", "skip": f"~~{s}~~"}[st]
                         for s, st in wf.stepper(case)) + "\n\n"
        conds = [str(c) for c in pay.get("possible_conditions", [])][:4]
        if conds:
            md += (f"**Latest AI assessment:** {RISK_BADGE.get(case['risk_level'], case['risk_level'])} — may point to "
                   + ", ".join(conds) + " _(AI prediction — not a diagnosis)_\n\n")
        if pay.get("radiology"):
            urgent = any(r["verdict"] == wf.RADIOLOGY_VERDICTS[2] for r in pay["radiology"]["reads"])
            md += "🩻 **Radiologist:** your scans have been read" + (
                " — an urgent finding was flagged and the care team has been alerted." if urgent else ".") + "\n\n"
        if tri:
            md += (f"🧭 **Care coordinator:** priority {wf.PRIORITY_ICON.get(tri['priority'], '')} {tri['priority']}, "
                   f"assigned to **{tri['clinician_name']}**.\n\n")
        if case["status"] == wf.VALIDATED:
            md += f"✅ **Confirmed by {case.get('reviewer') or 'your clinician'}:**\n\n{case.get('final_message') or ''}\n\n"
        elif case["status"] == wf.RETURNED:
            md += (f"↩️ **{case.get('reviewer') or 'Your clinician'} asked:** {case.get('review_note') or 'more information'} "
                   "— reply here or attach a report; it goes into the same case.\n\n")
        else:
            md += "**Next:** " + wf.next_step_text(case["status"], tri.get("clinician_name")) + "\n\n"
        events = pay.get("timeline") or []
        if events:
            md += "<details><summary>Timeline</summary>\n\n" + "\n".join(
                f"- {e['at']} · **{e['actor']}** — {e['event']}" for e in events[-10:]) + "\n\n</details>"
        return md

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

        if is_patient:
            yield from self._patient_answer(question, session, plan, turn, run, upload_doc_ids)
            return

        # 4) 💡 Insight Generation (LLM) — staff
        config = self._config(session)
        label = AGENT_LABELS[AgentName.INSIGHT]
        yield Event("status", f"{label} — writing", "Consolidating the findings (AI)…", "pending", "agent-insight")
        web = None
        if GUIDELINE_ASK.search(question):
            # "guideline for her condition" names no condition -> add the top findings; "…for hyperkalemia" stays as is
            vague = not (set(re.findall(r"[a-z][a-z-]{3,}", GUIDELINE_ASK.sub(" ", question.lower()))) - GENERIC_WORDS)
            terms = " ".join(an.possible_conditions[:2]) if vague else ""
            web = yield from self._web("guideline", f"{question} {terms} clinical guideline", session, run)
        ins = self._run_insight(question, policy(role).audience, plan, ext, an, risk, session.history_text(), config, run,
                                web=web)
        out = ins.output or {}
        run.review_reasons = self._review_reasons(ext, an, risk, ins)
        yield Event("status", f"{label} — {'failed' if ins.error else 'done'}", _timing(ins), "done", "agent-insight")
        session.last_insight = {"patient_key": plan.patient_key, "question": question, "output": out,
                                "risk": risk.risk_level, "facts": an.facts_text()}
        md = render_insight(ins, run.review_reasons, ext.sources)
        if web and web.results and not ins.error:
            md += web.links_markdown("🌐 Guideline / evidence sources (web — verify before acting)")
        turn.answers.append({"label": label, "markdown": md, "summary": str(out.get("answer", ins.error))[:300]})
        yield Event("answer", label, md, data={"agent": "insight", "review": run.review_reasons})
        if plan.mode == "explain" and not ins.error:
            yield from self._communicate(question, session, plan, turn, run, out, risk.clinical_level,
                                         bool(an.abnormal or risk.escalate))

    def _patient_answer(self, question: str, session: ChatSession, plan: RoutePlan, turn: Turn, run: _Run,
                        upload_doc_ids: Sequence[str]) -> Iterator[Event]:
        """Patients wait for ONE AI call (Communication). The clinician-facing Insight is not on the patient's
        critical path: it is reused when nothing changed, or written in the background once a case is opened."""
        ext, an, risk, key = run.ext, run.analysis, run.risk, plan.patient_key
        label = AGENT_LABELS[AgentName.INSIGHT]
        cached = None
        if key:
            run.evidence_fp = self._evidence_fp(key, risk.risk_level)
            run.case_fp = self._evidence_fp(key, risk.risk_level, reads=False)
            run.file_case = self._needs_case(plan, run, upload_doc_ids, risk)
            if not (upload_doc_ids or run.new_symptoms or plan.drugs or plan.allergens):
                cached = self.store.insight(key, run.evidence_fp)
        run.review_reasons = self._review_reasons(ext, an, risk, AgentResult(agent=AgentName.INSIGHT, label=label,
                                                                             output=cached or {}))
        background = bool(key and run.file_case and cached is None)
        if cached is not None:
            yield Event("status", f"{label} — reused", "Nothing new since the last assessment — reusing it (no AI call)",
                        "done", "agent-insight")
        elif background:
            yield Event("status", f"{label} — for the care team", "The clinical summary for your care team is written "
                        "in the background — you don't wait for it", "done", "agent-insight")
        else:
            yield Event("status", f"{label} — not needed", "Your question is answered from the findings above "
                        "(no clinical summary needed)", "done", "agent-insight")
        basis = cached or self._rule_based_insight(question, run)
        session.last_insight = {"patient_key": key, "question": question, "output": basis, "risk": risk.risk_level,
                                "facts": an.facts_text()}
        yield from self._communicate(question, session, plan, turn, run, basis, risk.clinical_level,
                                     bool(an.abnormal or risk.escalate))
        if background and turn.case_id:
            self._insight_in_background(turn.case_id, question, plan, run, session.history_text(), self._config(session))

    def _rule_based_insight(self, question: str, run: _Run) -> Dict[str, Any]:
        """What the Communication agent writes from when no AI clinical summary is needed or ready yet:
        the deterministic findings plus a short excerpt of the relevant records."""
        an, ext = run.analysis, run.ext
        excerpt = self.engine.records_context(question, ext.sections[:3], ext.focus, reference_k=1) if ext else ""
        return {"key_findings": an.facts_text().splitlines()[:25] if an else [],
                "possible_conditions": an.possible_conditions[:6] if an else [],
                "records_excerpt": excerpt[:2500], "source": "rule-based findings (validated code)"}

    def _insight_in_background(self, case_id: str, question: str, plan: RoutePlan, run: _Run, history: str,
                               config: dict) -> None:
        self.store.mark_insight_pending(case_id)
        audience = policy(Role.CLINICIAN).audience

        def work() -> None:
            ins = self._run_insight(question, audience, plan, run.ext, run.analysis, run.risk, history, config, None)
            if ins.error or not ins.parsed:
                self.store.set_case_insight(case_id, None, error=ins.error or "AI output could not be validated")
                return
            self.store.save_insight(plan.patient_key, run.evidence_fp, ins.output or {})
            self.store.set_case_insight(case_id, ins.output or {})

        self._background.append(self._pool.submit(work))

    def drain(self, timeout: float = 600) -> None:
        """Wait for background work (Insight for the care team). Used by tests and on shutdown."""
        for f in list(self._background):
            f.result(timeout=timeout)
        self._background.clear()

    def _run_insight(self, question, audience, plan, ext, an, risk, history, config, run,
                     web: Optional[WebSearchOutcome] = None) -> AgentResult:
        context = self.engine.records_context(question, ext.sections, ext.focus)
        if web and web.results:
            context += ("\n\n## WEB EVIDENCE (trusted guideline sites; cite as [W1], [W2]… for guideline / treatment "
                        "statements; it never overrides this patient's records)\n" + web.context())
        ins = self.insight.run(question=question, audience=audience, patient=self.engine.patient_summary(plan.patient_key),
                               extraction=self._extraction_brief(ext), facts=an.facts_text(), risk=self._risk_brief(risk),
                               feedback=self._feedback(), context=context, history=history, config=config)
        if run is not None:
            run.llm.append(ins)
        return ins

    # ------------------------------------------------------------------ 🌐 web search (Tavily, trusted sites)
    def _patient_names(self) -> List[str]:
        return [p.name for p in self.engine.registry.patients.values()]

    def _web(self, kind: str, query: str, session: ChatSession, run: _Run,
             wait: Optional[float] = None) -> Iterator[Event]:
        """Search trusted sites; returns the outcome (or None when off / too slow for ``wait`` seconds)."""
        if not self.web.enabled:
            return None
        where = {"health": "WHO, NIH, CDC, NHS, Mayo Clinic…", "guideline": "NICE, NCBI/PubMed, KDIGO, AHA…",
                 "drug": "DailyMed, MedlinePlus, drugs.com…"}[kind]
        yield Event("status", "🌐 Web search — trusted sites", f"Searching {where} (no personal details are sent)",
                    "pending", "web")
        fut = self._web_pool.submit(self.web.search, query, kind, self._patient_names(), self._config(session))
        try:
            out = fut.result(timeout=wait)
        except FutureTimeout:  # answer now; the search finishes in the background and is cached for next time
            yield Event("status", "🌐 Web search — skipped", f"Took longer than {wait:.0f}s — answering without it",
                        "done", "web")
            return None
        run.web.append(out)
        if out.error:
            text = f"Unavailable ({out.error[:80]})"
        else:
            sites = ", ".join(dict.fromkeys(r.domain for r in out.results)) or "nothing relevant"
            text = f"{len(out.results)} result(s): {sites}" + (" (cached)" if out.cached else f" ({out.latency_ms} ms)")
        yield Event("status", "🌐 Web search — done", text, "done", "web")
        return out

    def _drug_candidates(self, question: str, plan: RoutePlan) -> List[str]:
        kg = self.engine.kg
        found = []
        for m in [*DRUG_CANDIDATE.finditer(question), *DRUG_SAFE_ASK.finditer(question)]:
            cand = m.group(1).strip(" -")
            head = cand.split()[0].lower()
            if (head in NOT_DRUGS or kg.resolve_drugs(cand) or kg.resolve_allergen(cand) or kg.symptoms_in_text(cand)
                    or head in {d.lower() for d in plan.drugs}):
                continue
            found.append(cand)
        return list(dict.fromkeys(found))[:2]

    def _lookup_unknown_drugs(self, question: str, session: ChatSession, plan: RoutePlan, run: _Run) -> Iterator[Event]:
        """A medicine the knowledge graph does not know (often a brand, e.g. "Dolo 650", "Augmentin"): look up what it
        contains on trusted drug sites, then run the normal deterministic allergy / interaction checks on those
        ingredients. The web only *adds* checks — it can never clear a warning."""
        if not self.web.enabled:
            return
        for cand in self._drug_candidates(question, plan):
            out = yield from self._web("drug", f"{cand} medicine active ingredient generic name", session, run)
            if out is None:
                continue
            head = cand.split()[0].lower()
            matching = [r for r in out.results if head in f"{r.title} {r.content}".lower()][:3]
            ingredients = list(dict.fromkeys(d for r in matching for d in self.engine.kg.resolve_drugs(f"{r.title} {r.content}")))[:3]
            if ingredients:
                src = matching[0]
                plan.drugs = list(dict.fromkeys([*plan.drugs, *ingredients]))
                plan.check_allergies = True
                if Domain.ALLERGY in policy(session.role).domains and Domain.ALLERGY not in plan.focus:
                    plan.focus.append(Domain.ALLERGY)
                plan.drug_notes.append(f"🌐 **{cand}** — according to [{src.domain}]({src.url}) it contains "
                                       f"**{', '.join(ingredients)}**; checked below. _Web lookup — confirm with a pharmacist._")
                plan.notes.append(f"Web lookup: {cand} → {', '.join(ingredients)}")
            elif matching:
                plan.drug_notes.append(f"🌐 **{cand}** — found on [{matching[0].domain}]({matching[0].url}), but its "
                                       "ingredients are not in our safety database, so it could **not** be checked against "
                                       "allergies. Please ask a pharmacist or your doctor before taking it.")

    def _write(self, title: str, session: ChatSession, abnormal: bool, **kwargs) -> Iterator[Event]:
        """Run the Communication agent, streaming the answer as it is written. Each partial is passed through the
        deterministic guard first; if the wording needs a check, streaming pauses and only the final checked
        answer is shown. Returns the AgentResult."""
        com, paused = None, False
        for kind, value in self.communication.stream(config=self._config(session), **kwargs):
            if kind == "done":
                com = value
            elif not paused:
                md = communication_markdown(value)
                if any(k in i for i in check_patient_text([md], abnormal) for k in BLOCKING_GUARD):
                    paused = True
                    md = "_Checking the wording before showing it…_"
                if md.strip():
                    yield Event("partial", title, md + ("" if paused else " ▍"), data={"agent": "communication"})
        return com

    def _communicate(self, question: str, session: ChatSession, plan: RoutePlan, turn: Turn, run: _Run,
                     insight: Dict[str, Any], risk_level: str, abnormal: bool) -> Iterator[Event]:
        is_patient = session.role == Role.PATIENT
        label = AGENT_LABELS[AgentName.COMMUNICATION]
        yield Event("status", f"{label} — writing", "Writing a plain-language explanation (AI)…", "pending", "agent-communication")
        com = yield from self._write("💬 Writing your answer…" if is_patient else f"{label} — writing…", session, abnormal,
                                     question=question, patient=self.engine.patient_summary(plan.patient_key),
                                     risk=f"{risk_level}: {PATIENT_GUIDANCE[risk_level]}",
                                     insight=json.dumps(insight, ensure_ascii=False)[:6000])
        run.llm.append(com)
        yield Event("status", f"{label} — {'failed' if com.error else 'done'}", _timing(com), "done", "agent-communication")
        if com.error:
            if is_patient and plan.patient_key:
                yield from self._file_case_without_ai(question, session, plan, turn, run, com.error, insight)
                return
            yield Event("answer", label, f"_Could not write the explanation right now._ {com.error}", data={"agent": "communication"})
            return
        draft = communication_markdown(com.output or {})
        issues = check_patient_text([draft], abnormal)
        if is_patient and plan.patient_key:
            level = run.risk.risk_level if run.risk else risk_level
            md = patient_assessment_markdown(insight, level, draft, issues, reviewed=run.file_case)
            turn.answers.append({"label": label, "markdown": md, "summary": str((com.output or {}).get("message", ""))[:300]})
            if run.file_case:
                yield from self._file_case(question, session, plan, turn, run, insight, level, draft, issues, md)
            else:
                yield from self._answer_doubt(plan, md)
        else:
            md = draft + (("\n\n**🛡️ Guard check before sending:**\n" + "\n".join(f"- ⚠️ {i}" for i in issues))
                          if issues else "\n\n<sub>🛡️ Guard check passed (no doses, no definitive diagnosis).</sub>")
            turn.answers.append({"label": label, "markdown": md, "summary": str((com.output or {}).get("message", ""))[:300]})
            yield Event("answer", f"{label} — patient-friendly draft", md, data={"agent": "communication"})

    def _file_case(self, question: str, session: ChatSession, plan: RoutePlan, turn: Turn, run: _Run,
                   insight: Dict[str, Any], level: str, draft: str, issues: List[str], md: str) -> Iterator[Event]:
        """Save the patient's case (reusing the open one) and route it to the care team."""
        fields = dict(question=question, risk_level=level, escalate=bool(run.risk and run.risk.escalate),
                      review_required=bool(run.review_reasons or issues),
                      payload={"insight": insight, "facts": run.analysis.facts_text() if run.analysis else "",
                               "risk": run.risk.model_dump() if run.risk else {}, "review_reasons": run.review_reasons,
                               "guard_issues": issues, "sources": run.ext.sources if run.ext else [],
                               "possible_conditions": insight.get("possible_conditions", []),
                               "evidence_fp": run.case_fp, **self._care_team_payload(plan.patient_key, run)},
                      patient_draft=draft)
        existing = self.store.open_case(plan.patient_key)
        unread = wf.unread_imaging(fields["payload"]["imaging"], self.store.read_imaging_docs(plan.patient_key))
        if existing:
            case_id = existing["case_id"]
            status = wf.status_after_patient_update(existing, unread)
            self.store.update_case(case_id, status=status, **fields)
        else:
            status = wf.initial_status(unread)
            case_id = self.store.create_case(patient_key=plan.patient_key, created_by=session.username, status=status,
                                             **fields)
        assigned = (self.store.case(case_id)["payload"].get("triage") or {}).get("clinician_name")
        turn.case_id = case_id
        yield Event("answer", "🩺 Your health picture — AI assessment", md, data={"agent": "communication",
                                                                                  "case_id": case_id})
        case_text = (f"📁 **Case {case_id}** — "
                     + ("your open case was **updated** with this question (no new case created). " if existing
                        else "created for you. ")
                     + f"**Now:** {f'⏳ With {assigned}' if status == wf.CLINICIAN and assigned else wf.STAGE_LABEL[status][0]}. "
                     + wf.next_step_text(status, assigned)
                     + " The confirmed answer will appear here and under **My case updates** — ask *“what is the "
                       "status of my case?”* any time. The guidance above applies right now.")
        yield Event("case", "📁 Your case with the care team", case_text,
                    data={"case_id": case_id, "updated": bool(existing), "status": status})

    def _answer_doubt(self, plan: RoutePlan, md: str) -> Iterator[Event]:
        """A question about records the care team already has: answer it, no case opened or changed."""
        open_ = self.store.open_case(plan.patient_key)
        if open_:
            tri = open_["payload"].get("triage") or {}
            md += (f"\n\n📁 Your case **{open_['case_id']}** is still with the care team "
                   f"({wf.STAGE_LABEL[open_['status']][0]}{' · ' + tri['clinician_name'] if tri.get('clinician_name') else ''})"
                   " — this question did not change it.")
        yield Event("answer", "🩺 Your health picture — AI answer", md, data={"agent": "communication", "case_id": ""})

    def _file_case_without_ai(self, question: str, session: ChatSession, plan: RoutePlan, turn: Turn, run: _Run,
                              error: str, insight: Optional[Dict[str, Any]] = None) -> Iterator[Event]:
        """The AI is unavailable (e.g. rate-limited): still file and route the case from the rule-based agents'
        findings, so reports — especially critical ones — always reach the care team."""
        an, level = run.analysis, (run.risk.risk_level if run.risk else "MODERATE")
        reason = "AI explanation unavailable (service busy) — rule-based findings only"
        if reason not in run.review_reasons:
            run.review_reasons.append(reason)
        conditions = list((insight or {}).get("possible_conditions") or (an.possible_conditions if an else []))[:5]
        findings = [r.line() for r, _ in (an.abnormal if an else [])][:8]
        for f in (an.imaging if an else [])[:4]:
            notes = f.red_flags[:2] or f.abnormal[:2]
            if notes:
                findings.append(f"{f.section.title or 'Imaging'}: " + "; ".join(notes))
        fallback = {**(insight or {}), "possible_conditions": conditions, "ai_unavailable": True,
                    "answer": (insight or {}).get("answer") or "AI unavailable — rule-based findings only.",
                    "confidence": (insight or {}).get("confidence", 0)}
        body = ""
        if findings:
            body += "\n\n**What the automatic checks found in your reports:**\n" + "\n".join(f"- {x}" for x in findings)
        if conditions:
            body += ("\n\n**These findings can be seen with** _(not a diagnosis)_:\n" + "\n".join(f"- {c}" for c in conditions))
        risk_line = f"**Overall risk:** {RISK_BADGE.get(level, level)}"
        draft = risk_line + body + "\n\nPlease discuss these results with your clinician."
        md = risk_line + " · _rule-based check (AI busy)_" + body
        if not run.file_case:
            md += ("\n\n_The AI assistant is busy right now, so this is the automatic check only. Ask again in a few "
                   "minutes for a full explanation._\n\n<sub>🤖 Automatic rule-based check — not a diagnosis.</sub>")
            turn.answers.append({"label": AGENT_LABELS[AgentName.COMMUNICATION], "markdown": md, "summary": reason})
            yield from self._answer_doubt(plan, md)
            return
        md += ("\n\n_The AI assistant is busy right now, so I couldn't write a full explanation. Your reports have still "
               "been sent to the care team, who will review them. Ask again in a few minutes for the full explanation — "
               "it will update this same case._")
        md += "\n\n<sub>🤖 Automatic rule-based check — not a diagnosis. A clinician will confirm or correct it.</sub>"
        turn.answers.append({"label": AGENT_LABELS[AgentName.COMMUNICATION], "markdown": md, "summary": reason})
        yield from self._file_case(question, session, plan, turn, run, fallback, level, draft, [], md)

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
        label = AGENT_LABELS[AgentName.COMMUNICATION]
        p = self.engine.patient(plan.patient_key)
        allergies = ", ".join(sorted(a.allergen for a in p.allergies)) if p and p.allergies else "none recorded"
        hit = self.store.faq_lookup(question, allergies)
        if hit:
            yield Event("status", f"{label} — instant", f"Answered before (“{hit['question'][:60]}”) — reused, no AI call",
                        "done", "agent-communication")
            md = hit["answer"]
            turn.answers.append({"label": label, "markdown": md, "summary": md[:300]})
            yield Event("answer", label, md, data={"agent": "communication", "faq_cache": True})
            return
        web = yield from self._web("health", question, session, run, wait=WEB_WAIT_PATIENT)
        context, sources = self.engine.reference_context(question)
        if web and web.results:
            context += ("\n\n## Trusted health websites (current, public information)\n" + web.context())
        yield Event("status", f"{label} — answering", "Looking up reference information and writing an answer (AI)…",
                    "pending", "agent-communication")
        com = yield from self._write("💬 Writing your answer…", session, False, question=question,
                                     patient=f"Documented allergies: {allergies}",
                                     risk="Not applicable (general question).", insight=context, general=True)
        run.llm.append(com)
        yield Event("status", f"{label} — {'failed' if com.error else 'done'}", _timing(com), "done", "agent-communication")
        if com.error:
            md = f"_The AI assistant is busy right now — please try again shortly._ ({com.error[:120]})"
        else:
            md = communication_markdown(com.output or {})
            md += "\n\n<sub>ℹ️ General health information, not personal medical advice."
            md += f" Sources: {', '.join(sources)}</sub>" if sources else "</sub>"
            if web and web.results:
                md += web.links_markdown("🌐 From trusted health sites")
            if com.parsed and not any(k in i for i in check_patient_text([md], False) for k in BLOCKING_GUARD):
                self.store.faq_save(question, allergies, md)
        turn.answers.append({"label": label, "markdown": md, "summary": md[:300]})
        yield Event("answer", label, md, data={"agent": "communication"})

    # ------------------------------------------------------------------ helpers
    def _care_team_payload(self, patient_key: str, run: _Run) -> Dict[str, Any]:
        """What the radiologist and the coordinator need: imaging (for the radiology read), allergies with a contrast
        safety check, and the report types in the case."""
        imaging = []
        for f in (run.analysis.imaging if run.analysis else []):
            imaging.append({"doc_id": f.section.doc_id, "title": f.section.title, "source": f.source,
                            "date": f.section.report_date, "impression": f.impression, "red_flags": f.red_flags,
                            "abnormal": f.abnormal, "text": f.section.text[:2500]})
        contrast = [alert_line(a) for a in self.engine.safety_alerts(patient_key, ["iodinated contrast"])] if imaging else []
        domains = {self.engine.store.domain_of(s) for s in (run.ext.sections if run.ext else [])}
        return {"imaging": imaging, "allergies": run.analysis.allergies if run.analysis else [],
                "contrast_alerts": contrast, "report_types": sorted(d.value for d in domains if d)}

    def _save_symptoms(self, key: str, symptoms: List[str]) -> List[str]:
        known = set(self.engine.kg.patient_symptoms(key))
        new = [s for s in dict.fromkeys(symptoms) if s not in known]
        for s in new:
            self.store.add_fact(key, "symptom", s)
            self.engine.apply_fact(key, "symptom", s)
        return new

    def _evidence_fp(self, key: str, risk_level: str, reads: bool = True) -> str:
        """Fingerprint of the patient's evidence (reports, facts, risk; + radiologist reads for the Insight cache).
        Unchanged fingerprint = nothing new for the Insight agent / nothing the care team hasn't already seen."""
        basis = {"uploads": sorted(u["doc_id"] for u in self.store.uploads(key)),
                 "facts": sorted(f"{f['kind']}:{f['value']}" for f in self.store.facts(key)), "risk": risk_level}
        if reads:
            basis["reads"] = sorted(self.store.read_imaging_docs(key))
        return hashlib.sha1(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:16]

    def _needs_case(self, plan: RoutePlan, run: _Run, upload_doc_ids: Sequence[str], risk: RiskAssessment) -> bool:
        """A case is for the care team: only when there is something new to review, the patient asks for a doctor,
        the clinician is waiting for the patient's reply, or escalating findings were never put in front of a
        clinician. Simple doubts about records the care team already has are just answered."""
        if upload_doc_ids or run.new_symptoms or plan.wants_review:
            return True
        open_ = self.store.open_case(plan.patient_key)
        if open_:
            return open_["status"] == wf.RETURNED
        if risk.escalate:  # safety net: escalating findings that no case has covered yet
            return not any(c["payload"].get("evidence_fp") == run.case_fp
                           for c in self.store.cases(patient_key=plan.patient_key))
        return False

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
        lines = list(plan.drug_notes)
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
            "web_searches": [w.audit() for w in run.web],
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


def patient_assessment_markdown(insight: Dict[str, Any], risk_level: str, draft: str, issues: List[str],
                                reviewed: bool = True) -> str:
    """What the patient sees straight away: the AI's predicted health picture + plain-language explanation.
    Clearly labelled as an AI assessment awaiting clinician confirmation; unsafe wording is never shown."""
    conditions = [str(c) for c in insight.get("possible_conditions") or [] if str(c).strip()][:5]
    conf = insight.get("confidence")
    md = f"**Overall risk:** {RISK_BADGE.get(risk_level, risk_level)}"
    if isinstance(conf, (int, float)) and not insight.get("ai_unavailable"):
        md += f" · **AI confidence:** {conf:.0%}"
    if conditions:
        who = "automatic checks" if insight.get("source") else "AI prediction"
        md += (f"\n\n**What your reports and symptoms may point to** _({who} — not a diagnosis)_:\n"
               + "\n".join(f"- {c}" for c in conditions))
    blocking = [i for i in issues if any(k in i for k in BLOCKING_GUARD)]
    if blocking:
        md += ("\n\n_The detailed explanation needs a clinician's check before I can show it (it contained wording "
               "that must be confirmed by a doctor). Please discuss these findings with your doctor._")
    elif draft.strip():
        md += "\n\n---\n" + draft
        if not any(w in draft.lower() for w in ("doctor", "clinician")):
            md += "\n\nPlease discuss these results with your doctor."
    if reviewed:
        md += "\n\n<sub>🤖 AI assessment from your records — not a diagnosis. A clinician will confirm or correct it.</sub>"
    else:
        md += ("\n\n<sub>🤖 AI answer from your records — not a diagnosis. No case was opened for this question; to have a "
               "doctor check it, say *“please ask my doctor to review this”*.</sub>")
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
