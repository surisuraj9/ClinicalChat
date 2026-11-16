"""Report upload: read every attached file in parallel (text, tables, OCR for scans), check each one (name on the
report, duplicates, extraction quality), save the accepted ones, then run the agent pipeline ONCE over all of them —
answering the patient's message if there is one (same AI cost as a single report)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from clinicalchat.domain.clock import utc_minute
from clinicalchat.domain.conversation import ChatSession, Event, Turn
from clinicalchat.domain.models import ClinicalDocument, Domain, PatientProfile
from clinicalchat.domain.roles import AGENT_LABELS, AGENT_ORDER, AgentName, Role, RolePolicy, policy
from clinicalchat.infrastructure.repositories.uploads import UploadRepository
from clinicalchat.infrastructure.tracing import activate
from clinicalchat.ingestion.quality import assess_document
from clinicalchat.ingestion.uploads import MAX_UPLOAD_FILES, foreign_name, parse_upload
from clinicalchat.services.chat import SymptomRecorder, routed_event
from clinicalchat.services.knowledge_base import KnowledgeBase
from clinicalchat.services.pipeline import AnalysisPipeline
from clinicalchat.services.responders import PatientResponder, StaffResponder
from clinicalchat.services.router import RoutePlan, Router
from clinicalchat.services.turn import TurnContext
from clinicalchat.services.turn_recorder import TurnRecorder

LABEL = AGENT_LABELS[AgentName.EXTRACTION]
OTHER_PATIENT = ("Your message is about a different patient than the one these reports are being added to. Select that "
                 "patient first, then attach the reports.")
Files = Sequence[Tuple[Path, str]]  # (path on disk, name shown to the user)


class UploadService:
    def __init__(self, kb: KnowledgeBase, uploads: UploadRepository, router: Router, pipeline: AnalysisPipeline,
                 staff: StaffResponder, patient: PatientResponder, symptoms: SymptomRecorder, recorder: TurnRecorder):
        self.kb = kb
        self.uploads = uploads
        self.router = router
        self.pipeline = pipeline
        self.staff = staff
        self.patient = patient
        self.symptoms = symptoms
        self.recorder = recorder

    def upload(self, path: Path, session: ChatSession, display_name: Optional[str] = None,
               report_type: str = "auto") -> Iterator[Event]:
        """Single-file upload (CLI / tests) — see ``upload_many``."""
        return self.upload_many([(Path(path), display_name or Path(path).name)], session, report_type=report_type)

    def upload_many(self, files: Files, session: ChatSession, message: str = "",
                    report_type: str = "auto") -> Iterator[Event]:
        message = (message or "").strip()
        pol = policy(session.role)
        key = session.own_patient if pol.own_record_only else session.selected_patient
        question = message or f"[upload] {', '.join(n for _, n in files)}"
        ctx = TurnContext(message, session, _upload_plan(key, pol, session.role),
                          Turn(question=question, role=session.role.value, patient_key=key),
                          trace=self.recorder.start(question, session, name="report-upload"))
        try:
            refusal = _refusal(pol, key, files)
            if refusal:
                yield from self._block(ctx, refusal, "🔒 Upload" if not pol.can_upload or not key else "📄 Upload")
            else:
                allowed = (yield from self._route_message(ctx)) if message else True
                if allowed:
                    yield from self._process(ctx, files, report_type)
        finally:
            self.recorder.finish(ctx)

    # ------------------------------------------------------------------ steps
    def _route_message(self, ctx: TurnContext) -> Iterator[Event]:
        """The message decides symptoms / drugs / allergens to check; the record stays the session's patient."""
        yield Event("status", "🧭 Understanding your message", "Detecting symptoms, medicines and topic…", "pending", "route")
        plan, session = ctx.plan, ctx.session
        routed = self.router.route(ctx.question, session.role, session_patient=plan.patient_key,
                                   own_patient=session.own_patient)
        if routed.blocked or (routed.patient_key and routed.patient_key != plan.patient_key):
            yield Event("status", "🧭 Routed", "Blocked", "done", "route")
            yield from self._block(ctx, routed.blocked or OTHER_PATIENT, "🔒 Access")
            return False
        plan.symptoms, plan.drugs, plan.allergens, plan.notes = routed.symptoms, routed.drugs, routed.allergens, routed.notes
        yield routed_event(plan, self.kb.patients)
        return True

    def _process(self, ctx: TurnContext, files: Files, report_type: str) -> Iterator[Event]:
        key, session = ctx.patient_key, ctx.session
        patient = self.kb.patients.get(key)
        results = yield from self._read_all(files, ctx, patient, report_type)
        with activate(ctx.trace.child("index-reports", input={"files": [n for _, n in files]})) as span:
            accepted, lines = self._accept(files, results, patient, session.username)
            span.end({"accepted": [n for n, _ in accepted]})
        total = len(files)
        head = (f"**{len(accepted)} of {total}** report(s) added to the record." if total > 1 else
                ("Report added to the record." if accepted else "The report was not added."))
        yield Event("step", f"{LABEL} — reports checked", head + "\n" + "\n".join(lines), data={"agent": "extraction"})
        if not accepted:
            text = "None of the uploaded files could be added, so there is nothing new to analyse."
            yield Event("status", f"{LABEL} — nothing added", text, "done", "agent-extraction")
            yield from self._block(ctx, text, "📄 Upload")
            return
        ctx.plan.focus = list(dict.fromkeys(d.domain for _, d in accepted if d.domain != Domain.REFERENCE))
        if ctx.question and ctx.is_patient and ctx.plan.symptoms:
            self.symptoms.save(key, ctx.plan.symptoms)
        ctx.question = ctx.turn.question = _upload_question(ctx, [n for n, _ in accepted], patient)
        doc_ids = [d.doc_id for _, d in accepted]
        if ctx.is_patient:
            yield from self.pipeline.run(ctx, upload_doc_ids=doc_ids)
            yield from self.patient.respond(ctx, doc_ids)
        else:
            yield from self.pipeline.run(ctx, upload_doc_ids=doc_ids, on_analysed=self.staff.prefetch)
            yield from self.staff.respond(ctx)

    def _read_all(self, files: Files, ctx: TurnContext, patient: PatientProfile,
                  report_type: str) -> Iterator[Event]:
        """Read / OCR all files at the same time; returns {index: (document or None, error or None)}."""
        total = len(files)
        yield Event("status", f"{LABEL} — reading {total} file{'s' if total > 1 else ''}"
                    + (" in parallel" if total > 1 else ""), "Extracting text (OCR for scans), sections and lab values…",
                    "pending", "agent-extraction")

        def read(item: Tuple[Path, str]):
            path, name = item
            span = ctx.trace.child("parse-report", as_type="tool", input={"file": name, "report_type": report_type})
            try:
                doc = parse_upload(Path(path), patient_key=patient.patient_key, patient_name=patient.name,
                                   report_type=report_type, uploaded_by=ctx.session.username, uploaded_at=utc_minute(),
                                   display_name=name, cache_dir=self.kb.settings.work_dir / "ocr_cache")
            except Exception as exc:  # one unreadable file must not stop the others
                span.end(error=f"{type(exc).__name__}: {exc}")
                return None, exc
            span.end({"category": doc.category, "domain": doc.domain.value, "sections": len(doc.sections),
                      "extraction": doc.extraction_method,
                      "lab_values": sum(len(s.lab_results) for s in doc.sections)})
            return doc, None

        results: Dict[int, tuple] = {}
        with ThreadPoolExecutor(max_workers=min(4, total), thread_name_prefix="upload") as pool:
            futures = {pool.submit(read, f): i for i, f in enumerate(files)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                if total > 1:
                    yield Event("status", f"{LABEL} — read {len(results)}/{total}", f"{files[futures[fut]][1]} done",
                                "pending", "agent-extraction")
        return results

    def _accept(self, files: Files, results: Dict[int, tuple], patient: PatientProfile,
                username: str) -> Tuple[List[Tuple[str, ClinicalDocument]], List[str]]:
        """Check each file in the order it was attached; save and index the good ones."""
        accepted: List[Tuple[str, ClinicalDocument]] = []
        lines: List[str] = []
        seen: set = set()
        for i, (_, name) in enumerate(files):
            doc, exc = results[i]
            problem = self._problem(doc, exc, seen, patient)
            if problem:
                lines.append(f"- {problem[0]} **{name}** — {problem[1]}")
                continue
            seen.add(doc.doc_id)
            quality = assess_document(doc, patient.name)
            self.uploads.save(doc.model_dump_json(), quality.model_dump(), patient.patient_key, doc.doc_id, name, username)
            self.kb.add_document(doc)
            accepted.append((name, doc))
            lines.append(_accepted_line(name, doc, quality))
        return accepted, lines

    def _problem(self, doc: Optional[ClinicalDocument], exc: Optional[Exception], seen: set,
                 patient: PatientProfile) -> Optional[Tuple[str, str]]:
        if exc is not None:
            return "❌", f"could not be read: {str(exc)[:160]}"
        if doc.doc_id in seen or doc.doc_id in self.kb.catalog:
            return "↩️", "already on your record, skipped"
        other = foreign_name(doc, patient.name)
        if other:
            return "🪪", (f"the name on the report is **{other}**, not **{patient.name}**. "
                         "Not added (ask the lab for a corrected copy, or a clinician can add it).")
        return None

    @staticmethod
    def _block(ctx: TurnContext, text: str, title: str) -> Iterator[Event]:
        ctx.turn.blocked = text
        yield Event("blocked", title, text)


def _upload_plan(key: Optional[str], pol: RolePolicy, role: Role) -> RoutePlan:
    agents = [a for a in AGENT_ORDER if a in pol.agents and (a != AgentName.COMMUNICATION or role == Role.PATIENT)]
    return RoutePlan(mode="case", patient_key=key, patient_source="session", check_allergies=True, agents=agents)


def _refusal(pol: RolePolicy, key: Optional[str], files: Files) -> str:
    if not pol.can_upload:
        return "Uploading reports is not available for your role."
    if not key:
        return "Select a patient first, then upload their report."
    if not files:
        return "No files were received."
    if len(files) > MAX_UPLOAD_FILES:
        return (f"You selected **{len(files)}** files. Please upload at most **{MAX_UPLOAD_FILES}** reports at a "
                "time so each one can be checked properly.")
    return ""


def _accepted_line(name: str, doc: ClinicalDocument, quality: Any) -> str:
    labs = sum(len(s.lab_results) for s in doc.sections)
    line = (f"- ✅ **{name}** — {doc.category.replace('_', ' ')} report ({doc.extraction_method}), "
            f"{labs} lab value(s), extraction confidence **{quality.level}** ({quality.score:.0%})")
    line += "".join(f"\n  - ⚠️ {issue}" for issue in quality.issues[:3])
    if quality.low_confidence:
        line += "\n  - 🚩 Low-confidence extraction — flagged for clinician review"
    return line


def _upload_question(ctx: TurnContext, names: List[str], patient: PatientProfile) -> str:
    listed = ", ".join(f"'{n}'" for n in names)
    word = "report" if len(names) == 1 else "reports"
    if ctx.question:
        return f"{ctx.question}\n(Attached {word}: {listed})"
    if ctx.is_patient:
        return f"Please review my newly uploaded {word} {listed}."
    return f"Review the newly uploaded {word} {listed} for {patient.name}."
