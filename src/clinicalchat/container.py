"""Composition root: the ONE place that creates every object and wires them together (dependency injection).
Nothing else constructs infrastructure; services receive what they need through their constructor, so tests can
swap any part (e.g. a fake LLM or web search) by passing it to ``build_container``."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from clinicalchat.agents.analysis import ClinicalAnalysisAgent
from clinicalchat.agents.communication import PatientCommunicationAgent
from clinicalchat.agents.extraction import DataExtractionAgent
from clinicalchat.agents.insight import InsightGenerationAgent
from clinicalchat.agents.risk import RiskDetectionAgent
from clinicalchat.config import Settings, get_settings
from clinicalchat.infrastructure.audit_log import AuditLog
from clinicalchat.infrastructure.database import Database
from clinicalchat.infrastructure.llm.models import FAST, INSIGHT, SPECIALIST, create_chat_model, install_cache
from clinicalchat.infrastructure.repositories.caches import FaqCache, InsightCache, WebCache
from clinicalchat.infrastructure.repositories.cases import CaseRepository
from clinicalchat.infrastructure.repositories.feedback import FeedbackRepository
from clinicalchat.infrastructure.repositories.patient_facts import PatientFactRepository
from clinicalchat.infrastructure.repositories.uploads import UploadRepository
from clinicalchat.infrastructure.repositories.users import UserRepository
from clinicalchat.infrastructure.tracing import Tracer, activate
from clinicalchat.infrastructure.web_search import TavilySearch
from clinicalchat.services.accounts import AccountService
from clinicalchat.services.admin import PatientDataAdmin
from clinicalchat.services.answer_writer import AnswerWriter
from clinicalchat.services.care_team import CareTeamService
from clinicalchat.services.cases import CaseFiler
from clinicalchat.services.chat import ChatService, SymptomRecorder
from clinicalchat.services.communicator import Communicator
from clinicalchat.services.export import SessionExporter
from clinicalchat.services.insight_service import InsightService
from clinicalchat.services.knowledge_base import KnowledgeBase
from clinicalchat.services.pipeline import AnalysisPipeline
from clinicalchat.services.responders import GeneralResponder, PatientResponder, StaffResponder
from clinicalchat.services.router import Router
from clinicalchat.services.specialist_panel import SpecialistPanel
from clinicalchat.services.turn_recorder import TurnRecorder
from clinicalchat.services.uploads import UploadService
from clinicalchat.services.web_enrichment import WebEnrichment


@dataclass
class Repositories:
    users: UserRepository
    facts: PatientFactRepository
    uploads: UploadRepository
    cases: CaseRepository
    feedback: FeedbackRepository
    insights: InsightCache
    faq: FaqCache
    web: WebCache

    @classmethod
    def on(cls, db: Database) -> "Repositories":
        return cls(UserRepository(db), PatientFactRepository(db), UploadRepository(db), CaseRepository(db),
                   FeedbackRepository(db), InsightCache(db), FaqCache(db), WebCache(db))


@dataclass
class Container:
    """What the web app, the CLI and the tests use."""
    settings: Settings
    kb: KnowledgeBase
    repos: Repositories
    audit: AuditLog
    tracer: Tracer
    llm: BaseChatModel
    fast_llm: BaseChatModel
    web: TavilySearch
    chat: ChatService
    uploads: UploadService
    care_team: CareTeamService
    insight: InsightService
    pipeline: AnalysisPipeline
    specialists: SpecialistPanel
    accounts: AccountService
    exporter: SessionExporter
    admin: PatientDataAdmin


def build_container(settings: Optional[Settings] = None, *, kb: Optional[KnowledgeBase] = None,
                    llm: Optional[BaseChatModel] = None, web: Optional[TavilySearch] = None,
                    db_path: Optional[Path] = None, persist: bool = True) -> Container:
    settings = settings or get_settings()
    settings.ensure_dirs()
    repos = Repositories.on(Database(db_path or settings.database_path))
    repos.cases.consolidate_open_cases()
    tracer = Tracer(settings)
    if kb is None:
        with activate(tracer.trace("knowledge-base-load", tags=["startup"])) as span:
            kb = KnowledgeBase.load(settings, persist=persist)
            kb.load_user_data(repos.users.all(), repos.facts.all(), repos.uploads.all())
            span.end(kb.stats())
    if llm is None:
        install_cache(settings)
    insight_llm = llm or create_chat_model(settings, INSIGHT)
    fast_llm = llm or create_chat_model(settings, FAST)
    specialist_llm = llm or create_chat_model(settings, SPECIALIST)
    audit = AuditLog(settings.audit_dir)
    web = web or TavilySearch(settings, repos.web)

    # agents
    extraction = DataExtractionAgent(kb.catalog, kb.patients, kb.kg, kb.records)
    analysis = ClinicalAnalysisAgent(kb.catalog, kb.patients, kb.kg, kb.safety)
    insight_agent = InsightGenerationAgent(insight_llm)
    communication_agent = PatientCommunicationAgent(fast_llm)
    specialists = SpecialistPanel(specialist_llm, enabled=settings.specialists_use_llm)

    # services
    router = Router(kb.patients, kb.kg)
    pipeline = AnalysisPipeline(extraction, analysis, RiskDetectionAgent(), kb.patients, kb.safety, specialists)
    enrichment = WebEnrichment(web, kb.patients, kb.kg)
    insight = InsightService(insight_agent, kb.patients, kb.records, repos.feedback, repos.cases, repos.insights,
                             specialists)
    writer = AnswerWriter(communication_agent)
    filer = CaseFiler(repos.cases, repos.uploads, repos.facts, kb.catalog, kb.safety)
    communicator = Communicator(writer, filer, kb.patients)
    staff = StaffResponder(insight, enrichment, communicator)
    patient = PatientResponder(insight, filer, communicator)
    general = GeneralResponder(writer, enrichment, kb.records, kb.patients, repos.faq)
    symptoms = SymptomRecorder(repos.facts, kb.patients, kb.kg)
    recorder = TurnRecorder(audit, tracer)
    exporter = SessionExporter(kb.patients, settings.export_dir)
    return Container(
        settings=settings, kb=kb, repos=repos, audit=audit, tracer=tracer, llm=insight_llm, fast_llm=fast_llm, web=web,
        chat=ChatService(router, kb.patients, enrichment, pipeline, staff, patient, general, communicator, symptoms,
                         repos.cases, recorder),
        uploads=UploadService(kb, repos.uploads, router, pipeline, staff, patient, symptoms, recorder),
        care_team=CareTeamService(repos.cases, repos.feedback, audit, tracer), insight=insight, pipeline=pipeline,
        specialists=specialists,
        accounts=AccountService(repos.users, repos.facts, kb.patients, settings.demo_password),
        exporter=exporter,
        admin=PatientDataAdmin(repos.users, repos.uploads, repos.facts, repos.cases, repos.feedback, repos.insights,
                               kb, exporter))


_CONTAINER: Optional[Container] = None
_LOCK = threading.Lock()


def get_container() -> Container:
    """The app-wide container, built on first use (loads the dataset and the vector index)."""
    global _CONTAINER
    with _LOCK:
        if _CONTAINER is None:
            _CONTAINER = build_container()
        return _CONTAINER
