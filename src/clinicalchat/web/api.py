"""REST API (course Week 8): the same agents and access rules as the chat UI, for scripts and integrations.

  GET  /api/health              liveness (no login)
  GET  /api/status              knowledge base, models, embeddings, tracing (login)
  POST /api/query               ask a question as the logged-in persona (HTTP Basic auth, same accounts as the UI)
  GET  /api/allergy/{allergen}  knowledge-graph cross-reactivity card (login; no patient data)

Interactive docs: /docs. Identity always comes from the credentials, never from the request body, so a patient can
only ever query their own record and staff only see what their role allows.
"""

import time
from collections import OrderedDict
from typing import Annotated, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from clinicalchat.container import Container
from clinicalchat.domain.conversation import ChatSession, Event
from clinicalchat.domain.personas import Persona
from clinicalchat.domain.roles import Role
from clinicalchat.infrastructure.llm.models import describe_llm

MAX_SESSIONS = 200


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000, description="The question, as you would type it in the chat")
    patient: Optional[str] = Field(None, description="Staff only: patient key, e.g. ananya-rao-54 (patients: ignored)")
    session_id: Optional[str] = Field(None, description="Send back the session_id you received to ask follow-ups")


class QueryResponse(BaseModel):
    answer: str
    user: str
    role: str
    patient: Optional[str] = None
    agents: List[str] = []
    risk_level: Optional[str] = None
    safety: str = ""
    guidance: str = ""
    blocked: str = ""
    case_id: str = ""
    session_id: str
    trace_id: str = ""
    latency_ms: int


class AllergyCard(BaseModel):
    query: str
    allergen: Optional[str]
    category: str = ""
    cross_reactive_allergens: List[Dict] = []
    avoid_medicines: List[Dict] = []
    safe_alternatives: List[str] = []


def compose(events: List[Event]) -> Tuple[str, str, str, str, Optional[str]]:
    """(answer markdown, safety, guidance, blocked, risk level) from a turn's events."""
    answers = [f"### {e.title}\n{e.text}" for e in events if e.kind == "answer"]
    answers += [f"### {e.title}\n{e.text}" for e in events if e.kind == "case"]
    risk = next((e.data.get("level") for e in events if e.kind == "step" and e.data.get("agent") == "risk"), None)
    return "\n\n".join(answers), _texts(events, "safety"), _texts(events, "guidance"), _texts(events, "blocked"), risk


def _texts(events: List[Event], kind: str) -> str:
    return "\n\n".join(e.text for e in events if e.kind == kind)


class ApiSessions:
    """Chat memory per (user, session id) so API callers can ask follow-ups; least-recently-used are dropped."""

    def __init__(self, limit: int = MAX_SESSIONS):
        self._sessions: OrderedDict[Tuple[str, str], ChatSession] = OrderedDict()
        self.limit = limit

    def get(self, persona: Persona, session_id: Optional[str], patient: Optional[str]) -> ChatSession:
        key = (persona.username, session_id or "")
        session = self._sessions.get(key) if session_id else None
        if session is None:
            session = ChatSession(username=persona.username, role=persona.role, display_name=persona.display_name,
                                  own_patient=persona.patient_key, selected_patient=persona.patient_key)
            key = (persona.username, session.session_id)
        if persona.role != Role.PATIENT and patient:
            session.selected_patient = patient
        self._sessions[key] = session
        self._sessions.move_to_end(key)
        while len(self._sessions) > self.limit:
            self._sessions.popitem(last=False)
        return session


def api_router(container: Container) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["ClinicalChat API"])
    basic = HTTPBasic(description="Same username / password as the chat login")
    sessions = ApiSessions()

    def login(credentials: Annotated[HTTPBasicCredentials, Depends(basic)]) -> Persona:
        persona = container.accounts.get_persona(credentials.username)
        if not (persona and container.accounts.authenticate(credentials.username, credentials.password)):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password",
                                headers={"WWW-Authenticate": "Basic"})
        return persona

    LoggedIn = Annotated[Persona, Depends(login)]  # noqa: N806 (a type alias, FastAPI style)

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @router.get("/status")
    def status_(persona: LoggedIn) -> dict:
        s = container.settings
        return {"status": "ok", "user": persona.username, "role": persona.role.value,
                "knowledge_base": container.kb.stats({"patients": len(container.kb.patients.all())}),
                "llm": describe_llm(s), "specialist_agents": "llm" if container.specialists.enabled else "rules",
                "web_search": "tavily" if container.web.enabled else "off",
                "tracing": "langfuse" if container.tracer.enabled else "off"}

    @router.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest, persona: LoggedIn) -> QueryResponse:
        session = sessions.get(persona, body.session_id, body.patient)
        started = time.perf_counter()
        events = await run_in_threadpool(container.chat.run, body.query, session)
        answer, safety, guidance, blocked, risk = compose(events)
        turn = session.turns[-1] if session.turns else None
        return QueryResponse(answer=answer or blocked, user=persona.username, role=persona.role.value,
                             patient=turn.patient_key if turn else None, agents=turn.agents if turn else [],
                             risk_level=risk, safety=safety, guidance=guidance, blocked=blocked,
                             case_id=turn.case_id if turn else "", session_id=session.session_id,
                             trace_id=turn.trace_id if turn else "",
                             latency_ms=int((time.perf_counter() - started) * 1000))

    @router.get("/allergy/{allergen}", response_model=AllergyCard)
    def allergy(allergen: str, persona: LoggedIn) -> AllergyCard:
        return AllergyCard(**container.kb.kg.allergy_safety_card(allergen))

    return router
