import json
import os
import re
from typing import List

# tests never use the real keys in .env: no OpenAI or Tavily spend, no test traces in Langfuse
for _var in ("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TAVILY_API_KEY"):
    os.environ[_var] = ""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from clinicalchat.config import Settings, get_settings
from clinicalchat.container import build_container
from clinicalchat.services.knowledge_base import KnowledgeBase


SPECIALIST_PAYLOADS = {  # course schemas; confidence 85 checks the lenient "percent -> fraction" parsing
    "LAB INTERPRETER": {"interpretations": [{
        "test_name": "Hemoglobin", "result_value": "9.1 g/dL", "status": "LOW", "clinical_significance": "Anaemia",
        "potential_causes": ["iron deficiency"], "recommended_actions": ["iron studies"], "urgency": "stat",
        "confidence": 85}]},
    "RADIOLOGY ANALYZER": {"studies": [{
        "study_type": "X-ray", "body_region": "Chest", "findings": ["clear lungs"], "impression": "No acute disease",
        "differential_diagnoses": [], "recommendations": ["none"], "urgency": "routine", "confidence": 0.9}]},
    "ALLERGY SAFETY": {"assessments": [{
        "allergen": "Penicillin", "category": "Medication", "severity": "SEVERE", "reactions": ["hives"],
        "cross_reactivities": ["amoxicillin"], "safe_alternatives": ["azithromycin"],
        "emergency_protocol": "Epinephrine and call emergency services", "confidence": 0.9}]},
}


class FakeLLM(BaseChatModel):
    """Returns schema-shaped JSON and records every prompt (so tests can count LLM calls)."""

    calls: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = "\n".join(str(m.content) for m in messages)
        self.calls.append(text)
        agent = re.search(r"You are the ([A-Z ]+) agent", text).group(1)
        if agent in SPECIALIST_PAYLOADS:
            payload = SPECIALIST_PAYLOADS[agent]
        elif agent.startswith("INSIGHT"):
            payload = {"answer": "INSIGHT answer", "key_findings": ["finding"], "possible_conditions": ["Anaemia"],
                       "next_steps": ["repeat test"], "additional_data_needed": [], "confidence": 0.8, "sources": []}
        else:
            payload = {"message": "COMMUNICATION answer. Please discuss this with your doctor.",
                       "what_it_means": ["meaning"], "next_steps": ["rest"], "seek_help_if": ["worse"],
                       "questions_for_doctor": ["what next?"]}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="```json\n" + json.dumps(payload) + "\n```"))])


class NoWeb:
    """Web search switched off (tests never call Tavily)."""
    enabled = False

    def search(self, query, kind, names=(), config=None):  # pragma: no cover - never called when disabled
        raise AssertionError("web search is off in tests")


@pytest.fixture(scope="session")
def kb():
    return KnowledgeBase.load(get_settings(), persist=False)


@pytest.fixture()
def fake_llm():
    return FakeLLM(cache=False, calls=[])


@pytest.fixture()
def make_container(kb, tmp_path):
    """Builds the app with a fake model, a fake web search and a throw-away database / audit folder."""
    def make(llm=None, web=None, **settings_kw):
        settings = Settings(work_dir=tmp_path, **settings_kw)
        return build_container(settings, kb=kb, llm=llm or FakeLLM(cache=False, calls=[]), web=web or NoWeb(),
                               db_path=tmp_path / "test.db", persist=False)
    return make


@pytest.fixture()
def container(make_container, fake_llm):
    return make_container(llm=fake_llm)


@pytest.fixture()
def repos(container):
    return container.repos
