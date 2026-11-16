import json
import os
import re
from typing import List

# tests never use the real keys in .env: no OpenAI spend, no test traces in Langfuse
for _var in ("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ[_var] = ""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from clinicalchat.audit import Auditor
from clinicalchat.config import Settings
from clinicalchat.engine import ClinicalEngine
from clinicalchat.orchestrator import Orchestrator
from clinicalchat.store import Store


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
        if agent.startswith("INSIGHT"):
            payload = {"answer": "INSIGHT answer", "key_findings": ["finding"], "possible_conditions": ["Anaemia"],
                       "next_steps": ["repeat test"], "additional_data_needed": [], "confidence": 0.8, "sources": []}
        else:
            payload = {"message": "COMMUNICATION answer. Please discuss this with your doctor.",
                       "what_it_means": ["meaning"], "next_steps": ["rest"], "seek_help_if": ["worse"],
                       "questions_for_doctor": ["what next?"]}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="```json\n" + json.dumps(payload) + "\n```"))])


@pytest.fixture(scope="session")
def engine():
    return ClinicalEngine(persist=False)


@pytest.fixture()
def fake_llm():
    return FakeLLM(cache=False, calls=[])


@pytest.fixture()
def store(tmp_path):
    return Store(path=tmp_path / "test.db")


@pytest.fixture()
def orch(engine, fake_llm, store, tmp_path):
    return Orchestrator(engine, llm=fake_llm, auditor=Auditor(Settings(work_dir=tmp_path)), store=store)
