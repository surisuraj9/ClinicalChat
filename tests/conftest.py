import json
import re
from typing import List

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from clinicalchat.audit import Auditor
from clinicalchat.config import Settings
from clinicalchat.engine import ClinicalEngine
from clinicalchat.orchestrator import Orchestrator


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
        payload = {"answer": f"{agent} answer", "key_findings": ["finding"], "recommendations": ["repeat test"],
                   "urgency": "ROUTINE", "sources": []}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="```json\n" + json.dumps(payload) + "\n```"))])


@pytest.fixture(scope="session")
def engine():
    return ClinicalEngine()


@pytest.fixture()
def fake_llm():
    return FakeLLM(cache=False, calls=[])


@pytest.fixture()
def orch(engine, fake_llm, tmp_path):
    return Orchestrator(engine, llm=fake_llm, auditor=Auditor(Settings(work_dir=tmp_path)))
