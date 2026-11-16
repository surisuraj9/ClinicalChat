"""Streams the 💬 Patient Communication agent's answer into the chat, passing every partial answer through the
deterministic guard first. If the wording needs a check, streaming pauses and only the final checked text is shown."""

from __future__ import annotations

from typing import Iterator

from clinicalchat.agents.communication import PatientCommunicationAgent, communication_markdown
from clinicalchat.agents.guard import blocking, check_patient_text
from clinicalchat.domain.conversation import ChatSession, Event
from clinicalchat.infrastructure.tracing import Tracer

PAUSED = "_Checking the wording before showing it…_"
CURSOR = " ▍"


class AnswerWriter:
    def __init__(self, agent: PatientCommunicationAgent, tracer: Tracer):
        self.agent = agent
        self.tracer = tracer

    def write(self, title: str, session: ChatSession, abnormal: bool, **inputs) -> Iterator[Event]:
        """Yields "partial" events while the AI writes; returns the final AgentResult."""
        result, paused = None, False
        for kind, value in self.agent.stream(config=self.tracer.config_for(session), **inputs):
            if kind == "done":
                result = value
            elif not paused:
                md = communication_markdown(value)
                if blocking(check_patient_text([md], abnormal)):
                    paused, md = True, PAUSED
                if md.strip():
                    yield Event("partial", title, md + ("" if paused else CURSOR), data={"agent": "communication"})
        return result
