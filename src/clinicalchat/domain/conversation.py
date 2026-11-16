"""A chat conversation: the logged-in session, its turns and the live events streamed to the UI."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clinicalchat.domain.clock import local_minute
from clinicalchat.domain.roles import Role

HISTORY_TURNS = 2  # short history keeps prompts (and time-to-first-word) small


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
    ts: str = field(default_factory=local_minute)

    def add_answer(self, label: str, markdown: str, summary: str) -> None:
        self.answers.append({"label": label, "markdown": markdown, "summary": summary[:300]})


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
    last_insight: Optional[Dict[str, Any]] = None  # {patient_key, question, output, risk, facts}
    busy: bool = False  # a question is streaming (the UI poller must not touch the chat)

    def history_text(self) -> str:
        lines = []
        for t in self.turns[-HISTORY_TURNS:]:
            answer = " ".join(a["summary"] for a in t.answers)[:400] if t.answers else t.blocked
            lines.append(f"Q: {t.question}\nA: {answer}")
        return "\n".join(lines)


@dataclass
class Event:
    """One live update for the chat. kind: status | step | safety | guidance | partial | answer | case | blocked."""
    kind: str
    title: str = ""
    text: str = ""
    status: str = "done"  # pending | done
    key: str = ""  # stable id so the UI can update a status line in place
    data: Dict[str, Any] = field(default_factory=dict)
