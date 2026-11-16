"""Turns the stream of service Events into chat messages (live activity log, findings, answers) + the agent strip."""

from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional, Tuple

from clinicalchat.domain.conversation import ChatSession, Event
from clinicalchat.domain.roles import AGENT_ORDER, AgentName, Role, policy
from clinicalchat.web.components import agent_strip

Message = dict
CARE_TEAM_NOTE = "for the care team"
SIMPLE_HEADINGS = {"safety": "### 🛡️ Safety check\n", "guidance": "### 🚦 What to do now\n", "blocked": "🔒 "}


class ActivityLog:
    """Single collapsible 'agent activity' message updated in place while agents work."""

    def __init__(self):
        self.steps: Dict[str, str] = {}
        self.started = time.time()
        self.current = "Starting…"

    def apply(self, ev: Event) -> None:
        icon = "⏳" if ev.status == "pending" else "✅"
        text = ev.text.replace("\n", " · ")
        self.steps[ev.key or ev.title] = f"{icon} **{ev.title}** — {text}" if text else f"{icon} **{ev.title}**"
        if ev.status == "pending":
            self.current = ev.title

    def message(self, done: bool = False) -> Message:
        meta = {"title": "🧠 Agent activity" if done else f"⏳ {self.current}…", "status": "done" if done else "pending"}
        if done:
            meta["duration"] = round(time.time() - self.started, 1)
        return {"role": "assistant", "content": "\n\n".join(self.steps.values()) or "…", "metadata": meta}


class _Turn:
    """Mutable chat state of one question while its events arrive."""

    def __init__(self, role: Role, user_text: str, history: List[Message]):
        self.role = role
        self.history = list(history or []) + [{"role": "user", "content": user_text}]
        self.log = ActivityLog()
        self.history.append(self.log.message())
        self.log_idx = len(self.history) - 1
        self.findings: Dict[str, str] = {}
        self.findings_idx: Optional[int] = None
        self.partial_idx: Optional[int] = None
        self.states: Dict[str, tuple] = {}

    def view(self) -> Tuple[List[Message], str]:
        return self.history, agent_strip(self.role, self.states)

    def put(self, idx: Optional[int], content: str) -> int:
        """Replace message ``idx`` (or append a new one) and return its index."""
        msg = {"role": "assistant", "content": content}
        if idx is None:
            self.history.append(msg)
            return len(self.history) - 1
        self.history[idx] = msg
        return idx

    # ---------------------------------------------------------------- event handlers
    def status(self, ev: Event) -> None:
        self.log.apply(ev)
        self.history[self.log_idx] = self.log.message()
        if ev.key == "route" and ev.status == "done":
            self._routed(ev.data.get("plan", {}))
        elif ev.key.startswith("agent-"):
            self._agent(ev.key.removeprefix("agent-"), ev)

    def _routed(self, plan: dict) -> None:
        chosen = set(plan.get("agents", []))
        for a in AGENT_ORDER:
            if a in policy(self.role).agents:
                self.states[a.value] = ("queued", "") if a.value in chosen else ("skipped", "")
        if plan.get("denied_domains"):
            self.states[AgentName.EXTRACTION.value] = ("denied" if not chosen else "queued",
                                                       "no " + ", ".join(plan["denied_domains"]) + " access")

    def _agent(self, agent: str, ev: Event) -> None:
        if ev.status == "pending":
            self.states[agent] = ("working", "")
            return
        short = ev.title.split(" — ", 1)[1] if " — " in ev.title else ""
        failed = "failed" in ev.title
        note = "" if short in ("done", "") else short
        if agent == AgentName.INSIGHT.value and self.role == Role.PATIENT and not failed:
            note = CARE_TEAM_NOTE
        self.states[agent] = ("failed" if failed else "done", note)

    def step(self, ev: Event) -> None:
        self.findings[ev.title] = f"**{ev.title}**\n{ev.text}"
        head = ("### ⚡ Quick check — your findings (instant, no AI)\n" if self.role == Role.PATIENT
                else "### 🔎 What the agents found\n")
        self.findings_idx = self.put(self.findings_idx, head + "\n\n".join(self.findings.values()))

    def partial(self, ev: Event) -> None:
        self.partial_idx = self.put(self.partial_idx, f"### {ev.title}\n{ev.text}")

    def answer(self, ev: Event) -> None:
        content = f"### {ev.title}\n{ev.text}"
        if self.partial_idx is not None and ev.data.get("agent") == "communication":
            self.put(self.partial_idx, content)  # the checked final answer replaces the streamed draft
            self.partial_idx = None
        else:
            self.put(None, content)

    def simple(self, ev: Event) -> None:
        heading = SIMPLE_HEADINGS.get(ev.kind, f"### {ev.title}\n")
        self.put(None, heading + ev.text)

    def handle(self, ev: Event) -> None:
        handler = {"status": self.status, "step": self.step, "partial": self.partial, "answer": self.answer}.get(ev.kind)
        if handler:
            handler(ev)
        elif ev.kind in ("safety", "guidance", "blocked", "case"):
            self.simple(ev)

    def finish(self) -> None:
        self.history[self.log_idx] = self.log.message(done=True)


def render(events: Iterator[Event], user_text: str, history: List[Message],
           session: ChatSession) -> Iterator[Tuple[List[Message], str]]:
    """Yield (chat history, agent strip) after every event; never leaves the UI hanging on an error."""
    turn = _Turn(session.role, user_text, history)
    session.busy = True
    yield turn.view()
    try:
        for ev in events:
            turn.handle(ev)
            yield turn.view()
    except Exception as exc:  # noqa: BLE001 - shown to the user instead of a frozen spinner
        turn.put(None, f"⚠️ Something went wrong: {exc}")
    finally:
        session.busy = False
    turn.finish()
    yield turn.view()
