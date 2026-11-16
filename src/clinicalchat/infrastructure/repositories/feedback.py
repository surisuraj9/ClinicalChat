"""Clinician review notes, fed back into later Insight prompts (learnability)."""

from __future__ import annotations

from typing import Any, Dict, List

from clinicalchat.domain.clock import local_minute
from clinicalchat.infrastructure.database import Database


class FeedbackRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, case_id: str, reviewer: str, action: str, note: str, topic: str) -> None:
        self.db.execute("INSERT INTO feedback (case_id, reviewer, action, note, topic, created) VALUES (?,?,?,?,?,?)",
                        (case_id, reviewer, action, note, topic, local_minute()))

    def recent(self, limit: int = 3) -> List[Dict[str, Any]]:
        return self.db.rows("SELECT * FROM feedback WHERE note != '' ORDER BY id DESC LIMIT ?", (limit,))

    def prompt_text(self, limit: int = 3) -> str:
        """Recent notes as bullet lines for the Insight prompt."""
        return "\n".join(f"- ({n['action'].lower()}) {n['note']}" for n in self.recent(limit)) or "None."

    def delete_for_patient(self, patient_key: str) -> int:
        return self.db.execute("DELETE FROM feedback WHERE case_id IN (SELECT case_id FROM cases WHERE patient_key=?)",
                               (patient_key,))
