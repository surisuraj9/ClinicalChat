"""Admin: wipe a patient's stored data (uploads, facts, cases + review notes, cached insight, exports; optionally the
login) so a flow can be tested again from the start."""

from __future__ import annotations

from typing import Dict, Optional, Protocol

from clinicalchat.infrastructure.repositories.caches import InsightCache
from clinicalchat.infrastructure.repositories.cases import CaseRepository
from clinicalchat.infrastructure.repositories.feedback import FeedbackRepository
from clinicalchat.infrastructure.repositories.patient_facts import PatientFactRepository
from clinicalchat.infrastructure.repositories.uploads import UploadRepository
from clinicalchat.infrastructure.repositories.users import UserRepository
from clinicalchat.services.export import SessionExporter


class UploadIndex(Protocol):
    """Anything that can forget a patient's uploaded reports: the VectorIndex (CLI) or the live KnowledgeBase."""

    def purge_uploads(self, patient_key: str) -> int: ...


class PatientDataAdmin:
    def __init__(self, users: UserRepository, uploads: UploadRepository, facts: PatientFactRepository,
                 cases: CaseRepository, feedback: FeedbackRepository, insights: InsightCache, index: UploadIndex,
                 exporter: SessionExporter):
        self.users = users
        self.uploads = uploads
        self.facts = facts
        self.cases = cases
        self.feedback = feedback
        self.insights = insights
        self.index = index
        self.exporter = exporter

    def find_user(self, patient: str) -> Optional[Dict]:
        """By login username or patient key."""
        return self.users.get(patient) or next((u for u in self.users.all() if u["patient_key"] == patient), None)

    def summary(self, key: str) -> Dict[str, int]:
        return {"uploads": len(self.uploads.all(key)), "cases": len(self.cases.list(patient_key=key))}

    def purge(self, key: str, delete_login: bool = False) -> Dict[str, int]:
        """Returns the number of rows / chunks / files removed per kind."""
        user = self.find_user(key)
        removed = {"feedback": self.feedback.delete_for_patient(key),  # finds the notes via the cases: delete first
                   "cases": self.cases.delete_for_patient(key), "uploads": self.uploads.delete_for_patient(key),
                   "profile_facts": self.facts.delete_for_patient(key), "insights": self.insights.delete_for_patient(key)}
        if delete_login:
            removed["users"] = self.users.delete_for_patient(key)
        removed["index_chunks"] = self.index.purge_uploads(key)
        exports = self.exporter.files_of(user["username"]) if user else []
        for path in exports:
            path.unlink()
        removed["exports"] = len(exports)
        return removed
