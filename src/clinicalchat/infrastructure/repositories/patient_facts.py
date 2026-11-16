"""What patients told us about themselves: symptoms, allergies, medicines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from clinicalchat.domain.clock import local_minute
from clinicalchat.infrastructure.database import Database


class PatientFactRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, patient_key: str, kind: str, value: str) -> None:
        self.db.execute("INSERT INTO profile_facts (patient_key, kind, value, created) VALUES (?,?,?,?)",
                        (patient_key, kind, value, local_minute()))

    def all(self, patient_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if patient_key:
            return self.db.rows("SELECT * FROM profile_facts WHERE patient_key=? ORDER BY id", (patient_key,))
        return self.db.rows("SELECT * FROM profile_facts ORDER BY id")

    def delete_for_patient(self, patient_key: str) -> int:
        return self.db.execute("DELETE FROM profile_facts WHERE patient_key=?", (patient_key,))
