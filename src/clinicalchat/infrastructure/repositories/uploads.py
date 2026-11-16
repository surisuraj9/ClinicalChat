"""Uploaded reports: the parsed document (JSON) and its extraction-quality score."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from clinicalchat.domain.clock import local_minute
from clinicalchat.infrastructure.database import Database


class UploadRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, doc_json: str, extraction: Dict[str, Any], patient_key: str, doc_id: str, filename: str,
             uploaded_by: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO uploads VALUES (?,?,?,?,?,?,?)",
                        (doc_id, patient_key, filename, doc_json, json.dumps(extraction), uploaded_by, local_minute()))

    def all(self, patient_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if patient_key:
            return self.db.rows("SELECT * FROM uploads WHERE patient_key=? ORDER BY created", (patient_key,))
        return self.db.rows("SELECT * FROM uploads ORDER BY created")

    def delete_for_patient(self, patient_key: str) -> int:
        return self.db.execute("DELETE FROM uploads WHERE patient_key=?", (patient_key,))
