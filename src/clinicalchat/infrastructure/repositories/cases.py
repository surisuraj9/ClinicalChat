"""Care-team cases. Rows come back as dicts with a parsed ``payload``; the workflow rules live in domain/workflow."""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, List, Optional

from clinicalchat.domain import workflow as wf
from clinicalchat.domain.clock import local_minute
from clinicalchat.infrastructure.database import Database

_OPEN_SQL = "(" + ",".join(f"'{s}'" for s in wf.OPEN_STATUSES) + ")"
_BY_RISK = "CASE risk_level WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 WHEN 'MODERATE' THEN 2 ELSE 3 END"


def _dump(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _case(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    row["payload"] = json.loads(row["payload"] or "{}")
    row["escalate"] = bool(row["escalate"])
    row["review_required"] = bool(row["review_required"])
    return row


class CaseRepository:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ reads
    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        rows = self.db.rows("SELECT * FROM cases WHERE case_id=?", (case_id,))
        return _case(rows[0]) if rows else None

    def require(self, case_id: str) -> Dict[str, Any]:
        case = self.get(case_id)
        if not case:
            raise ValueError(f"Unknown case {case_id}")
        return case

    def open_case(self, patient_key: str) -> Optional[Dict[str, Any]]:
        """The patient's case that has not been closed by a clinician yet (anywhere in the care-team workflow)."""
        rows = self.db.rows(f"SELECT * FROM cases WHERE patient_key=? AND status IN {_OPEN_SQL} "
                            "ORDER BY created DESC LIMIT 1", (patient_key,))
        return _case(rows[0]) if rows else None

    def list(self, status: Optional[str] = None, patient_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Most urgent first, then newest."""
        sql, args = "SELECT * FROM cases WHERE 1=1", []
        if status:
            sql += " AND status=?"
            args.append(status)
        if patient_key:
            sql += " AND patient_key=?"
            args.append(patient_key)
        return [_case(r) for r in self.db.rows(f"{sql} ORDER BY {_BY_RISK}, created DESC", tuple(args))]

    def read_imaging_docs(self, patient_key: str) -> List[str]:
        """Imaging documents of this patient that a radiologist has already read (in any case)."""
        return sorted({d for c in self.list(patient_key=patient_key)
                       for d in (c["payload"].get("radiology") or {}).get("doc_ids", [])})

    def undelivered(self, patient_key: str) -> List[Dict[str, Any]]:
        return [_case(r) for r in self.db.rows(
            "SELECT * FROM cases WHERE patient_key=? AND status IN ('VALIDATED','RETURNED') AND delivered=0 ORDER BY reviewed",
            (patient_key,))]

    # ------------------------------------------------------------------ writes
    def create(self, *, patient_key: str, created_by: str, question: str, risk_level: str, escalate: bool,
               review_required: bool, payload: Dict[str, Any], patient_draft: str, status: str = wf.TRIAGE) -> str:
        payload = wf.opening_payload(payload, status)
        case_id = "C-" + uuid.uuid4().hex[:8].upper()
        self.db.execute("INSERT INTO cases (case_id, patient_key, created_by, question, risk_level, escalate, review_required,"
                        " payload, patient_draft, status, created) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (case_id, patient_key, created_by, question, risk_level, int(escalate), int(review_required),
                         _dump(payload), patient_draft, status, local_minute()))
        return case_id

    def update(self, case_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """Write the given columns (payload is serialised, booleans stored as 0/1); returns the updated case."""
        cols = {k: (_dump(v) if k == "payload" else int(v) if isinstance(v, bool) else v) for k, v in values.items()}
        self.db.execute(f"UPDATE cases SET {', '.join(f'{k}=?' for k in cols)} WHERE case_id=?",
                        (*cols.values(), case_id))
        return self.require(case_id)

    def patch_payload(self, case_id: str, patch: Callable[[Dict[str, Any]], None]) -> None:
        """Read-modify-write of one payload in a single locked transaction (safe against concurrent workflow
        updates, e.g. the background Insight finishing while the coordinator triages)."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT payload FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                return
            payload = json.loads(row["payload"])
            patch(payload)
            conn.execute("UPDATE cases SET payload=? WHERE case_id=?", (_dump(payload), case_id))

    def mark_delivered(self, case_id: str) -> None:
        self.db.execute("UPDATE cases SET delivered=1 WHERE case_id=?", (case_id,))

    def consolidate_open_cases(self) -> int:
        """One-time tidy-up: older duplicate open cases of a patient are merged into the newest one."""
        merged = 0
        keys = [r["patient_key"] for r in self.db.rows(
            f"SELECT patient_key FROM cases WHERE status IN {_OPEN_SQL} GROUP BY patient_key HAVING COUNT(*) > 1")]
        for key in keys:
            rows = [_case(r) for r in self.db.rows(f"SELECT * FROM cases WHERE patient_key=? AND status IN {_OPEN_SQL} "
                                                   "ORDER BY created DESC", (key,))]
            payload, level = wf.merge_duplicates(rows)
            keep = rows[0]["case_id"]
            self.update(keep, {"payload": payload, "risk_level": level})
            for c in rows[1:]:
                self.update(c["case_id"], {"status": wf.MERGED, "review_note": f"merged into {keep}", "delivered": True})
                merged += 1
        return merged

    def delete_for_patient(self, patient_key: str) -> int:
        return self.db.execute("DELETE FROM cases WHERE patient_key=?", (patient_key,))
