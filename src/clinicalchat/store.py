"""SQLite persistence for self-registered patients, their symptoms and uploads, review cases and clinician feedback."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from clinicalchat import workflow as wf
from clinicalchat.config import Settings, get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, display_name TEXT NOT NULL, patient_key TEXT NOT NULL UNIQUE,
    age INTEGER, gender TEXT, salt TEXT NOT NULL, pw_hash TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profile_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_key TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
    created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS uploads (
    doc_id TEXT PRIMARY KEY, patient_key TEXT NOT NULL, filename TEXT NOT NULL, doc_json TEXT NOT NULL,
    extraction_json TEXT NOT NULL, uploaded_by TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY, patient_key TEXT NOT NULL, created_by TEXT NOT NULL, question TEXT NOT NULL,
    risk_level TEXT NOT NULL, escalate INTEGER NOT NULL, review_required INTEGER NOT NULL, payload TEXT NOT NULL,
    patient_draft TEXT NOT NULL, status TEXT NOT NULL, reviewer TEXT, review_note TEXT, final_message TEXT,
    delivered INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL, reviewed TEXT);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, reviewer TEXT, action TEXT, note TEXT, topic TEXT,
    created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS insights (
    patient_key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, insight TEXT NOT NULL, created TEXT NOT NULL);
"""

USERNAME_RX = re.compile(r"^[a-z][a-z0-9._-]{2,31}$")
PBKDF2_ROUNDS = 200_000
CASE_STATUSES = wf.CASE_STATUSES
OPEN_STATUSES = wf.OPEN_STATUSES
_OPEN_SQL = "(" + ",".join(f"'{s}'" for s in OPEN_STATUSES) + ")"
RISK_ORDER = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS).hex()
    return salt, digest


class Store:
    def __init__(self, settings: Optional[Settings] = None, path: Optional[Path] = None):
        self.settings = settings or get_settings()
        self.path = path or (self.settings.work_dir / "state" / "clinicalchat.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
        self.consolidate_open_cases()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _exec(self, sql: str, args: tuple = ()) -> None:
        with self._lock, self._conn() as c:
            c.execute(sql, args)

    def _rows(self, sql: str, args: tuple = ()) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    # ------------------------------------------------------------------ users
    def create_user(self, username: str, password: str, display_name: str, patient_key: str,
                    age: Optional[int], gender: Optional[str]) -> None:
        salt, digest = hash_password(password)
        self._exec("INSERT INTO users VALUES (?,?,?,?,?,?,?,?)",
                   (username, display_name, patient_key, age, gender, salt, digest, _now()))

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM users WHERE username=?", (username,))
        return rows[0] if rows else None

    def users(self) -> List[Dict[str, Any]]:
        return self._rows("SELECT username, display_name, patient_key, age, gender, created FROM users ORDER BY created")

    def check_password(self, username: str, password: str) -> bool:
        user = self.get_user(username)
        if not user:
            return False
        _, digest = hash_password(password, user["salt"])
        return hmac.compare_digest(digest, user["pw_hash"])

    # ------------------------------------------------------------------ profile facts (symptoms, allergies, meds)
    def add_fact(self, patient_key: str, kind: str, value: str) -> None:
        self._exec("INSERT INTO profile_facts (patient_key, kind, value, created) VALUES (?,?,?,?)",
                   (patient_key, kind, value, _now()))

    def facts(self, patient_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if patient_key:
            return self._rows("SELECT * FROM profile_facts WHERE patient_key=? ORDER BY id", (patient_key,))
        return self._rows("SELECT * FROM profile_facts ORDER BY id")

    # ------------------------------------------------------------------ uploads
    def save_upload(self, doc_json: str, extraction: Dict[str, Any], patient_key: str, doc_id: str, filename: str,
                    uploaded_by: str) -> None:
        self._exec("INSERT OR REPLACE INTO uploads VALUES (?,?,?,?,?,?,?)",
                   (doc_id, patient_key, filename, doc_json, json.dumps(extraction), uploaded_by, _now()))

    def uploads(self, patient_key: Optional[str] = None) -> List[Dict[str, Any]]:
        if patient_key:
            return self._rows("SELECT * FROM uploads WHERE patient_key=? ORDER BY created", (patient_key,))
        return self._rows("SELECT * FROM uploads ORDER BY created")

    # ------------------------------------------------------------------ review cases
    def create_case(self, *, patient_key: str, created_by: str, question: str, risk_level: str, escalate: bool,
                    review_required: bool, payload: Dict[str, Any], patient_draft: str, status: str = wf.TRIAGE) -> str:
        if status not in OPEN_STATUSES:
            raise ValueError(f"Unknown status {status}")
        case_id = "C-" + uuid.uuid4().hex[:8].upper()
        payload = {**payload, "timeline": [_event("Patient", "Case opened — AI assessment shared with the patient"),
                                           _event("System", f"Routed: {wf.STAGE_LABEL[status][0]}")]}
        self._exec("INSERT INTO cases (case_id, patient_key, created_by, question, risk_level, escalate, review_required,"
                   " payload, patient_draft, status, created) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (case_id, patient_key, created_by, question, risk_level, int(escalate), int(review_required),
                    json.dumps(payload, default=str), patient_draft, status, _now()))
        return case_id

    def open_case(self, patient_key: str) -> Optional[Dict[str, Any]]:
        """The patient's case that has not been closed by a clinician yet (anywhere in the care-team workflow)."""
        rows = self._rows(f"SELECT * FROM cases WHERE patient_key=? AND status IN {_OPEN_SQL} "
                          "ORDER BY created DESC LIMIT 1", (patient_key,))
        return _case(rows[0]) if rows else None

    def update_case(self, case_id: str, *, question: str, risk_level: str, escalate: bool, review_required: bool,
                    payload: Dict[str, Any], patient_draft: str, status: Optional[str] = None) -> None:
        """New question / reports for an open case: keep ONE case per patient. The latest assessment replaces the draft;
        the question history, radiology read, triage and timeline are kept, and the risk level never drops until a
        clinician has looked at it."""
        old = self.case(case_id)
        if not old:
            raise ValueError(f"Unknown case {case_id}")
        status = status or old["status"]
        if status not in OPEN_STATUSES:
            raise ValueError(f"Unknown status {status}")
        history = list(old["payload"].get("history") or [{"question": old["question"], "at": old["created"]}])
        history.append({"question": question, "at": _now()})
        keep = {k: old["payload"][k] for k in ("radiology", "triage", "timeline") if k in old["payload"]}
        timeline = list(keep.get("timeline") or [])
        timeline.append(_event("Patient", "Added a message / reports — AI assessment updated"))
        if status != old["status"]:
            timeline.append(_event("System", f"Routed: {wf.STAGE_LABEL[status][0]}"))
        payload = {**payload, **keep, "history": history[-20:], "timeline": timeline[-40:]}
        level = max(old["risk_level"], risk_level, key=lambda x: RISK_ORDER.index(x) if x in RISK_ORDER else 0)
        self._exec("UPDATE cases SET question=?, risk_level=?, escalate=?, review_required=?, payload=?, patient_draft=?, "
                   "status=?, reviewer=NULL, review_note=NULL, final_message=NULL, delivered=0, reviewed=NULL "
                   "WHERE case_id=?",
                   (question, level, int(escalate or old["escalate"]), int(review_required or old["review_required"]),
                    json.dumps(payload, default=str), patient_draft, status, case_id))

    def _save_payload(self, case_id: str, payload: Dict[str, Any], status: str, risk_level: Optional[str] = None) -> None:
        if risk_level:
            self._exec("UPDATE cases SET payload=?, status=?, risk_level=? WHERE case_id=?",
                       (json.dumps(payload, default=str), status, risk_level, case_id))
        else:
            self._exec("UPDATE cases SET payload=?, status=? WHERE case_id=?",
                       (json.dumps(payload, default=str), status, case_id))

    def radiology_read(self, case_id: str, *, reader: str, verdict: str, report: str, doc_ids: List[str]) -> Dict[str, Any]:
        case = self.case(case_id)
        if not case or case["status"] != wf.RADIOLOGY:
            raise ValueError(f"Case {case_id} is not waiting for a radiologist")
        critical = verdict == wf.RADIOLOGY_VERDICTS[2]
        pay = case["payload"]
        reads = list((pay.get("radiology") or {}).get("reads") or [])
        reads.append({"reader": reader, "verdict": verdict, "report": report, "doc_ids": doc_ids, "at": _now()})
        pay["radiology"] = {"reads": reads, "doc_ids": sorted({d for r in reads for d in r["doc_ids"]})}
        status = wf.status_after_radiology(case, critical)
        pay.setdefault("timeline", []).append(_event(reader, f"Radiology read: {verdict}"))
        pay["timeline"].append(_event("System", f"Routed: {wf.STAGE_LABEL[status][0]}"))
        self._save_payload(case_id, pay, status, "CRITICAL" if critical else None)
        return self.case(case_id)

    def triage(self, case_id: str, *, coordinator: str, priority: str, clinician: str, clinician_name: str,
               note: str) -> Dict[str, Any]:
        case = self.case(case_id)
        if not case or case["status"] != wf.TRIAGE:
            raise ValueError(f"Case {case_id} is not waiting for triage")
        if priority not in wf.PRIORITIES:
            raise ValueError(f"Unknown priority {priority}")
        pay = case["payload"]
        pay["triage"] = {"coordinator": coordinator, "priority": priority, "clinician": clinician,
                         "clinician_name": clinician_name, "note": note, "at": _now()}
        pay.setdefault("timeline", []).append(
            _event(coordinator, f"Triaged {wf.PRIORITY_ICON[priority]} {priority} → assigned to {clinician_name}"))
        self._save_payload(case_id, pay, wf.CLINICIAN)
        return self.case(case_id)

    def read_imaging_docs(self, patient_key: str) -> List[str]:
        """Imaging documents of this patient that a radiologist has already read (in any case)."""
        return sorted({d for c in self.cases(patient_key=patient_key)
                       for d in (c["payload"].get("radiology") or {}).get("doc_ids", [])})

    def consolidate_open_cases(self) -> int:
        """One-time tidy-up: older duplicate open cases of a patient are merged into the newest one."""
        merged = 0
        keys = [r["patient_key"] for r in self._rows(
            f"SELECT patient_key FROM cases WHERE status IN {_OPEN_SQL} GROUP BY patient_key HAVING COUNT(*) > 1")]
        for key in keys:
            rows = [_case(r) for r in self._rows(f"SELECT * FROM cases WHERE patient_key=? AND status IN {_OPEN_SQL} "
                                                 "ORDER BY created DESC", (key,))]
            keep, older = rows[0], rows[1:]
            history = [{"question": c["question"], "at": c["created"]} for c in reversed(older)]
            history += keep["payload"].get("history") or [{"question": keep["question"], "at": keep["created"]}]
            payload = {**keep["payload"], "history": history[-20:]}
            level = max([c["risk_level"] for c in rows], key=lambda x: RISK_ORDER.index(x) if x in RISK_ORDER else 0)
            self._exec("UPDATE cases SET payload=?, risk_level=? WHERE case_id=?",
                       (json.dumps(payload, default=str), level, keep["case_id"]))
            for c in older:
                self._exec("UPDATE cases SET status='MERGED', review_note=?, delivered=1 WHERE case_id=?",
                           (f"merged into {keep['case_id']}", c["case_id"]))
                merged += 1
        return merged

    def case(self, case_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT * FROM cases WHERE case_id=?", (case_id,))
        return _case(rows[0]) if rows else None

    def cases(self, status: Optional[str] = None, patient_key: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, args = "SELECT * FROM cases WHERE 1=1", []
        if status:
            sql += " AND status=?"
            args.append(status)
        if patient_key:
            sql += " AND patient_key=?"
            args.append(patient_key)
        sql += " ORDER BY CASE risk_level WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 WHEN 'MODERATE' THEN 2 ELSE 3 END, created DESC"
        return [_case(r) for r in self._rows(sql, tuple(args))]

    def review_case(self, case_id: str, *, reviewer: str, status: str, note: str, final_message: str) -> None:
        if status not in CASE_STATUSES:
            raise ValueError(f"Unknown status {status}")
        case = self.case(case_id)
        if not case:
            raise ValueError(f"Unknown case {case_id}")
        pay = case["payload"]
        pay.setdefault("timeline", []).append(_event(reviewer, "Confirmed the assessment — case closed" if status == wf.VALIDATED
                                                     else "Asked the patient for more information"))
        self._exec("UPDATE cases SET status=?, reviewer=?, review_note=?, final_message=?, reviewed=?, delivered=0, payload=? "
                   "WHERE case_id=?", (status, reviewer, note, final_message, _now(), json.dumps(pay, default=str), case_id))

    def undelivered(self, patient_key: str) -> List[Dict[str, Any]]:
        return [_case(r) for r in self._rows(
            "SELECT * FROM cases WHERE patient_key=? AND status IN ('VALIDATED','RETURNED') AND delivered=0 ORDER BY reviewed",
            (patient_key,))]

    def mark_delivered(self, case_id: str) -> None:
        self._exec("UPDATE cases SET delivered=1 WHERE case_id=?", (case_id,))

    # ------------------------------------------------------------------ learnability: clinician feedback
    def add_feedback(self, case_id: str, reviewer: str, action: str, note: str, topic: str) -> None:
        self._exec("INSERT INTO feedback (case_id, reviewer, action, note, topic, created) VALUES (?,?,?,?,?,?)",
                   (case_id, reviewer, action, note, topic, _now()))

    def recent_feedback(self, limit: int = 3) -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM feedback WHERE note != '' ORDER BY id DESC LIMIT ?", (limit,))

    # ------------------------------------------------------------------ latest Insight per patient (reused while evidence is unchanged)
    def save_insight(self, patient_key: str, fingerprint: str, insight: Dict[str, Any]) -> None:
        self._exec("INSERT OR REPLACE INTO insights (patient_key, fingerprint, insight, created) VALUES (?,?,?,?)",
                   (patient_key, fingerprint, json.dumps(insight, default=str), _now()))

    def insight(self, patient_key: str, fingerprint: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT insight FROM insights WHERE patient_key=? AND fingerprint=?", (patient_key, fingerprint))
        return json.loads(rows[0]["insight"]) if rows else None


def _event(actor: str, text: str) -> Dict[str, str]:
    return {"at": _now(), "actor": actor, "event": text}


def _case(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    row["payload"] = json.loads(row["payload"] or "{}")
    row["escalate"] = bool(row["escalate"])
    row["review_required"] = bool(row["review_required"])
    return row


_STORE: Optional[Store] = None
_LOCK = threading.Lock()


def get_store() -> Store:
    global _STORE
    with _LOCK:
        if _STORE is None:
            _STORE = Store()
        return _STORE
