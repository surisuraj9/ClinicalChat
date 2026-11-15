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
"""

USERNAME_RX = re.compile(r"^[a-z][a-z0-9._-]{2,31}$")
PBKDF2_ROUNDS = 200_000
CASE_STATUSES = ("PENDING_REVIEW", "VALIDATED", "RETURNED")


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
                    review_required: bool, payload: Dict[str, Any], patient_draft: str) -> str:
        case_id = "C-" + uuid.uuid4().hex[:8].upper()
        self._exec("INSERT INTO cases (case_id, patient_key, created_by, question, risk_level, escalate, review_required,"
                   " payload, patient_draft, status, created) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (case_id, patient_key, created_by, question, risk_level, int(escalate), int(review_required),
                    json.dumps(payload, default=str), patient_draft, "PENDING_REVIEW", _now()))
        return case_id

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
        self._exec("UPDATE cases SET status=?, reviewer=?, review_note=?, final_message=?, reviewed=?, delivered=0 "
                   "WHERE case_id=?", (status, reviewer, note, final_message, _now(), case_id))

    def undelivered(self, patient_key: str) -> List[Dict[str, Any]]:
        return [_case(r) for r in self._rows(
            "SELECT * FROM cases WHERE patient_key=? AND status!='PENDING_REVIEW' AND delivered=0 ORDER BY reviewed",
            (patient_key,))]

    def mark_delivered(self, case_id: str) -> None:
        self._exec("UPDATE cases SET delivered=1 WHERE case_id=?", (case_id,))

    # ------------------------------------------------------------------ learnability: clinician feedback
    def add_feedback(self, case_id: str, reviewer: str, action: str, note: str, topic: str) -> None:
        self._exec("INSERT INTO feedback (case_id, reviewer, action, note, topic, created) VALUES (?,?,?,?,?,?)",
                   (case_id, reviewer, action, note, topic, _now()))

    def recent_feedback(self, limit: int = 3) -> List[Dict[str, Any]]:
        return self._rows("SELECT * FROM feedback WHERE note != '' ORDER BY id DESC LIMIT ?", (limit,))


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
