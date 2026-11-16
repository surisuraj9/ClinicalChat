"""SQLite database: one file, one schema, thread-safe writes. Repositories (see ``repositories/``) own the SQL."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

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
CREATE TABLE IF NOT EXISTS faq_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT, terms TEXT NOT NULL, allergies TEXT NOT NULL, question TEXT NOT NULL,
    answer TEXT NOT NULL, hits INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS web_cache (
    kind TEXT NOT NULL, query TEXT NOT NULL, results TEXT NOT NULL, created TEXT NOT NULL, PRIMARY KEY (kind, query));
CREATE TABLE IF NOT EXISTS insights (
    patient_key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, insight TEXT NOT NULL, created TEXT NOT NULL);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str, args: tuple = ()) -> int:
        """Run one write statement; returns the number of rows changed."""
        with self.transaction() as conn:
            return conn.execute(sql, args).rowcount

    def rows(self, sql: str, args: tuple = ()) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Several statements as one locked unit (committed on success, rolled back on error)."""
        with self._lock, self._connect() as conn:
            yield conn
