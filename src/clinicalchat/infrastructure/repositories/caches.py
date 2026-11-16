"""Caches that save AI / web calls: the latest Insight per patient, answers to general questions, web results."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

from clinicalchat.domain.clock import local_minute
from clinicalchat.infrastructure.database import Database


class InsightCache:
    """The latest clinician-facing Insight per patient, reused while the evidence fingerprint is unchanged."""

    def __init__(self, db: Database):
        self.db = db

    def save(self, patient_key: str, fingerprint: str, insight: Dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO insights (patient_key, fingerprint, insight, created) VALUES (?,?,?,?)",
                        (patient_key, fingerprint, json.dumps(insight, default=str), local_minute()))

    def get(self, patient_key: str, fingerprint: str) -> Optional[Dict[str, Any]]:
        rows = self.db.rows("SELECT insight FROM insights WHERE patient_key=? AND fingerprint=?", (patient_key, fingerprint))
        return json.loads(rows[0]["insight"]) if rows else None

    def delete_for_patient(self, patient_key: str) -> int:
        return self.db.execute("DELETE FROM insights WHERE patient_key=?", (patient_key,))


class FaqCache:
    """Answers to general (non-personal) questions; a similarly worded question with the same allergies reuses one."""

    def __init__(self, db: Database):
        self.db = db

    def lookup(self, question: str, allergies: str, max_age_days: int = 7) -> Optional[Dict[str, Any]]:
        terms = faq_terms(question)
        if not terms:
            return None
        rows = self.db.rows("SELECT * FROM faq_cache WHERE allergies=? AND created >= datetime('now', ?) ORDER BY id DESC "
                            "LIMIT 500", (allergies, f"-{max_age_days} days"))
        best, best_score = None, 0.0
        for r in rows:
            other = set(r["terms"].split())
            score = len(terms & other) / len(terms | other)
            if score > best_score:
                best, best_score = r, score
        if best and (best_score == 1.0 or (best_score >= 0.75 and len(terms) >= 3)):
            self.db.execute("UPDATE faq_cache SET hits=hits+1 WHERE id=?", (best["id"],))
            return best
        return None

    def save(self, question: str, allergies: str, answer: str) -> None:
        terms = faq_terms(question)
        if terms:
            self.db.execute("INSERT INTO faq_cache (terms, allergies, question, answer, created) VALUES (?,?,?,?,datetime('now'))",
                            (" ".join(sorted(terms)), allergies, question, answer))


class WebCache:
    """Trusted-site search results (no patient data) kept for a week."""

    def __init__(self, db: Database):
        self.db = db

    def lookup(self, kind: str, query: str, days: int = 7) -> Optional[List[Dict[str, Any]]]:
        rows = self.db.rows("SELECT results FROM web_cache WHERE kind=? AND query=? AND created >= datetime('now', ?)",
                            (kind, query.lower(), f"-{days} days"))
        return json.loads(rows[0]["results"]) if rows else None

    def save(self, kind: str, query: str, results: List[Dict[str, Any]]) -> None:
        if results:
            self.db.execute("INSERT OR REPLACE INTO web_cache VALUES (?,?,?,datetime('now'))",
                            (kind, query.lower(), json.dumps(results)))


_FAQ_STOP = set("""a an the is are was were be been am do does did what whats what's which who how why when where can could
should would will shall may might must i me my we you your it its this that these those of in on for to from by with about
and or as at if than then so please tell explain mean means meaning define definition know want need help some any much
many there their they them he she his her our us normal""".split())


def faq_terms(question: str) -> Set[str]:
    """Content words of a question, lightly normalised ("What's HbA1c?" == "what is hba1c" == "HbA1c meaning")."""
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower().replace("'s", "")) if w not in _FAQ_STOP]
    return {_us_spelling(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us")) else w) for w in words}


def _us_spelling(word: str) -> str:
    """haemoglobin/hemoglobin, anaemia/anemia, oedema/edema, tumour/tumor, -isation/-ization -> one key."""
    word = re.sub(r"isation$", "ization", word)
    word = re.sub(r"(?<=[a-z])our$", "or", word) if len(word) > 5 else word
    return word.replace("ae", "e").replace("oe", "e")
