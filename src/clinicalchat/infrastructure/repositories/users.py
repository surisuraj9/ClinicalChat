"""Self-registered patient logins (PBKDF2-hashed passwords)."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from clinicalchat.domain.clock import local_minute
from clinicalchat.infrastructure.database import Database

USERNAME_RX = re.compile(r"^[a-z][a-z0-9._-]{2,31}$")
PBKDF2_ROUNDS = 200_000


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS).hex()
    return salt, digest


class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(self, username: str, password: str, display_name: str, patient_key: str,
               age: Optional[int], gender: Optional[str]) -> None:
        salt, digest = hash_password(password)
        self.db.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?)",
                        (username, display_name, patient_key, age, gender, salt, digest, local_minute()))

    def get(self, username: str) -> Optional[Dict[str, Any]]:
        rows = self.db.rows("SELECT * FROM users WHERE username=?", (username,))
        return rows[0] if rows else None

    def all(self) -> List[Dict[str, Any]]:
        return self.db.rows("SELECT username, display_name, patient_key, age, gender, created FROM users ORDER BY created")

    def check_password(self, username: str, password: str) -> bool:
        user = self.get(username)
        if not user:
            return False
        _, digest = hash_password(password, user["salt"])
        return hmac.compare_digest(digest, user["pw_hash"])

    def delete_for_patient(self, patient_key: str) -> int:
        return self.db.execute("DELETE FROM users WHERE patient_key=?", (patient_key,))
