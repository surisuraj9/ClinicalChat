"""Logins: the built-in demo personas (one shared demo password) and self-registered patients (database, PBKDF2)."""

from __future__ import annotations

import hmac
import sqlite3
from typing import List, Optional, Tuple

from clinicalchat.domain.personas import PERSONAS, Persona, self_registered
from clinicalchat.infrastructure.repositories.patient_facts import PatientFactRepository
from clinicalchat.infrastructure.repositories.users import USERNAME_RX, UserRepository
from clinicalchat.services.patients import PatientDirectory

GENDERS = ("Female", "Male", "Other")


def _split(text: str) -> List[str]:
    return [x.strip() for x in (text or "").replace(";", ",").split(",") if x.strip()][:15]


def _parse_age(age: str) -> Optional[int]:
    try:
        n = int(str(age).strip())
    except ValueError:
        return None
    return n if 0 < n < 120 else None


class AccountService:
    def __init__(self, users: UserRepository, facts: PatientFactRepository, patients: PatientDirectory,
                 demo_password: str):
        self.users = users
        self.facts = facts
        self.patients = patients
        self._demo_password = demo_password

    def get_persona(self, username: Optional[str]) -> Optional[Persona]:
        name = (username or "").strip().lower()
        if name in PERSONAS:
            return PERSONAS[name]
        user = self.users.get(name) if name else None
        if user:
            return self_registered(user["username"], user["display_name"], user["patient_key"])
        return None

    def authenticate(self, username: str, password: str) -> bool:
        name = (username or "").strip().lower()
        if name in PERSONAS:
            return hmac.compare_digest(password.encode(), self._demo_password.encode())
        return self.users.check_password(name, password)

    def signup(self, username: str, password: str, confirm: str, full_name: str, age: str, gender: str,
               allergies: str = "", medications: str = "") -> Tuple[bool, str]:
        """Self-registration for patients: creates the login and an (empty) patient record only they can see.
        Returns (ok, patient key or error message)."""
        username = (username or "").strip().lower()
        full_name = " ".join((full_name or "").split()).title()
        error = self._validate(username, password, confirm, full_name, age)
        if error:
            return False, error
        age_n = _parse_age(age)
        gender = gender if gender in GENDERS else None
        key = self.patients.new_key(full_name, age_n)
        try:
            self.users.create(username, password, full_name, key, age_n, gender)
        except sqlite3.IntegrityError:
            return False, "That username is already taken."
        self.patients.create(key, full_name, age_n, gender)
        for kind, values in (("allergy", _split(allergies)), ("medication", _split(medications))):
            for v in values:
                self.facts.add(key, kind, v)
                self.patients.apply_fact(key, kind, v)
        return True, key

    def _validate(self, username: str, password: str, confirm: str, full_name: str, age: str) -> str:
        if not USERNAME_RX.match(username):
            return "Username must be 3–32 characters: lowercase letters, digits, '.', '_' or '-', starting with a letter."
        if username in PERSONAS or self.users.get(username):
            return "That username is already taken."
        if len(password or "") < 8:
            return "Password must be at least 8 characters."
        if password != confirm:
            return "Passwords do not match."
        if not 2 <= len(full_name) <= 80:
            return "Please enter your full name."
        if _parse_age(age) is None:
            return "Please enter a valid age."
        return ""
