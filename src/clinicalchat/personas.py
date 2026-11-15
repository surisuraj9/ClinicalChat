"""Login accounts: built-in demo personas (one per role) + self-registered patients (SQLite, PBKDF2 passwords).
Patients are always bound to their own record."""

from __future__ import annotations

import hmac
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from clinicalchat.config import get_settings
from clinicalchat.rbac import Role


@dataclass(frozen=True)
class Persona:
    username: str
    display_name: str
    role: Role
    title: str
    patient_key: Optional[str] = None


PERSONAS: Dict[str, Persona] = {p.username: p for p in [
    Persona("dr.sharma", "Dr. Neha Sharma", Role.CLINICIAN, "Consultant Physician"),
    Persona("dr.menon", "Dr. Kavya Menon", Role.CLINICIAN, "General Physician"),
    Persona("dr.iyer", "Dr. Arjun Iyer", Role.RADIOLOGIST, "Consultant Radiologist"),
    Persona("coord.thomas", "Maria Thomas", Role.COORDINATOR, "Care Coordinator"),
    Persona("ananya.rao", "Ananya Rao", Role.PATIENT, "Patient", "ananya-rao-54"),
    Persona("rahul.mehta", "Rahul Mehta", Role.PATIENT, "Patient", "rahul-mehta-32"),
    Persona("riya.das", "Riya Das", Role.PATIENT, "Patient", "riya-das-27"),
]}


def get_persona(username: Optional[str]) -> Optional[Persona]:
    name = (username or "").strip().lower()
    if name in PERSONAS:
        return PERSONAS[name]
    from clinicalchat.store import get_store

    user = get_store().get_user(name) if name else None
    if user:
        return Persona(user["username"], user["display_name"], Role.PATIENT, "Patient (self-registered)", user["patient_key"])
    return None


def authenticate(username: str, password: str) -> bool:
    name = (username or "").strip().lower()
    if name in PERSONAS:
        return hmac.compare_digest(password.encode(), get_settings().demo_password.encode())
    from clinicalchat.store import get_store

    return get_store().check_password(name, password)


def _split(text: str) -> List[str]:
    return [x.strip() for x in (text or "").replace(";", ",").split(",") if x.strip()][:15]


def signup(username: str, password: str, confirm: str, full_name: str, age: str, gender: str,
           allergies: str = "", medications: str = "") -> Tuple[bool, str]:
    """Self-registration for patients: creates the login and an (empty) patient record that only they can see."""
    from clinicalchat.engine import get_engine
    from clinicalchat.store import USERNAME_RX, get_store

    username = (username or "").strip().lower()
    full_name = " ".join((full_name or "").split()).title()
    if not USERNAME_RX.match(username):
        return False, "Username must be 3–32 characters: lowercase letters, digits, '.', '_' or '-', starting with a letter."
    store = get_store()
    if username in PERSONAS or store.get_user(username):
        return False, "That username is already taken."
    if len(password or "") < 8:
        return False, "Password must be at least 8 characters."
    if password != confirm:
        return False, "Passwords do not match."
    if len(full_name) < 2 or len(full_name) > 80:
        return False, "Please enter your full name."
    try:
        age_n = int(str(age).strip())
        if not 0 < age_n < 120:
            raise ValueError
    except ValueError:
        return False, "Please enter a valid age."
    gender = gender if gender in ("Female", "Male", "Other") else None

    engine = get_engine()
    key = engine.new_patient_key(full_name, age_n)
    try:
        store.create_user(username, password, full_name, key, age_n, gender)
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    engine.create_patient(key, full_name, age_n, gender)
    for kind, values in (("allergy", _split(allergies)), ("medication", _split(medications))):
        for v in values:
            store.add_fact(key, kind, v)
            engine.apply_fact(key, kind, v)
    return True, key
