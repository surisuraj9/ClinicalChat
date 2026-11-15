"""Demo login accounts — one per role from the problem statement (patients are bound to their own record)."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Dict, Optional

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
    Persona("dr.iyer", "Dr. Arjun Iyer", Role.RADIOLOGIST, "Consultant Radiologist"),
    Persona("coord.thomas", "Maria Thomas", Role.COORDINATOR, "Care Coordinator"),
    Persona("ananya.rao", "Ananya Rao", Role.PATIENT, "Patient", "ananya-rao-54"),
    Persona("rahul.mehta", "Rahul Mehta", Role.PATIENT, "Patient", "rahul-mehta-32"),
    Persona("riya.das", "Riya Das", Role.PATIENT, "Patient", "riya-das-27"),
]}


def get_persona(username: Optional[str]) -> Optional[Persona]:
    return PERSONAS.get((username or "").strip().lower())


def authenticate(username: str, password: str) -> bool:
    persona = get_persona(username)
    expected = get_settings().demo_password
    return persona is not None and hmac.compare_digest(password.encode(), expected.encode())
