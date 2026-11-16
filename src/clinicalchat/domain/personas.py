"""Built-in demo accounts, one or more per role. Self-registered patients live in the database (see services/accounts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from clinicalchat.domain.roles import Role


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


def clinicians() -> List[Persona]:
    return [p for p in PERSONAS.values() if p.role == Role.CLINICIAN]


def self_registered(username: str, display_name: str, patient_key: str) -> Persona:
    return Persona(username, display_name, Role.PATIENT, "Patient (self-registered)", patient_key)
