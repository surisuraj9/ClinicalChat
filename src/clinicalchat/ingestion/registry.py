"""Patient registry: resolves patient identities across documents and extracts safety-critical facts
(allergies, medications, abnormal labs, imaging) from per-patient sections."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from clinicalchat.models import AllergyRecord, ClinicalDocument, Domain, PatientProfile, Section

_NONE_WORDS = {"none", "nil", "n/a", "na", "not applicable", "no", "negative", "not required",
               "no symptoms", "never reported", "none identified", "-", "–"}

# filename keyword -> (canonical allergen, category)
ALLERGEN_FROM_FILENAME = {
    "peanut": ("peanut", "Food"), "fish": ("fish", "Food"), "soy": ("soy", "Food"),
    "wheat": ("wheat", "Food"), "gluten": ("gluten", "Food"), "milk": ("milk", "Food"),
    "house mite": ("dust mite", "Environmental"), "cockroach": ("cockroach", "Environmental"),
    "pet dander": ("pet dander", "Environmental"), "pollen": ("pollen", "Environmental"),
    "ibuprofen": ("ibuprofen", "Medication"), "paracetamol": ("paracetamol", "Medication"),
    "sulfamethoxazole": ("sulfamethoxazole", "Medication"),
    "atopic dermatitis": ("atopic dermatitis triggers", "Skin"),
    "contact dermatitis": ("contact allergen", "Skin"),
    "photodermatitis": ("UV light", "Skin"), "urticaria": ("urticaria trigger", "Skin"),
}

_MED_SECTION_HEADERS = ("discharge medications", "medications", "current medications", "treatment given",
                        "medications used", "emergency medication prescribed")


INTAKE_SOURCE = "Patient intake"


def patient_key(name: str, age: Optional[int]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-{age}" if age is not None else slug


def is_none(value: str) -> bool:
    v = value.strip().lower().rstrip(".")
    return not v or v in _NONE_WORDS or v.startswith(("none", "no ", "not applicable", "nil"))


def normalise_severity(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("anaphyla", "life-threatening", "critical")):
        return "CRITICAL"
    if any(w in t for w in ("severe", "systemic")):
        return "SEVERE"
    if "moderate" in t:
        return "MODERATE"
    if "mild" in t:
        return "MILD"
    return "UNKNOWN"


def _first(fields: Dict[str, str], *keys: str) -> str:
    lowered = {k.lower(): v for k, v in fields.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return ""


def allergen_for(doc: ClinicalDocument) -> tuple[str, str]:
    name = doc.filename.lower()
    for keyword, value in ALLERGEN_FROM_FILENAME.items():
        if keyword in name:
            return value
    return doc.title.lower(), "Unknown"


def extract_allergy(doc: ClinicalDocument, section: Section) -> Optional[AllergyRecord]:
    """Return an AllergyRecord when the section documents a *positive* allergy, else None."""
    allergen, category = allergen_for(doc)
    f = section.fields
    title = section.title.lower()
    severity_text = _first(f, "Reaction Severity", "Severity Level", "Severity")
    reactions = _first(f, "Symptoms", "Reaction Type", "Rash Type", "Rash", "Reaction")
    specific_agent = _first(f, "Allergen / Contact Agent", "Allergen", "Contact Agent")
    if category == "Skin" and specific_agent and not is_none(specific_agent):
        allergen = specific_agent

    positive = False
    if "abnormal" in title or "positive" in title:
        positive = True
    elif "normal" in title:
        positive = False
    elif severity_text and not is_none(severity_text):
        positive = True
    elif reactions and not is_none(reactions):
        positive = True
    else:
        positive = any(r.abnormal and "ige" in r.parameter.lower() and "total" not in r.parameter.lower()
                       for r in section.lab_results)
    if not positive:
        return None

    severity = normalise_severity(severity_text or reactions or "")
    if severity == "UNKNOWN" and "anaphyla" in section.text.lower():
        severity = "CRITICAL"
    evidence = [r.line() for r in section.lab_results if r.abnormal][:6]
    return AllergyRecord(
        allergen=allergen, category=category, severity=severity,
        reactions=[p.strip() for p in re.split(r"[,;]", reactions) if p.strip() and not is_none(p)],
        emergency_medication=_first(f, "Emergency Medication Prescribed", "Emergency Medication"),
        restrictions=_first(f, "Prescribing Restrictions", "Dietary Restrictions", "Dietary Restriction"),
        alternatives=_first(f, "Alternative Medications"),
        evidence=evidence, source=doc.filename,
    )


def extract_medications(section: Section) -> List[str]:
    meds: List[str] = []
    lines = section.text.splitlines()
    capture = False
    for line in lines:
        low = line.lower().strip()
        if any(low.startswith(h) for h in _MED_SECTION_HEADERS) and low.endswith(":"):
            capture = True
            continue
        if capture:
            if low.endswith(":") or not low:
                capture = False
                continue
            if len(line) < 80:
                meds.append(line.strip("•- ").strip())
    for key in ("Current Medications", "Medications Used", "Medication Status", "Medications"):
        val = section.fields.get(key, "")
        if val and not is_none(val) and "no " not in val.lower()[:3]:
            meds.extend(v.strip() for v in re.split(r"[,;]", val) if v.strip())
    seen, out = set(), []
    for m in meds:
        if m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
    return out


def extract_conditions(section: Section) -> List[str]:
    conditions: List[str] = []
    for key in ("Diagnosis", "Impression / Diagnosis", "Impression", "Diabetes Diagnosis Status"):
        val = section.fields.get(key)
        if val and not is_none(val):
            conditions.append(val)
    for line in section.text.splitlines():
        if line.lower().startswith("findings consistent with"):
            conditions.append(line)
        if line.lower().startswith("summary:") and "normal" not in line.lower():
            conditions.append(line.split(":", 1)[1].strip())
    return conditions[:6]


class PatientRegistry:
    def __init__(self) -> None:
        self.patients: Dict[str, PatientProfile] = {}

    def build(self, documents: Iterable[ClinicalDocument]) -> "PatientRegistry":
        for doc in documents:
            self.add_document(doc)
        return self

    def add_document(self, doc: ClinicalDocument) -> List[str]:
        """Merge one document's patient sections into the registry; returns the affected patient keys."""
        touched: List[str] = []
        for section in doc.sections:
            if not section.patient_key or section.patient_key not in self.patients:
                continue
            profile = self.patients[section.patient_key]
            if section.section_id in profile.section_ids:
                continue
            if section.patient_key not in touched:
                touched.append(section.patient_key)
            profile.section_ids.append(section.section_id)
            if doc.filename not in profile.sources:
                profile.sources.append(doc.filename)
            if doc.domain not in profile.domains:
                profile.domains.append(doc.domain)
            if doc.domain == Domain.ALLERGY:
                record = extract_allergy(doc, section)
                if record:
                    profile.allergies.append(record)
                else:
                    profile.negative_allergy_tests.append(allergen_for(doc)[0])
            if doc.domain == Domain.RADIOLOGY:
                profile.imaging_studies.append(section.title)
            for med in extract_medications(section):
                if med not in profile.medications:
                    profile.medications.append(med)
            for cond in extract_conditions(section):
                if cond not in profile.conditions:
                    profile.conditions.append(cond)
            profile.abnormal_labs.extend(r for r in section.lab_results if r.abnormal and doc.domain == Domain.LAB)
        return touched

    def upsert_patient(self, name: str, age: Optional[int] = None, gender: Optional[str] = None,
                       patient_key_hint: Optional[str] = None) -> str:
        """Create (or update demographics of) a patient, e.g. from intake or an upload for a new patient."""
        if patient_key_hint and patient_key_hint in self.patients:
            profile = self.patients[patient_key_hint]
            profile.age = profile.age if profile.age is not None else age
            profile.gender = profile.gender or gender
            return patient_key_hint
        return self.register(name.strip(), age, gender, [])

    def register(self, name: str, age: Optional[int], gender: Optional[str], identifiers: List[str]) -> str:
        key = patient_key(name, age)
        if key not in self.patients:
            # fall back to a name-only match only when it is unambiguous
            same_name = [p for p in self.patients.values() if p.name == name]
            if age is None and len(same_name) == 1:
                key = same_name[0].patient_key
            else:
                self.patients[key] = PatientProfile(patient_key=key, name=name, age=age, gender=gender)
        profile = self.patients[key]
        profile.gender = profile.gender or gender
        for ident in identifiers:
            if ident not in profile.identifiers:
                profile.identifiers.append(ident)
        return key

    def get(self, key: Optional[str]) -> Optional[PatientProfile]:
        return self.patients.get(key) if key else None

    def all(self) -> List[PatientProfile]:
        return sorted(self.patients.values(), key=lambda p: p.name)

    def search(self, text: str) -> List[PatientProfile]:
        low = text.lower()
        return [p for p in self.patients.values() if p.name.lower() in low]
