"""Case workflow across the care team.

Patient case ──► 🩻 Radiologist (only when the case has imaging nobody has read yet)
             ──► 🧭 Care Coordinator (sets priority, assigns a clinician)
             ──► 👩‍⚕️ Clinician (confirms the AI assessment, or asks the patient for more information)

Allergy reports do not go to the radiologist; the radiologist only sees allergies to judge contrast safety.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from clinicalchat.rbac import Role

RADIOLOGY = "AWAITING_RADIOLOGY"
TRIAGE = "AWAITING_TRIAGE"
CLINICIAN = "PENDING_REVIEW"
RETURNED = "RETURNED"
VALIDATED = "VALIDATED"
MERGED = "MERGED"

OPEN_STATUSES = (RADIOLOGY, TRIAGE, CLINICIAN, RETURNED)
CASE_STATUSES = (*OPEN_STATUSES, VALIDATED, MERGED)

# which queue each staff role works on
QUEUE_FOR_ROLE = {Role.RADIOLOGIST: RADIOLOGY, Role.COORDINATOR: TRIAGE, Role.CLINICIAN: CLINICIAN}

STAGE_LABEL = {
    RADIOLOGY: ("🩻 With radiologist", "#6d28d9"),
    TRIAGE: ("🧭 With care coordinator", "#0e7490"),
    CLINICIAN: ("⏳ With clinician", "#b45309"),
    RETURNED: ("↩️ More info needed", "#991b1b"),
    VALIDATED: ("✅ Confirmed", "#166534"),
}

PRIORITIES = ("Urgent", "Soon", "Routine")
PRIORITY_ICON = {"Urgent": "🔴", "Soon": "🟡", "Routine": "🟢"}
RADIOLOGY_VERDICTS = ("Agree with AI imaging findings", "Additional / different findings", "Critical finding — urgent")


def suggested_priority(risk_level: str) -> str:
    return {"CRITICAL": "Urgent", "HIGH": "Urgent", "MODERATE": "Soon"}.get(risk_level, "Routine")


def unread_imaging(imaging: Iterable[Dict[str, Any]], read_doc_ids: Iterable[str]) -> List[str]:
    read = set(read_doc_ids)
    return sorted({i["doc_id"] for i in imaging if i.get("doc_id") and i["doc_id"] not in read})


def initial_status(unread: List[str]) -> str:
    return RADIOLOGY if unread else TRIAGE


def status_after_patient_update(old: Dict[str, Any], unread: List[str]) -> str:
    """The patient added a message / reports to an open case."""
    if unread:
        return RADIOLOGY
    if old["status"] == RETURNED:  # the patient answered: back to whoever is handling it
        return CLINICIAN if old["payload"].get("triage") else TRIAGE
    return old["status"]


def status_after_radiology(case: Dict[str, Any], critical: bool) -> str:
    """Already triaged and nothing critical: straight back to the assigned clinician; otherwise (re-)triage."""
    return CLINICIAN if case["payload"].get("triage") and not critical else TRIAGE


def assigned_clinician(case: Dict[str, Any]) -> Optional[str]:
    return (case["payload"].get("triage") or {}).get("clinician")


def next_step_text(status: str, clinician_name: Optional[str] = None) -> str:
    return {
        RADIOLOGY: "A **radiologist** will first read your scan(s); then the **care coordinator** prioritises your case "
                   "and assigns a clinician.",
        TRIAGE: "The **care coordinator** will prioritise your case and assign a clinician.",
        CLINICIAN: f"**{clinician_name or 'A clinician'}** will confirm or correct this AI assessment.",
        RETURNED: "Your clinician has asked for more information.",
        VALIDATED: "Your clinician has confirmed this case.",
    }.get(status, "")


def stepper(case: Dict[str, Any]) -> List[tuple]:
    """[(label, state)] with state in done | now | todo | skip, for the patient's progress bar."""
    pay, status = case["payload"], case["status"]
    has_imaging = bool(pay.get("imaging")) or bool(pay.get("radiology"))
    order = [("AI assessment", "ai"), ("Radiologist", RADIOLOGY), ("Coordinator", TRIAGE), ("Clinician", CLINICIAN)]
    pos = {"ai": 0, RADIOLOGY: 1, TRIAGE: 2, CLINICIAN: 3, RETURNED: 3, VALIDATED: 4}.get(status, 3)
    out = []
    for i, (label, key) in enumerate(order):
        if key == RADIOLOGY and not has_imaging:
            out.append((label, "skip"))
        elif i < pos:
            out.append((label, "done"))
        elif i == pos:
            out.append((label, "now"))
        else:
            out.append((label, "todo"))
    return out
