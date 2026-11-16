"""Case workflow across the care team.

Patient case ──► 🩻 Radiologist (only when the case has imaging nobody has read yet)
             ──► 🧭 Care Coordinator (sets priority, assigns a clinician)
             ──► 👩‍⚕️ Clinician (confirms the AI assessment, or asks the patient for more information)

Allergy reports do not go to the radiologist; the radiologist only sees allergies to judge contrast safety.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from clinicalchat.domain.clock import local_minute
from clinicalchat.domain.risk import highest
from clinicalchat.domain.roles import Role

RADIOLOGY = "AWAITING_RADIOLOGY"
TRIAGE = "AWAITING_TRIAGE"
CLINICIAN = "PENDING_REVIEW"
RETURNED = "RETURNED"
VALIDATED = "VALIDATED"
MERGED = "MERGED"

OPEN_STATUSES = (RADIOLOGY, TRIAGE, CLINICIAN, RETURNED)

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
CRITICAL_VERDICT = RADIOLOGY_VERDICTS[2]
REVIEW_CONFIDENCE = 0.6  # below this the AI insight is flagged "Review Required"
HISTORY_LIMIT, TIMELINE_LIMIT = 20, 40


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


# ---------------------------------------------------------------------- case transitions (pure: input case -> new values)
def timeline_event(actor: str, text: str) -> Dict[str, str]:
    return {"at": local_minute(), "actor": actor, "event": text}


def routed_event(status: str) -> Dict[str, str]:
    return timeline_event("System", f"Routed: {STAGE_LABEL[status][0]}")


def require_open(status: str) -> None:
    if status not in OPEN_STATUSES:
        raise ValueError(f"Unknown status {status}")


def opening_payload(payload: Dict[str, Any], status: str) -> Dict[str, Any]:
    require_open(status)
    return {**payload, "timeline": [timeline_event("Patient", "Case opened — AI assessment shared with the patient"),
                                     routed_event(status)]}


def patient_update(old: Dict[str, Any], *, question: str, risk_level: str, escalate: bool, review_required: bool,
                   payload: Dict[str, Any], patient_draft: str, status: Optional[str] = None) -> Dict[str, Any]:
    """New question / reports for an open case: keep ONE case per patient. The latest assessment replaces the draft;
    the question history, radiology read, triage and timeline are kept, and the risk level never drops until a
    clinician has looked at it. Returns the new column values."""
    status = status or old["status"]
    require_open(status)
    history = list(old["payload"].get("history") or [{"question": old["question"], "at": old["created"]}])
    history.append({"question": question, "at": local_minute()})
    keep = {k: old["payload"][k] for k in ("radiology", "triage", "timeline") if k in old["payload"]}
    timeline = list(keep.get("timeline") or [])
    timeline.append(timeline_event("Patient", "Added a message / reports — AI assessment updated"))
    if status != old["status"]:
        timeline.append(routed_event(status))
    return {"question": question, "risk_level": highest([old["risk_level"], risk_level]),
            "escalate": escalate or old["escalate"], "review_required": review_required or old["review_required"],
            "payload": {**payload, **keep, "history": history[-HISTORY_LIMIT:], "timeline": timeline[-TIMELINE_LIMIT:]},
            "patient_draft": patient_draft, "status": status, "reviewer": None, "review_note": None,
            "final_message": None, "delivered": False, "reviewed": None}


def radiology_read(case: Dict[str, Any], *, reader: str, verdict: str, report: str,
                   doc_ids: List[str]) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """Returns (payload, next status, raised risk level or None)."""
    if case["status"] != RADIOLOGY:
        raise ValueError(f"Case {case['case_id']} is not waiting for a radiologist")
    if verdict not in RADIOLOGY_VERDICTS:
        raise ValueError(f"Unknown verdict {verdict}")
    critical = verdict == CRITICAL_VERDICT
    pay = case["payload"]
    reads = list((pay.get("radiology") or {}).get("reads") or [])
    reads.append({"reader": reader, "verdict": verdict, "report": report, "doc_ids": doc_ids, "at": local_minute()})
    pay["radiology"] = {"reads": reads, "doc_ids": sorted({d for r in reads for d in r["doc_ids"]})}
    status = status_after_radiology(case, critical)
    pay.setdefault("timeline", []).extend([timeline_event(reader, f"Radiology read: {verdict}"), routed_event(status)])
    return pay, status, "CRITICAL" if critical else None


def triage(case: Dict[str, Any], *, coordinator: str, priority: str, clinician: str, clinician_name: str,
           note: str) -> Dict[str, Any]:
    if case["status"] != TRIAGE:
        raise ValueError(f"Case {case['case_id']} is not waiting for triage")
    if priority not in PRIORITIES:
        raise ValueError(f"Unknown priority {priority}")
    pay = case["payload"]
    pay["triage"] = {"coordinator": coordinator, "priority": priority, "clinician": clinician,
                     "clinician_name": clinician_name, "note": note, "at": local_minute()}
    pay.setdefault("timeline", []).append(
        timeline_event(coordinator, f"Triaged {PRIORITY_ICON[priority]} {priority} → assigned to {clinician_name}"))
    return pay


def clinician_review(case: Dict[str, Any], *, reviewer: str, approve: bool, note: str,
                     edited_message: str = "") -> Dict[str, Any]:
    """The clinician confirms (VALIDATED) or asks the patient for more (RETURNED). Returns the new column values."""
    if case["status"] != CLINICIAN:
        raise ValueError(f"Case {case['case_id']} is not waiting for a clinician")
    note = note.strip()
    final = (edited_message.strip() or case["patient_draft"]) if approve else ""
    if approve and note:
        final += f"\n\n**Note from your clinician:** {note}"
    pay = case["payload"]
    pay.setdefault("timeline", []).append(timeline_event(
        reviewer, "Confirmed the assessment — case closed" if approve else "Asked the patient for more information"))
    return {"status": VALIDATED if approve else RETURNED, "reviewer": reviewer, "review_note": note,
            "final_message": final, "reviewed": local_minute(), "delivered": False, "payload": pay}


def attach_insight(payload: Dict[str, Any], insight: Optional[Dict[str, Any]], error: str = "") -> None:
    """The background Insight agent finished: add its clinical summary to the case payload (or the failure)."""
    payload["insight_pending"] = False
    if insight is None:
        payload["insight_error"] = error[:300]
        return
    payload["insight"], payload["insight_error"] = insight, ""
    if insight.get("possible_conditions"):
        payload["possible_conditions"] = insight["possible_conditions"]
    conf = insight.get("confidence")
    if (isinstance(conf, (int, float)) and conf < REVIEW_CONFIDENCE
            and not any("Low confidence" in r for r in payload.get("review_reasons", []))):
        payload.setdefault("review_reasons", []).append(f"Low confidence ({conf:.0%})")
    payload.setdefault("timeline", []).append(timeline_event("System", "💡 AI clinical summary added for the care team"))


def merge_duplicates(open_cases: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """Older duplicate open cases (newest first in the list) are folded into the newest: returns its payload + risk."""
    keep, older = open_cases[0], open_cases[1:]
    history = [{"question": c["question"], "at": c["created"]} for c in reversed(older)]
    history += keep["payload"].get("history") or [{"question": keep["question"], "at": keep["created"]}]
    return {**keep["payload"], "history": history[-HISTORY_LIMIT:]}, highest(c["risk_level"] for c in open_cases)


def stage_label(case: Dict[str, Any]) -> Tuple[str, str]:
    """(label, colour) of where the case is now — names the assigned clinician when it is with them."""
    label, color = STAGE_LABEL.get(case["status"], (case["status"], "#6b7280"))
    name = (case["payload"].get("triage") or {}).get("clinician_name")
    if case["status"] == CLINICIAN and name:
        label = f"⏳ With {name}"
    return label, color
