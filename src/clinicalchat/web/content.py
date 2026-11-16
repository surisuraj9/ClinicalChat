"""Fixed texts of the UI: example questions, welcome messages, the login-page persona table and queue labels."""

from __future__ import annotations

import html
import os
from typing import Dict, List

from clinicalchat.domain.personas import PERSONAS, Persona
from clinicalchat.domain.roles import AgentName, Role, policy
from clinicalchat.web.theme import LOGO_URL

N_CHIPS = 5
EXAMPLES: Dict[Role, List[str]] = {
    Role.PATIENT: [
        "I have fever and cough since 3 days",
        "Explain my latest report in simple terms",
        "Is it safe for me to take ibuprofen?",
        "What is HbA1c?",
        "What is the status of my case?",
    ],
    Role.CLINICIAN: [
        "Interpret this patient's lab results.",
        "Check for drug allergies before prescribing ibuprofen.",
        "What follow-up tests are recommended based on these results?",
        "Explain this to the patient in simple terms.",
    ],
    Role.RADIOLOGIST: [
        "Review the chest CT findings.",
        "Is iodinated contrast safe for this patient?",
        "Give a differential diagnosis for the imaging findings.",
        "What follow-up imaging is recommended?",
    ],
    Role.COORDINATOR: [
        "Summarize this patient's care status for handoff.",
        "What follow-up tests are recommended based on these results?",
        "Which specialists should be involved in this patient's care?",
        "Check for drug allergies before prescribing ibuprofen.",
        "Where is this patient's case now?",
    ],
}
DEFAULT_PATIENT = {Role.CLINICIAN: "ananya-rao-54", Role.RADIOLOGIST: "meera-nair-47", Role.COORDINATOR: "john-doe-54"}
DEFAULT_PLACEHOLDER = "Type a message and press Enter · 📎 Attach to add reports"
ATTACHED_PLACEHOLDER = "Add a message (optional) and press Send"

QUEUE_TITLE = {Role.CLINICIAN: "Review queue", Role.RADIOLOGIST: "Imaging queue", Role.COORDINATOR: "Triage queue"}
QUEUE_EMPTY = {Role.CLINICIAN: "✓ No cases assigned to you", Role.RADIOLOGIST: "✓ No scans waiting",
               Role.COORDINATOR: "✓ Nothing to triage"}
QUEUE_PICK = {Role.CLINICIAN: "— Select a case to review —", Role.RADIOLOGIST: "— Select a case to read —",
              Role.COORDINATOR: "— Select a case to triage —"}
NOTE_PLACEHOLDER = {Role.CLINICIAN: "Note to patient / feedback for the agents (optional)",
                    Role.RADIOLOGIST: "Your read: findings / impression (required unless you agree with the AI)",
                    Role.COORDINATOR: "Hand-off note for the clinician (optional)"}
ACTION_LABEL = {Role.CLINICIAN: "✅ Approve & send", Role.RADIOLOGIST: "🩻 Submit read", Role.COORDINATOR: "🧭 Route to clinician"}

STATE_TEXT = {"idle": "Ready", "locked": "🔒 Not for your role", "queued": "Queued…", "skipped": "Not needed",
              "working": "⏳ Working…", "done": "✓ Done", "failed": "✕ Failed", "denied": "🔒 Not permitted"}
AGENT_KIND = {AgentName.EXTRACTION: "STEP 1 · RULES", AgentName.ANALYSIS: "STEP 2 · RULES + KG", AgentName.RISK: "STEP 3 · RULES",
              AgentName.INSIGHT: "STEP 4 · AI", AgentName.COMMUNICATION: "STEP 5 · AI"}

WELCOME = {
    Role.PATIENT: (
        "Tell me how you are feeling (e.g. *“I have fever and cough since 3 days”*), attach up to 10 reports with **📎 Attach** and send them with a message, "
        "or ask any health question.\n\nFive agents work on your message: **📄 Data Extraction** reads your records, "
        "**🔬 Clinical Analysis** checks values, scans, allergies and symptoms, **🚨 Risk Detection** tells you right "
        "away if you need urgent care, **💡 Insight Generation** prepares a summary for your clinician and "
        "**💬 Patient Communication** explains it in plain language right away.\n\n"
        "📁 A **case** for the care team is opened only when there is something new to review (new reports or "
        "symptoms), when you ask for a doctor (*“please ask my doctor to review this”*), or when your results "
        "need urgent attention. Questions about reports the care team already has are simply answered."),
    Role.CLINICIAN: (
        "Pick a patient on the left (or name one in your question). Each question runs the 5-agent pipeline. "
        "Patient cases **assigned to you by the care coordinator** (with the radiologist's read, when there are "
        "scans) are in your **Review queue** — confirm them or ask the patient for more information."),
    Role.RADIOLOGIST: (
        "Patient cases with **scans** come to your **Imaging queue** first. Open one, read the imaging (with the "
        "allergy / contrast check), choose a verdict and **Submit read** — the case then goes to the care "
        "coordinator. You can also pick a patient and ask imaging questions here (lab results are not shown "
        "for your role)."),
    Role.COORDINATOR: (
        "Every patient case comes to your **Triage queue** (after the radiologist, when there are scans). Open "
        "one, set the **priority** and **assign a clinician** — the board shows where all open cases are. You "
        "can also pick a patient and ask for a hand-off summary here."),
}


def welcome(persona: Persona) -> List[dict]:
    first = persona.display_name.split()[-1] if persona.role != Role.PATIENT else persona.display_name.split()[0]
    greet = f"Hello {('Dr. ' + first) if persona.display_name.startswith('Dr.') else first} 👋"
    return [{"role": "assistant", "content": f"{greet}\n\n{WELCOME[persona.role]}"}]


def login_message() -> str:
    rows = "".join(
        f"<tr><td style='padding:3px 10px;font-family:monospace'>{p.username}</td>"
        f"<td style='padding:3px 10px'>{html.escape(p.display_name)}</td>"
        f"<td style='padding:3px 10px;color:#4338ca'>{policy(p.role).label}</td></tr>"
        for p in PERSONAS.values()
    )
    pwd = "the demo password" if os.getenv("DEMO_PASSWORD") else "<b>demo@123</b>"
    return (
        "<div style='text-align:center;font-family:system-ui,sans-serif'>"
        f"<img src='{LOGO_URL}' alt='' style='width:56px;height:56px;margin:4px auto 6px;display:block'/>"
        "<div style='font-size:22px;font-weight:700;color:#111827'>ClinicalChat</div>"
        "<div style='font-size:13px;color:#6b7280;margin-bottom:10px'>Role-aware multi-agent clinical assistant</div>"
        "<table style='margin:0 auto;font-size:13px;border-collapse:collapse;text-align:left'>"
        f"<tr style='color:#6b7280'><th style='padding:3px 10px'>Username</th><th style='padding:3px 10px'>Name</th>"
        f"<th style='padding:3px 10px'>Role</th></tr>{rows}</table>"
        f"<div style='font-size:12.5px;color:#6b7280;margin-top:8px'>Password for all demo accounts: {pwd}</div>"
        "<div style='font-size:13.5px;margin-top:10px;padding:8px 10px;background:#eef2ff;border-radius:9px;color:#3730a3'>"
        "New patient? <a href='/signup' style='color:#4338ca;font-weight:700'>Create an account</a> to describe your "
        "symptoms and upload your reports.</div></div>"
    )
