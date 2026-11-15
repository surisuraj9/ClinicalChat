import pytest

from clinicalchat.models import Domain
from clinicalchat.rbac import AgentName as A
from clinicalchat.router import Router

STAFF_PIPELINE = [A.EXTRACTION, A.ANALYSIS, A.RISK, A.INSIGHT]
FULL_PIPELINE = [*STAFF_PIPELINE, A.COMMUNICATION]

# the 8 example queries from the problem statement (+ the role they are typically asked by)
CASES = [
    ("Interpret my CBC results.", "patient", None, "ananya-rao-54", [Domain.LAB]),
    ("Analyze this patient's complete metabolic panel.", "clinician", "rahul-mehta-32", None, [Domain.LAB]),
    ("Review the chest CT findings.", "radiologist", "meera-nair-47", None, [Domain.RADIOLOGY]),
    ("Check for drug allergies before prescribing ibuprofen.", "clinician", "sneha-gupta-42", None, [Domain.ALLERGY]),
    ("What are the cross-reactivities for this sulfa allergy?", "patient", None, "riya-das-27", [Domain.ALLERGY]),
    ("Summarize this patient's care status for handoff.", "coordinator", "john-doe-54", None, []),
    ("What follow-up tests are recommended based on these results?", "clinician", "ananya-rao-54", None, []),
    ("Explain my thyroid results in simple terms.", "patient", None, "rahul-mehta-32", [Domain.LAB]),
]


@pytest.mark.parametrize("question,role,selected,own,focus", CASES)
def test_problem_statement_queries(engine, question, role, selected, own, focus):
    plan = Router(engine).route(question, role, session_patient=selected, own_patient=own)
    assert not plan.blocked
    assert plan.focus == focus
    assert plan.patient_key == (own or selected)
    assert plan.agents == (FULL_PIPELINE if role == "patient" and plan.mode == "case" else STAFF_PIPELINE) \
        or plan.mode == "general"


def test_patient_personal_question_is_a_case_general_question_is_not(engine):
    r = Router(engine)
    assert r.route("I have fever and cough since 3 days", "patient", own_patient="rahul-mehta-32").mode == "case"
    general = r.route("What is HbA1c?", "patient", own_patient="rahul-mehta-32")
    assert general.mode == "general" and general.agents == [A.COMMUNICATION]


def test_symptoms_detected_with_synonyms_and_negation(engine):
    plan = Router(engine).route("I've been throwing up and have a temperature, no cough", "patient", own_patient="rahul-mehta-32")
    assert "vomiting" in plan.symptoms and "fever" in plan.symptoms and "cough" not in plan.symptoms


def test_patient_cannot_ask_about_someone_else(engine):
    plan = Router(engine).route("Show Ananya Rao's CBC", "patient", own_patient="rahul-mehta-32")
    assert plan.blocked and not plan.agents


def test_radiologist_cannot_open_lab_records(engine):
    plan = Router(engine).route("Interpret the CBC", "radiologist", session_patient="rahul-mehta-32")
    assert plan.blocked and plan.denied_domains == [Domain.LAB]


def test_partial_denial_is_noted_not_blocked(engine):
    plan = Router(engine).route("Compare the CBC with the chest CT", "radiologist", session_patient="meera-nair-47")
    assert not plan.blocked and plan.focus == [Domain.RADIOLOGY] and plan.denied_domains == [Domain.LAB]


def test_staff_can_name_patient_in_question(engine):
    plan = Router(engine).route("Review Meera Nair's CT chest", "radiologist")
    assert plan.patient_key == "meera-nair-47" and plan.patient_source == "question"


def test_clinician_explain_reuses_last_insight(engine):
    r = Router(engine)
    plan = r.route("Explain this to the patient in simple terms", "clinician", session_patient="ananya-rao-54",
                   has_last_insight=True)
    assert plan.mode == "explain" and plan.agents == [A.COMMUNICATION]
    fresh = r.route("Explain this to the patient in simple terms", "clinician", session_patient="ananya-rao-54")
    assert fresh.agents == FULL_PIPELINE


def test_coordinator_has_no_patient_communication(engine):
    plan = Router(engine).route("Explain this to the patient", "coordinator", session_patient="john-doe-54")
    assert plan.mode == "case" and A.COMMUNICATION not in plan.agents


def test_drug_question_checks_allergies(engine):
    plan = Router(engine).route("Is it safe for me to take ibuprofen?", "patient", own_patient="rahul-mehta-32")
    assert plan.drugs == ["ibuprofen"] and Domain.ALLERGY in plan.focus and plan.check_allergies
