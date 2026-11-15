import pytest

from clinicalchat.rbac import AgentName as A
from clinicalchat.router import Router

# the 8 example queries from the problem statement (+ the role they are typically asked by)
CASES = [
    ("Interpret my CBC results.", "patient", None, "ananya-rao-54", [A.LAB]),
    ("Analyze this patient's complete metabolic panel.", "clinician", "rahul-mehta-32", None, [A.LAB]),
    ("Review the chest CT findings.", "radiologist", "meera-nair-47", None, [A.RADIOLOGY]),
    ("Check for drug allergies before prescribing ibuprofen.", "clinician", "sneha-gupta-42", None, [A.ALLERGY]),
    ("What are the cross-reactivities for this sulfa allergy?", "patient", None, "riya-das-27", [A.ALLERGY]),
    ("Summarize this patient's care status for handoff.", "coordinator", "john-doe-54", None, [A.COORDINATOR]),
    ("What follow-up tests are recommended based on these results?", "clinician", "ananya-rao-54", None, [A.LAB]),
    ("Explain my thyroid results in simple terms.", "patient", None, "rahul-mehta-32", [A.LAB]),
]


@pytest.mark.parametrize("question,role,selected,own,expected", CASES)
def test_problem_statement_queries(engine, question, role, selected, own, expected):
    plan = Router(engine).route(question, role, session_patient=selected, own_patient=own)
    assert not plan.blocked
    assert plan.agents == expected
    assert plan.patient_key == (own or selected)


def test_patient_cannot_ask_about_someone_else(engine):
    plan = Router(engine).route("Show Ananya Rao's CBC", "patient", own_patient="rahul-mehta-32")
    assert plan.blocked and not plan.agents


def test_patient_cannot_request_handoff(engine):
    plan = Router(engine).route("Summarize this patient's care status for handoff.", "patient", own_patient="rahul-mehta-32")
    assert plan.blocked and A.COORDINATOR in plan.denied


def test_radiologist_has_no_lab_agent(engine):
    plan = Router(engine).route("Interpret the CBC", "radiologist", session_patient="rahul-mehta-32")
    assert plan.blocked and plan.denied == [A.LAB]


def test_coordinator_gets_specialist_questions_as_summary(engine):
    plan = Router(engine).route("Interpret the CBC", "coordinator", session_patient="rahul-mehta-32")
    assert plan.agents == [A.COORDINATOR]


def test_staff_can_name_patient_in_question(engine):
    plan = Router(engine).route("Review Meera Nair's CT chest", "radiologist")
    assert plan.patient_key == "meera-nair-47" and plan.patient_source == "question"


def test_follow_up_reuses_previous_specialists(engine):
    plan = Router(engine).route("And what does that mean?", "clinician", session_patient="meera-nair-47",
                                previous_agents=[A.RADIOLOGY])
    assert plan.agents == [A.RADIOLOGY]


def test_drug_question_triggers_allergy_agent(engine):
    plan = Router(engine).route("Is it safe for me to take ibuprofen?", "patient", own_patient="rahul-mehta-32")
    assert plan.agents == [A.ALLERGY] and plan.drugs == ["ibuprofen"]
