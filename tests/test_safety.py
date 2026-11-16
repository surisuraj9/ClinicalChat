from clinicalchat.domain.conversation import ChatSession, Turn
from clinicalchat.domain.roles import Role
from clinicalchat.services.router import Router
from clinicalchat.services.turn import PipelineRun, TurnContext


def test_ibuprofen_contraindicated_for_ibuprofen_allergy(kb):
    alerts = kb.safety.alerts("sneha-gupta-42", ["ibuprofen"])
    assert alerts and alerts[0].level == "CONTRAINDICATED"


def test_nsaid_cross_reactivity_and_alternatives(kb):
    facts = kb.safety.allergy_facts("sneha-gupta-42", ["ibuprofen"], [])
    assert "ALERT [CONTRAINDICATED]" in facts and "paracetamol" in facts


def test_sulfa_cross_reactivity(kb):
    facts = kb.safety.allergy_facts("riya-das-27", [], ["sulfamethoxazole"])
    assert "Sulfonamide antibiotic" in facts and "celecoxib" in facts
    assert facts.count("Cross-reactivity for") == 1  # deduplicated


def test_no_patient_drug_check_is_explicit(kb):
    facts = kb.safety.allergy_facts(None, ["amoxicillin"], [])
    assert "no patient selected" in facts


def test_safety_banner_lists_documented_allergies_for_allergy_questions(container, kb):
    plan = Router(kb.patients, kb.kg).route("What are the cross-reactivities for my sulfa allergy?", "patient", own_patient="riya-das-27")
    q = "What are the cross-reactivities for my sulfa allergy?"
    ctx = TurnContext(q, ChatSession("riya", Role.PATIENT, own_patient="riya-das-27"), plan,
                      Turn(q, "patient", "riya-das-27"), PipelineRun(), 0.0)
    banner = container.pipeline.safety_banner(ctx)
    assert "Documented allergies" in banner and "sulfamethoxazole" in banner.lower()
