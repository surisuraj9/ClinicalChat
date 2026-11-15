from clinicalchat.router import Router


def test_ibuprofen_contraindicated_for_ibuprofen_allergy(engine):
    alerts = engine.safety_alerts("sneha-gupta-42", ["ibuprofen"])
    assert alerts and alerts[0].level == "CONTRAINDICATED"


def test_nsaid_cross_reactivity_and_alternatives(engine):
    facts = engine.allergy_facts("sneha-gupta-42", ["ibuprofen"], [])
    assert "ALERT [CONTRAINDICATED]" in facts and "paracetamol" in facts


def test_sulfa_cross_reactivity(engine):
    facts = engine.allergy_facts("riya-das-27", [], ["sulfamethoxazole"])
    assert "Sulfonamide antibiotic" in facts and "celecoxib" in facts
    assert facts.count("Cross-reactivity for") == 1  # deduplicated


def test_no_patient_drug_check_is_explicit(engine):
    facts = engine.allergy_facts(None, ["amoxicillin"], [])
    assert "no patient selected" in facts


def test_safety_banner_always_lists_documented_allergies(orch, engine):
    plan = Router(engine).route("Explain my results", "patient", own_patient="riya-das-27")
    banner = orch.safety_banner(plan)
    assert "Documented allergies" in banner and "sulfamethoxazole" in banner.lower()


def test_lab_context_is_focused_on_requested_panel(engine):
    context, sources = engine.lab_context("Explain my thyroid results", "rahul-mehta-32")
    assert sources == ["Thyroid function test sample zz.docx"] and "TSH" in context
