"""Course alignment (Week 1–8): hybrid RBAC, knowledge-graph vocabulary, panic values, corpus versioning, REST API."""

import json

import pytest
from fastapi.testclient import TestClient

from clinicalchat.config import Settings
from clinicalchat.domain.conversation import ChatSession
from clinicalchat.domain.models import Domain, LabStatus
from clinicalchat.domain.roles import Role
from clinicalchat.ingestion import corpus
from clinicalchat.ingestion.lab_tables import compute_status
from clinicalchat.knowledge.graph import CONFIRMS, CROSS_REACTIVE, INDICATES, TREATS
from clinicalchat.services.router import Router
from clinicalchat.web.api import ApiSessions
from clinicalchat.web.app import create_app

PASSWORD = "demo@123"


def _session(role, own=None, selected=None, username="tester"):
    return ChatSession(username=username, role=role, own_patient=own, selected_patient=selected)


# ------------------------------------------------------------------ RBAC (hybrid course model)
@pytest.mark.parametrize("question,role,patient,denied", [
    ("Review the chest CT findings.", "clinician", "meera-nair-47", [Domain.RADIOLOGY]),
    ("Show this patient's CBC results.", "radiologist", "ananya-rao-54", [Domain.LAB]),
    ("What are this patient's documented allergies?", "radiologist", "sneha-gupta-42", [Domain.ALLERGY]),
])
def test_course_rbac_denies_out_of_scope_domains(kb, question, role, patient, denied):
    plan = Router(kb.patients, kb.kg).route(question, role, session_patient=patient)
    assert plan.denied_domains == denied


def test_clinician_keeps_labs_and_allergies_coordinator_sees_everything(kb):
    router = Router(kb.patients, kb.kg)
    assert not router.route("Check for drug allergies before prescribing ibuprofen.", "clinician",
                            session_patient="sneha-gupta-42").denied_domains
    assert not router.route("Review the chest CT findings.", "coordinator", session_patient="meera-nair-47").denied_domains


def test_radiologist_keeps_the_contrast_alert_without_the_allergy_record(container):
    events = container.chat.run("Is iodinated contrast safe for this patient?",
                                _session(Role.RADIOLOGIST, selected="ananya-rao-54"))
    safety = "\n".join(e.text for e in events if e.kind == "safety")
    assert "contrast" in safety.lower()
    assert "Documented allergies" not in safety


# ------------------------------------------------------------------ knowledge graph (course vocabulary)
def test_graph_uses_course_node_and_edge_types(kb):
    g = kb.kg.g
    types = {d["type"] for _, d in g.nodes(data=True)}
    assert {"TEST", "SYMPTOM", "DIAGNOSIS", "TREATMENT", "ALLERGEN"} <= types
    rels = {d["rel"] for _, _, d in g.edges(data=True)}
    assert {CONFIRMS, INDICATES, CROSS_REACTIVE, TREATS} <= rels


def test_course_graph_queries(kb):
    kg = kb.kg
    diagnoses = [d["diagnosis"] for d in kg.find_diagnoses_for_symptoms(["fever", "cough"])]
    assert "Pneumonia" in diagnoses and "Bronchitis" in diagnoses
    latex = kg.get_cross_reactive_allergens("latex")
    assert {"banana", "avocado", "kiwi", "chestnut"} <= set(latex)
    assert "ibuprofen" in kg.get_cross_reactive_allergens("aspirin")
    assert "Chest X-ray" in kg.tests_confirming("Pneumonia")
    assert "aspirin" in kg.treatments_for("Myocardial infarction")
    card = kg.allergy_safety_card("Latex")
    assert card["allergen"] == "latex" and card["safe_alternatives"]
    assert kg.allergy_safety_card("unobtainium")["allergen"] is None
    assert kg.to_mermaid().startswith("flowchart")


def test_aspirin_is_not_flagged_against_the_condition_it_treats(kb):
    alerts = kb.kg.check_drug_for_patient("aspirin", "john-doe-54")
    assert not any(a.against == "Myocardial infarction" for a in alerts)


# ------------------------------------------------------------------ course panic values + corpus versioning
@pytest.mark.parametrize("param,value,low,high,expected", [
    ("WBC count", 1.5, 4.0, 11.0, LabStatus.CRITICAL_LOW),        # 1,500 /uL in 10^3/uL
    ("WBC count", 1500, 4000, 11000, LabStatus.CRITICAL_LOW),     # same value in /uL
    ("WBC count", 32000, 4000, 11000, LabStatus.CRITICAL_HIGH),
    ("WBC count", 14.8, 4.0, 11.0, LabStatus.HIGH),
    ("Potassium", 6.8, 3.5, 5.1, LabStatus.CRITICAL_HIGH),
    ("Hemoglobin", 6.5, 12.0, 16.0, LabStatus.CRITICAL_LOW),
])
def test_course_panic_values(param, value, low, high, expected):
    assert compute_status(param, value, low, high) == expected


def test_outdated_corpus_cache_is_rebuilt(tmp_path, monkeypatch):
    settings = Settings(work_dir=tmp_path)
    settings.ensure_dirs()
    corpus.corpus_path(settings).write_text(json.dumps({"version": 1, "documents": [], "patients": []}))
    rebuilt = []
    monkeypatch.setattr(corpus, "ingest", lambda s: rebuilt.append(True) or ([], corpus.PatientRegistry()))
    corpus.load_corpus(settings)
    assert rebuilt and json.loads(corpus.corpus_path(settings).read_text())["version"] == corpus.CORPUS_VERSION


# ------------------------------------------------------------------ REST API (course Week 8)
@pytest.fixture()
def api(container):
    with TestClient(create_app(container)) as client:
        yield client


def test_api_requires_login_and_reports_status(api):
    assert api.get("/api/health").json() == {"status": "ok"}
    assert api.get("/api/status").status_code == 401
    assert api.get("/api/status", auth=("dr.sharma", "wrong")).status_code == 401
    status = api.get("/api/status", auth=("dr.sharma", PASSWORD)).json()
    assert status["role"] == "clinician" and status["knowledge_base"]["graph"]["nodes"] > 100
    assert status["specialist_agents"] == "rules"


def test_api_query_runs_the_agents_with_the_callers_role(api, fake_llm):
    body = api.post("/api/query", json={"query": "Interpret this patient's CBC results.", "patient": "ananya-rao-54"},
                    auth=("dr.sharma", PASSWORD)).json()
    assert body["role"] == "clinician" and body["patient"] == "ananya-rao-54"
    assert body["agents"] == ["extraction", "analysis", "risk", "insight"] and "INSIGHT answer" in body["answer"]
    follow = api.post("/api/query", json={"query": "Explain this to the patient in simple terms",
                                          "session_id": body["session_id"]}, auth=("dr.sharma", PASSWORD)).json()
    assert follow["session_id"] == body["session_id"] and follow["patient"] == "ananya-rao-54"


def test_api_patient_cannot_query_someone_else(api, fake_llm):
    body = api.post("/api/query", json={"query": "Show Ananya Rao's CBC", "patient": "ananya-rao-54"},
                    auth=("rahul.mehta", PASSWORD)).json()
    assert body["blocked"] and not fake_llm.calls


def test_api_allergy_card(api):
    card = api.get("/api/allergy/peanut", auth=("ananya.rao", PASSWORD)).json()
    assert card["allergen"] == "peanut" and card["cross_reactive_allergens"]


def test_api_sessions_are_bounded_and_per_user(container):
    sessions = ApiSessions(limit=2)
    sharma, menon = (container.accounts.get_persona(u) for u in ("dr.sharma", "dr.menon"))
    first = sessions.get(sharma, None, "ananya-rao-54")
    assert sessions.get(sharma, first.session_id, None) is first
    assert sessions.get(menon, first.session_id, None) is not first  # another user cannot reuse the session
    sessions.get(sharma, None, None)
    assert sessions.get(sharma, first.session_id, None) is not first  # evicted (least recently used)
