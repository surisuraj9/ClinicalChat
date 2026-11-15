import json

from clinicalchat.agents import InsightOutput, check_patient_text, parse_output
from clinicalchat.export import session_markdown
from clinicalchat.orchestrator import ChatSession
from clinicalchat.rbac import Role


def _session(role, own=None, selected=None, username="tester"):
    return ChatSession(username=username, role=role, own_patient=own, selected_patient=selected)


def _agent_order(events):
    keys = []
    for e in events:
        if e.kind == "status" and e.key.startswith("agent-") and e.status == "done" and e.key not in keys:
            keys.append(e.key)
    return [k.removeprefix("agent-") for k in keys]


def test_staff_question_runs_pipeline_with_one_llm_call(orch, fake_llm):
    events = orch.run("Interpret this patient's CBC results.", _session(Role.CLINICIAN, selected="ananya-rao-54"))
    assert events[0].kind == "status" and events[0].status == "pending"  # user sees progress immediately
    assert _agent_order(events) == ["extraction", "analysis", "risk", "insight"]
    assert len(fake_llm.calls) == 1 and "INSIGHT GENERATION" in fake_llm.calls[0]
    assert "LAB INTERPRETER" in fake_llm.calls[0] and "Hemoglobin" in fake_llm.calls[0]
    answer = next(e for e in events if e.kind == "answer")
    assert "Review Required" in answer.text  # multiple potential conditions -> escalated for validation


def test_patient_case_is_held_for_clinician_review(orch, fake_llm, store):
    s = _session(Role.PATIENT, own="ananya-rao-54", username="ananya.rao")
    events = orch.run("Explain my CBC report", s)
    assert _agent_order(events) == ["extraction", "analysis", "risk", "insight", "communication"]
    assert len(fake_llm.calls) == 2 and "PATIENT COMMUNICATION" in fake_llm.calls[1]
    assert not any(e.kind == "answer" for e in events)  # the AI draft is NOT shown before validation
    case_ev = next(e for e in events if e.kind == "case")
    case = store.case(case_ev.data["case_id"])
    assert case["status"] == "PENDING_REVIEW" and "COMMUNICATION answer" in case["patient_draft"]
    assert store.undelivered("ananya-rao-54") == []

    orch.review(case["case_id"], "Dr. Neha Sharma", approve=True, note="Iron studies booked.")
    delivered = store.undelivered("ananya-rao-54")
    assert len(delivered) == 1 and "Iron studies booked." in delivered[0]["final_message"]
    assert store.recent_feedback(1)[0]["note"] == "Iron studies booked."
    # learnability: the next insight prompt includes the clinician's feedback
    orch.run("Interpret this patient's CBC results.", _session(Role.CLINICIAN, selected="ananya-rao-54"))
    assert "Iron studies booked." in fake_llm.calls[-1]


def test_new_symptoms_saved_and_red_flag_guidance_is_immediate(orch, fake_llm, engine, store):
    key = engine.new_patient_key("Test Chestpain", 50)
    engine.create_patient(key, "Test Chestpain", 50, "Male")
    events = orch.run("I have chest pain since an hour", _session(Role.PATIENT, own=key))
    kinds = [e.kind for e in events]
    guidance = next(e for e in events if e.kind == "guidance")
    assert "emergency" in guidance.text.lower() and kinds.index("guidance") < kinds.index("case")
    assert "chest pain" in engine.kg.patient_symptoms(key)
    assert [f["value"] for f in store.facts(key)] == ["chest pain"]
    risk = next(e for e in events if e.kind == "step" and e.data.get("agent") == "risk")
    assert risk.data["level"] == "CRITICAL"


def test_general_question_is_one_call_and_answered_directly(orch, fake_llm):
    events = orch.run("What is HbA1c?", _session(Role.PATIENT, own="rahul-mehta-32"))
    assert len(fake_llm.calls) == 1 and "general health question" in fake_llm.calls[0]
    assert any(e.kind == "answer" for e in events) and not any(e.kind == "case" for e in events)


def test_safety_shown_before_answer(orch, fake_llm):
    events = orch.run("Check for drug allergies before prescribing ibuprofen.", _session(Role.CLINICIAN, selected="sneha-gupta-42"))
    kinds = [e.kind for e in events]
    assert kinds.index("safety") < kinds.index("answer")
    assert "CONTRAINDICATED" in next(e for e in events if e.kind == "safety").text
    assert "ALERT [CONTRAINDICATED]" in fake_llm.calls[0]


def test_blocked_request_makes_no_llm_call(orch, fake_llm):
    events = orch.run("Show Ananya Rao's CBC", _session(Role.PATIENT, own="rahul-mehta-32"))
    assert any(e.kind == "blocked" for e in events) and not fake_llm.calls


def test_clinician_explain_follow_up_uses_one_call(orch, fake_llm):
    s = _session(Role.CLINICIAN)
    orch.run("Interpret Ananya Rao's CBC", s)  # patient named in the question
    events = orch.run("Explain this to the patient in simple terms", s)
    assert len(fake_llm.calls) == 2 and "PATIENT COMMUNICATION" in fake_llm.calls[1]
    answer = next(e for e in events if e.kind == "answer")
    assert "Guard check" in answer.text


def test_upload_pipeline(orch, fake_llm, engine, store, tmp_path):
    key = engine.new_patient_key("Upload Tester", 35)
    engine.create_patient(key, "Upload Tester", 35, "Male")
    report = tmp_path / "my_cbc.txt"
    report.write_text("Complete Blood Count\nPatient: Upload Tester  Age: 35\nDate: 10/11/2026\n"
                      "Hemoglobin 9.2 g/dL 13.0-17.0\nWBC 14500 /uL 4000-11000\nPlatelets 250000 /uL 150000-450000\n")
    s = _session(Role.PATIENT, own=key)
    events = list(orch.upload(report, s, display_name="my_cbc.txt"))
    step = next(e for e in events if e.kind == "step" and e.data.get("agent") == "extraction")
    assert "3 lab value(s)" in step.text and "lab" in step.text
    analysis = next(e for e in events if e.kind == "step" and e.data.get("agent") == "analysis")
    assert "Hemoglobin" in analysis.text
    assert store.uploads(key) and engine.patient(key).sources == ["my_cbc.txt"]
    assert any(e.kind == "case" for e in events) and len(fake_llm.calls) == 2


def test_radiologist_cannot_upload(orch, fake_llm, tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    events = list(orch.upload(f, _session(Role.RADIOLOGIST, selected="meera-nair-47")))
    assert events[-1].kind == "blocked" and not fake_llm.calls


def test_audit_log_written(orch, fake_llm):
    orch.run("Review the chest CT findings.", _session(Role.RADIOLOGIST, selected="meera-nair-47"))
    row = [json.loads(line) for line in orch.auditor.path.read_text().splitlines()][-1]
    assert row["role"] == "radiologist" and row["llm_calls"] == 1 and row["risk_level"] == "MODERATE"
    assert row["agents"] == ["extraction", "analysis", "risk", "insight"]


def test_conversation_memory_and_export(orch, fake_llm, engine):
    s = _session(Role.CLINICIAN, selected="ananya-rao-54")
    orch.run("Analyze this patient's complete metabolic panel.", s)
    orch.run("What follow-up tests are recommended based on these results?", s)
    assert "CONVERSATION SO FAR:\nQ: Analyze this patient's complete metabolic panel." in fake_llm.calls[-1]
    md = session_markdown(s, engine)
    assert "complete metabolic panel" in md and "Ananya Rao" in md and "not a substitute" in md.lower()


def test_parse_output_is_lenient():
    out, ok = parse_output('Sure!\n{"answer": "x", "key_findings": "Hb low", "next_steps": [{"a": 1}], "confidence": "75%"}',
                           InsightOutput, "answer")
    assert ok and out.key_findings == ["Hb low"] and out.next_steps == ["1"] and out.confidence == 0.75
    out, ok = parse_output("plain text reply", InsightOutput, "answer")
    assert not ok and out.answer == "plain text reply"


def test_patient_text_guard():
    assert check_patient_text(["Take 400 mg ibuprofen twice daily. Ask your doctor."], False)
    assert any("definitive" in i for i in check_patient_text(["You have cancer. Talk to your doctor."], True))
    assert any("Over-reassuring" in i for i in check_patient_text(["Nothing to worry about, see your doctor."], True))
    assert check_patient_text(["Your haemoglobin is a little low. Please discuss it with your doctor."], True) == []
