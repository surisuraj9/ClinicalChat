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
    answer = next(e for e in events if e.kind == "answer")  # AI assessment shown straight away, with the case id
    assert "Anaemia" in answer.text and "not a diagnosis" in answer.text and "COMMUNICATION answer" in answer.text
    case_ev = next(e for e in events if e.kind == "case")
    assert answer.data["case_id"] == case_ev.data["case_id"] and case_ev.data["case_id"] in case_ev.text
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


def test_one_open_case_per_patient_is_updated_not_duplicated(orch, fake_llm, store):
    s = _session(Role.PATIENT, own="rahul-mehta-32", username="rahul.mehta")
    first = next(e for e in orch.run("Explain my thyroid results", s) if e.kind == "case")
    second = next(e for e in orch.run("Based on my reports what is my current health condition?", s) if e.kind == "case")
    assert first.data["case_id"] == second.data["case_id"] and second.data["updated"] and "updated" in second.text
    cases = store.cases(patient_key="rahul-mehta-32")
    assert len(cases) == 1 and [h["question"] for h in cases[0]["payload"]["history"]] == [
        "Explain my thyroid results", "Based on my reports what is my current health condition?"]
    orch.review(first.data["case_id"], "Dr. Neha Sharma", approve=True)  # closed -> the next question opens a new case
    third = next(e for e in orch.run("I have a headache since yesterday", s) if e.kind == "case")
    assert third.data["case_id"] != first.data["case_id"] and not third.data["updated"]


def test_duplicate_open_cases_are_consolidated(store):
    kw = dict(created_by="x", escalate=False, review_required=False, payload={}, patient_draft="d")
    a = store.create_case(patient_key="p-1", question="q1", risk_level="HIGH", **kw)
    b = store.create_case(patient_key="p-1", question="q2", risk_level="LOW", **kw)
    assert store.consolidate_open_cases() == 1
    open_ = [c for c in store.cases(patient_key="p-1") if c["status"] == "PENDING_REVIEW"]
    assert len(open_) == 1 and open_[0]["risk_level"] == "HIGH"
    assert {open_[0]["case_id"]} < {a, b} and len(open_[0]["payload"]["history"]) == 2
    assert store.undelivered("p-1") == []  # merged case is not pushed to the patient


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


def test_multi_upload_runs_pipeline_once(orch, fake_llm, engine, store, tmp_path):
    key = engine.new_patient_key("Multi Uploader", 40)
    engine.create_patient(key, "Multi Uploader", 40, "Female")
    cbc = tmp_path / "cbc.txt"
    cbc.write_text("Complete Blood Count\nPatient Name: Multi Uploader\nDate: 10/11/2026\n"
                   "Hemoglobin 9.0 g/dL 12.0-15.0\nWBC 15000 /uL 4000-11000\nPlatelets 260000 /uL 150000-450000\n")
    xray = tmp_path / "xray.txt"
    xray.write_text("X-RAY CHEST PA VIEW\nPatient Name: Multi Uploader\nDate: 11/11/2026\nFindings:\n"
                    "Patchy consolidation in the right lower zone. No pleural effusion.\n"
                    "Impression:\nRight lower lobe consolidation, likely pneumonia.\n")
    other = tmp_path / "other.txt"
    other.write_text("Complete Blood Count\nPatient Name: Someone Else\nHemoglobin 8.0 g/dL 13.0-17.0\n")
    bad = tmp_path / "notes.csv"
    bad.write_text("a,b")
    files = [(cbc, "cbc.txt"), (xray, "xray.txt"), (cbc, "cbc copy.txt"), (other, "other.txt"), (bad, "notes.csv")]
    events = list(orch.upload_many(files, _session(Role.PATIENT, own=key)))
    step = next(e for e in events if e.kind == "step" and e.data.get("agent") == "extraction")
    assert "**2 of 5**" in step.text
    assert "already on your record" in step.text and "Someone Else" in step.text and "could not be read" in step.text
    assert sorted(u["filename"] for u in store.uploads(key)) == ["cbc.txt", "xray.txt"]
    analysis = next(e for e in events if e.kind == "step" and e.data.get("agent") == "analysis")
    assert "Lab Interpreter" in analysis.text and "Radiology Analyzer" in analysis.text
    assert len(fake_llm.calls) == 2  # one pipeline for the whole batch
    assert sum(e.kind == "case" for e in events) == 1


def test_upload_with_message_answers_it_in_one_pipeline(orch, fake_llm, engine, store, tmp_path):
    key = engine.new_patient_key("Message Uploader", 33)
    engine.create_patient(key, "Message Uploader", 33, "Male")
    cbc = tmp_path / "cbc.txt"
    cbc.write_text("Complete Blood Count\nPatient Name: Message Uploader\nDate: 10/11/2026\n"
                   "Hemoglobin 11.0 g/dL 13.0-17.0\nWBC 13000 /uL 4000-11000\nPlatelets 250000 /uL 150000-450000\n")
    msg = "I have fever and cough since 4 days. Here is my blood test — can I take ibuprofen?"
    events = list(orch.upload_many([(cbc, "cbc.txt")], _session(Role.PATIENT, own=key), message=msg))
    route = next(e for e in events if e.kind == "status" and e.key == "route" and e.status == "done")
    assert route.data["plan"]["symptoms"] and "ibuprofen" in route.data["plan"]["drugs"]
    assert {"fever", "cough"} <= set(engine.kg.patient_symptoms(key))
    assert len(fake_llm.calls) == 2 and "can I take ibuprofen" in fake_llm.calls[0] and "Hemoglobin" in fake_llm.calls[0]
    assert any(e.kind == "safety" and "ibuprofen" in e.text.lower() for e in events)
    case = store.case(next(e for e in events if e.kind == "case").data["case_id"])
    assert case["question"].startswith("I have fever")


def test_upload_message_about_another_patient_is_blocked(orch, fake_llm, engine, tmp_path):
    key = engine.new_patient_key("Nosy Person", 30)
    engine.create_patient(key, "Nosy Person", 30, "Male")
    f = tmp_path / "r.txt"
    f.write_text("Lab\nPatient Name: Nosy Person\nHemoglobin 14 g/dL 13-17\n")
    events = list(orch.upload_many([(f, "r.txt")], _session(Role.PATIENT, own=key), message="Show Ananya Rao's CBC"))
    assert events[-1].kind == "blocked" and not fake_llm.calls


def test_upload_batch_limits(orch, fake_llm, engine, tmp_path):
    key = engine.new_patient_key("Limit Tester", 30)
    engine.create_patient(key, "Limit Tester", 30, "Male")
    s = _session(Role.PATIENT, own=key)
    f = tmp_path / "r.txt"
    f.write_text("x")
    too_many = list(orch.upload_many([(f, f"r{i}.txt") for i in range(11)], s))
    assert too_many[-1].kind == "blocked" and "at most" in too_many[-1].text
    other = tmp_path / "o.txt"
    other.write_text("Lab report\nPatient Name: Not Me\nHemoglobin 12 g/dL 13-17\n")
    none_ok = list(orch.upload_many([(other, "o.txt")], s))
    assert none_ok[-1].kind == "blocked" and not fake_llm.calls


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
