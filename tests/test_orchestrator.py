import json

import pytest

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
    # no imaging -> straight to the care coordinator (not the radiologist)
    assert case["status"] == "AWAITING_TRIAGE" and "COMMUNICATION answer" in case["patient_draft"]
    assert store.undelivered("ananya-rao-54") == []
    with pytest.raises(ValueError):  # a clinician cannot confirm a case before it has been triaged
        orch.review(case["case_id"], "Dr. Neha Sharma", approve=True)

    orch.triage(case["case_id"], "Maria Thomas", "Soon", "dr.sharma")
    orch.review(case["case_id"], "Dr. Neha Sharma", approve=True, note="Iron studies booked.")
    delivered = store.undelivered("ananya-rao-54")
    assert len(delivered) == 1 and "Iron studies booked." in delivered[0]["final_message"]
    assert store.recent_feedback(1)[0]["note"] == "Iron studies booked."
    # learnability: the next insight prompt includes the clinician's feedback
    orch.run("Interpret this patient's CBC results.", _session(Role.CLINICIAN, selected="ananya-rao-54"))
    assert "Iron studies booked." in fake_llm.calls[-1]


def test_one_open_case_per_patient_is_updated_not_duplicated(orch, fake_llm, store):
    s = _session(Role.PATIENT, own="rahul-mehta-32", username="rahul.mehta")
    first = next(e for e in orch.run("Please ask my doctor to review my thyroid results", s) if e.kind == "case")
    second = next(e for e in orch.run("I have a headache since yesterday", s) if e.kind == "case")  # new symptom
    assert first.data["case_id"] == second.data["case_id"] and second.data["updated"] and "updated" in second.text
    cases = store.cases(patient_key="rahul-mehta-32")
    assert len(cases) == 1 and [h["question"] for h in cases[0]["payload"]["history"]] == [
        "Please ask my doctor to review my thyroid results", "I have a headache since yesterday"]
    orch.triage(first.data["case_id"], "Maria Thomas", "Routine", "dr.sharma")
    orch.review(first.data["case_id"], "Dr. Neha Sharma", approve=True)  # closed -> the next review request opens a new case
    third = next(e for e in orch.run("Can a doctor check my thyroid results again?", s) if e.kind == "case")
    assert third.data["case_id"] != first.data["case_id"] and not third.data["updated"]


def test_simple_doubt_is_answered_without_a_case_and_reuses_the_insight(orch, fake_llm, store):
    key = "priya-desai-44"  # MODERATE, nothing that escalates
    s = _session(Role.PATIENT, own=key, username="priya")
    events = orch.run("What do my reports mean?", s)
    assert not any(e.kind == "case" for e in events) and store.cases(patient_key=key) == []
    answer = next(e for e in events if e.kind == "answer")
    assert "No case was opened" in answer.text and len(fake_llm.calls) == 2  # Insight + Communication
    events = orch.run("Is that something serious for me?", s)  # nothing new -> Insight reused, 1 AI call
    assert len(fake_llm.calls) == 3 and "PATIENT COMMUNICATION" in fake_llm.calls[-1]
    assert any(e.kind == "status" and e.key == "agent-insight" and "reused" in e.title for e in events)
    assert not any(e.kind == "case" for e in events)
    case_id = next(e for e in orch.run("I have a headache since yesterday", s) if e.kind == "case").data["case_id"]
    assert len(fake_llm.calls) == 5  # new symptom = new evidence -> fresh Insight, and a case for the care team
    answer = next(e for e in orch.run("What do my reports mean?", s) if e.kind == "answer")
    assert case_id in answer.text and "did not change it" in answer.text and len(store.cases(patient_key=key)) == 1


def test_unreviewed_escalating_findings_open_a_case_once(orch, fake_llm, store):
    key = "amit-verma-46"  # HIGH risk in his records
    s = _session(Role.PATIENT, own=key, username="amit")
    case_id = next(e for e in orch.run("What do my reports mean?", s) if e.kind == "case").data["case_id"]
    if store.case(case_id)["status"] == "AWAITING_RADIOLOGY":
        orch.radiology_read(case_id, "Dr. Arjun Iyer", "Agree with AI imaging findings", "")
    orch.triage(case_id, "Maria Thomas", "Soon", "dr.sharma")
    orch.review(case_id, "Dr. Neha Sharma", approve=True)
    # same evidence already reviewed -> the next doubt is just answered
    assert not any(e.kind == "case" for e in orch.run("What do my reports mean?", s))
    assert len(store.cases(patient_key=key)) == 1


def test_duplicate_open_cases_are_consolidated(store):
    kw = dict(created_by="x", escalate=False, review_required=False, payload={}, patient_draft="d")
    a = store.create_case(patient_key="p-1", question="q1", risk_level="HIGH", **kw)
    b = store.create_case(patient_key="p-1", question="q2", risk_level="LOW", **kw)
    assert store.consolidate_open_cases() == 1
    open_ = [c for c in store.cases(patient_key="p-1") if c["status"] == "AWAITING_TRIAGE"]
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


def test_imaging_case_goes_radiologist_then_coordinator_then_assigned_clinician(orch, fake_llm, store):
    s = _session(Role.PATIENT, own="meera-nair-47", username="meera")
    case_id = next(e for e in orch.run("Please have a doctor review my CT scan report", s) if e.kind == "case").data["case_id"]
    case = store.case(case_id)
    assert case["status"] == "AWAITING_RADIOLOGY" and case["payload"]["imaging"]
    assert "contrast_alerts" in case["payload"] and "radiologist" in next(
        e for e in orch.run("what is the status of my case?", s) if e.kind == "answer").text.lower()
    with pytest.raises(ValueError):
        orch.triage(case_id, "Maria Thomas", "Soon", "dr.sharma")  # coordinator waits for the radiology read

    case = orch.radiology_read(case_id, "Dr. Arjun Iyer", "Agree with AI imaging findings", "")
    assert case["status"] == "AWAITING_TRIAGE" and case["payload"]["radiology"]["doc_ids"]
    with pytest.raises(ValueError):
        orch.triage(case_id, "Maria Thomas", "Soon", "dr.iyer")  # only clinicians can be assigned
    case = orch.triage(case_id, "Maria Thomas", "Urgent", "dr.menon", "Please see today")
    assert case["status"] == "PENDING_REVIEW" and case["payload"]["triage"]["clinician_name"] == "Dr. Kavya Menon"

    orch.review(case_id, "Dr. Kavya Menon", approve=False, note="How long have you had the headache?")
    assert store.case(case_id)["status"] == "RETURNED"
    # the patient answers -> same case goes straight back to the assigned clinician (imaging already read)
    again = next(e for e in orch.run("The headache started 3 days ago and is getting worse", s) if e.kind == "case")
    case = store.case(case_id)
    assert again.data["case_id"] == case_id and case["status"] == "PENDING_REVIEW"
    assert case["payload"]["triage"]["clinician"] == "dr.menon" and len(case["payload"]["radiology"]["reads"]) == 1
    assert [e["actor"] for e in case["payload"]["timeline"]][:2] == ["Patient", "System"]


def test_critical_radiology_read_raises_risk_and_goes_back_to_triage(orch, fake_llm, store):
    s = _session(Role.PATIENT, own="meera-nair-47", username="meera")
    case_id = next(e for e in orch.run("Please have a doctor review my CT scan report", s) if e.kind == "case").data["case_id"]
    case = orch.radiology_read(case_id, "Dr. Arjun Iyer", "Critical finding — urgent", "Acute bleed — needs urgent review")
    assert case["status"] == "AWAITING_TRIAGE" and case["risk_level"] == "CRITICAL"


def test_case_status_question_uses_no_llm(orch, fake_llm, store):
    s = _session(Role.PATIENT, own="rahul-mehta-32", username="rahul.mehta")
    events = orch.run("Has the doctor reviewed my case?", s)
    answer = next(e for e in events if e.kind == "answer")
    assert fake_llm.calls == [] and "don't have a case yet" in answer.text
    orch.run("Please ask my doctor to review my thyroid results", s)
    n = len(fake_llm.calls)
    answer = next(e for e in orch.run("what is the status of my case?", s) if e.kind == "answer")
    assert len(fake_llm.calls) == n and "care coordinator" in answer.text.lower()


def test_iodine_allergy_flags_contrast_for_radiologist(orch, fake_llm, engine, store):
    key = engine.new_patient_key("Iodine Allergic", 40)
    engine.create_patient(key, "Iodine Allergic", 40, "Female")
    engine.apply_fact(key, "allergy", "iodine")
    alerts = engine.safety_alerts(key, ["iodinated contrast"])
    assert alerts and alerts[0].level == "CONTRAINDICATED"


def test_case_is_filed_and_routed_even_when_the_ai_is_rate_limited(engine, store, tmp_path):
    from conftest import FakeLLM
    from clinicalchat.audit import Auditor
    from clinicalchat.config import Settings
    from clinicalchat.orchestrator import Orchestrator

    class RateLimited(FakeLLM):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("HTTP 429: rate limit exceeded")

    orch = Orchestrator(engine, llm=RateLimited(cache=False, calls=[]), auditor=Auditor(Settings(work_dir=tmp_path)),
                        store=store)
    s = _session(Role.PATIENT, own="meera-nair-47", username="meera")
    events = orch.run("Please have a doctor review my CT scan report", s)
    case_ev = next(e for e in events if e.kind == "case")
    answer = next(e for e in events if e.kind == "answer")
    assert "busy" in answer.text and "not a diagnosis" in answer.text and answer.data["case_id"] == case_ev.data["case_id"]
    case = store.case(case_ev.data["case_id"])
    # imaging still reaches the radiologist; the clinician has a rule-based draft to approve or edit
    assert case["status"] == "AWAITING_RADIOLOGY" and case["payload"]["insight"]["ai_unavailable"]
    assert "Overall risk" in case["patient_draft"] and "busy" not in case["patient_draft"]
    assert any("AI explanation unavailable" in r for r in case["payload"]["review_reasons"])
