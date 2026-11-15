import json

from clinicalchat.agents import LabOutput, parse_output
from clinicalchat.export import session_markdown
from clinicalchat.orchestrator import ChatSession
from clinicalchat.rbac import Role


def _session(role, own=None, selected=None):
    return ChatSession(username="tester", role=role, own_patient=own, selected_patient=selected)


def test_one_llm_call_per_agent_and_live_status(orch, fake_llm):
    events = orch.run("Interpret my CBC results.", _session(Role.PATIENT, own="ananya-rao-54"))
    kinds = [e.kind for e in events]
    assert len(fake_llm.calls) == 1
    assert kinds[0] == "status" and events[0].status == "pending"  # user sees progress immediately
    assert "answer" in kinds
    assert "AUDIENCE: PATIENT" in fake_llm.calls[0] and "CBC" in fake_llm.calls[0]


def test_safety_shown_before_answers(orch, fake_llm):
    events = orch.run("Check for drug allergies before prescribing ibuprofen.", _session(Role.CLINICIAN, selected="sneha-gupta-42"))
    kinds = [e.kind for e in events]
    assert kinds.index("safety") < kinds.index("answer")
    safety = next(e for e in events if e.kind == "safety")
    assert "CONTRAINDICATED" in safety.text
    assert len(fake_llm.calls) == 1 and "ALERT [CONTRAINDICATED]" in fake_llm.calls[0]


def test_blocked_request_makes_no_llm_call(orch, fake_llm):
    events = orch.run("Show Ananya Rao's CBC", _session(Role.PATIENT, own="rahul-mehta-32"))
    assert any(e.kind == "blocked" for e in events) and not fake_llm.calls


def test_missing_data_short_circuits_without_llm(orch, fake_llm):
    events = orch.run("Interpret my CBC results.", _session(Role.PATIENT, own="riya-das-27"))
    answer = next(e for e in events if e.kind == "answer")
    assert "no lab reports" in answer.text.lower() and not fake_llm.calls


def test_audit_log_written(orch, fake_llm):
    s = _session(Role.RADIOLOGIST, selected="meera-nair-47")
    orch.run("Review the chest CT findings.", s)
    rows = [json.loads(l) for l in orch.auditor.path.read_text().splitlines()]
    assert rows[-1]["agents"] == ["radiology"] and rows[-1]["role"] == "radiologist" and rows[-1]["llm_calls"] == 1


def test_conversation_memory_and_export(orch, fake_llm, engine):
    s = _session(Role.CLINICIAN, selected="ananya-rao-54")
    orch.run("Analyze this patient's complete metabolic panel.", s)
    orch.run("What follow-up tests are recommended based on these results?", s)
    assert "CONVERSATION SO FAR:\nQ: Analyze this patient's complete metabolic panel." in fake_llm.calls[-1]
    md = session_markdown(s, engine)
    assert "complete metabolic panel" in md and "Ananya Rao" in md and "not a substitute" in md.lower()


def test_parse_output_is_lenient():
    out, ok = parse_output('Sure!\n{"answer": "x", "abnormal_values": "Hb low", "key_findings": [{"a": 1}]}', LabOutput)
    assert ok and out.abnormal_values == ["Hb low"] and out.key_findings == ["1"]
    out, ok = parse_output("plain text reply", LabOutput)
    assert not ok and out.answer == "plain text reply"
