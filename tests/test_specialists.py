"""The course AI specialists (Lab Interpreter / Radiology Analyzer / Allergy Safety): role-scoped, parallel,
structured (course Pydantic models), and only able to RAISE the rule-based risk."""

import pytest

from clinicalchat.agents.specialists.ai import LabInterpreterAgent, LabPanel
from clinicalchat.domain.conversation import ChatSession
from clinicalchat.domain.roles import Role
from clinicalchat.presenters.case_views import clinician_view, coordinator_view, radiologist_view
from tests.conftest import FakeLLM


def _session(role, own=None, selected=None, username="tester"):
    return ChatSession(username=username, role=role, own_patient=own, selected_patient=selected)


def _agents_called(llm):
    return [c.split("You are the ", 1)[1].split(" agent", 1)[0] for c in llm.calls]


@pytest.fixture()
def ai_container(make_container, fake_llm):
    return make_container(llm=fake_llm, specialist_agents="llm")


def test_specialists_are_off_without_the_switch(container, fake_llm):
    container.chat.run("Interpret this patient's CBC results.", _session(Role.CLINICIAN, selected="ananya-rao-54"))
    assert _agents_called(fake_llm) == ["INSIGHT GENERATION"]


def test_clinician_gets_lab_specialist_only_and_ai_urgency_raises_risk(ai_container, fake_llm):
    events = ai_container.chat.run("Interpret this patient's CBC results.",
                                   _session(Role.CLINICIAN, selected="ananya-rao-54"))
    called = _agents_called(fake_llm)
    assert called[0] == "LAB INTERPRETER" and called[-1] == "INSIGHT GENERATION"
    assert "RADIOLOGY ANALYZER" not in called  # RBAC: clinicians do not see imaging
    step = next(e for e in events if e.kind == "step" and e.data.get("agent") == "specialists")
    assert "Hemoglobin" in step.text and "conf 85%" in step.text  # 85 -> 0.85 (course schema wants 0..1)
    risk = next(e for e in events if e.kind == "step" and e.data.get("agent") == "risk")
    assert "AI flagged as stat" in risk.text and risk.data["level"] in ("HIGH", "CRITICAL")
    assert "AI LAB INTERPRETER: Hemoglobin" in fake_llm.calls[-1]  # the Insight sees the specialists' opinions


def test_radiologist_sends_only_imaging_to_the_ai(ai_container, fake_llm):
    ai_container.chat.run("Summarise the imaging findings", _session(Role.RADIOLOGIST, selected="rohan-sharma-45"))
    called = _agents_called(fake_llm)
    assert "RADIOLOGY ANALYZER" in called
    assert "LAB INTERPRETER" not in called and "ALLERGY SAFETY" not in called


def test_clinician_allergy_question_uses_allergy_specialist(ai_container, fake_llm):
    ai_container.chat.run("Check for drug allergies before prescribing ibuprofen.",
                          _session(Role.CLINICIAN, selected="sneha-gupta-42"))
    allergy_prompt = next(c for c in fake_llm.calls if "You are the ALLERGY SAFETY agent" in c)
    assert "DOCUMENTED ALLERGIES" in allergy_prompt and "KNOWLEDGE GRAPH" in allergy_prompt


def test_patient_never_waits_for_specialists_and_case_views_are_role_scoped(ai_container, fake_llm, repos):
    s = _session(Role.PATIENT, own="ananya-rao-54", username="ananya.rao")
    events = ai_container.chat.run("Explain my CBC report", s)
    # the patient's answer is the first AI call; the specialists only run afterwards, in the background
    assert _agents_called(fake_llm)[0] == "PATIENT COMMUNICATION"
    assert not any(e.kind == "step" and e.data.get("agent") == "specialists" for e in events)
    case_id = next(e for e in events if e.kind == "case").data["case_id"]
    ai_container.insight.drain()
    called = _agents_called(fake_llm)
    assert "LAB INTERPRETER" in called and called[-1] == "INSIGHT GENERATION"
    case = repos.cases.get(case_id)
    assert case["payload"]["ai_specialists"]["lab"]["interpretations"][0]["test_name"] == "Hemoglobin"
    assert any("AI specialist flagged" in r for r in case["payload"]["review_reasons"])  # stat -> Review Required
    assert "AI specialists" in clinician_view(case, "Ananya") and "Hemoglobin" in clinician_view(case, "Ananya")
    assert "AI specialists" in coordinator_view(case, "Ananya")
    assert "Hemoglobin" not in radiologist_view(case, "Ananya")  # radiologists never see lab opinions


def test_bad_specialist_reply_is_an_error_not_a_crash():
    class Garbage(FakeLLM):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content='{"interpretations": "oops"}'))])

    result = LabInterpreterAgent(Garbage(cache=False, calls=[])).run("- Hemoglobin: 9.1 g/dL -> LOW")
    assert result.error and not result.parsed
    assert LabPanel.model_validate(result.output).interpretations == []


def test_guideline_search_runs_alongside_the_ai_specialists(make_container, fake_llm):
    """The staff guideline web search starts once the rule-based findings are known — while the AI specialists work —
    and is searched only once."""
    import threading

    from tests.test_chat_service import _FakeWeb

    class SignallingWeb(_FakeWeb):
        def __init__(self, results):
            super().__init__(results)
            self.started = threading.Event()

        def search(self, query, kind, names=(), config=None):
            self.started.set()
            return super().search(query, kind, names, config)

    web = SignallingWeb({"guideline": [{"title": "KDIGO", "url": "https://kdigo.org/g", "content": "Treat K+."}]})
    container = make_container(llm=fake_llm, web=web, specialist_agents="llm")
    panel, seen = container.specialists, []
    run = panel.run
    panel.run = lambda *a, **k: (seen.append(web.started.wait(5)), run(*a, **k))[1]
    events = container.chat.run("What is the current treatment guideline for her condition?",
                                _session(Role.CLINICIAN, selected="ananya-rao-54"))
    assert seen == [True] and len(web.queries) == 1
    assert "[KDIGO](https://kdigo.org/g)" in next(e for e in events if e.kind == "answer").text
