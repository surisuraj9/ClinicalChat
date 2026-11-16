from clinicalchat.domain.conversation import ChatSession, Event
from clinicalchat.domain.roles import Role
from clinicalchat.web.chat_renderer import render
from clinicalchat.web.controller import UIController, attach_files, update_attachments


def test_renderer_updates_findings_and_replaces_the_streamed_draft():
    s = ChatSession("p", Role.PATIENT, own_patient="rahul-mehta-32")
    events = [
        Event("status", "🧭 Understanding", "", "pending", "route"),
        Event("status", "🧭 Understanding", "", "done", "route", {"plan": {"agents": ["extraction", "communication"]}}),
        Event("step", "📄 Data Extraction", "2 reports"),
        Event("step", "🔬 Clinical Analysis", "all normal"),
        Event("partial", "💬 Answer", "Your results ▍"),
        Event("answer", "💬 Answer", "Your results are normal.", data={"agent": "communication"}),
    ]
    frames = list(render(iter(events), "hi", [], s))
    history, strip = frames[-1]
    texts = [m["content"] for m in history]
    assert texts[0] == "hi" and history[1]["metadata"]["status"] == "done"
    assert sum("Quick check" in t for t in texts) == 1 and "all normal" in texts[2]
    assert texts[-1].endswith("Your results are normal.") and not any("▍" in t for t in texts)
    assert "cc-agents" in strip and not s.busy


def test_renderer_reports_errors_instead_of_hanging():
    s = ChatSession("p", Role.PATIENT)

    def boom():
        yield Event("status", "start", "", "pending", "route")
        raise RuntimeError("kaput")

    history, _ = list(render(boom(), "hi", [], s))[-1]
    assert "kaput" in history[-1]["content"] and not s.busy


def test_attachment_chips_can_be_removed_one_by_one(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    current, _, _ = attach_files([str(a), str(b), str(a)], [])
    assert current == [str(a), str(b)]
    keep, _, _ = update_attachments([str(b)], current)
    assert keep == [str(b)]


def test_staff_queue_flow_through_the_controller(container):
    ui = UIController(container)
    patient = ChatSession("meera", Role.PATIENT, own_patient="meera-nair-47")
    case_id = next(e for e in container.chat.run("Please have a doctor review my CT scan report", patient)
                   if e.kind == "case").data["case_id"]

    radiologist = ui.start("dr.iyer")[0]
    assert case_id in [v for _, v in ui.sidebar.queue_choices(radiologist)]
    history, *_ = ui.open_case(case_id, [], radiologist)
    assert "Iodinated contrast check" in history[-1]["content"] and "Deterministic findings" not in history[-1]["content"]
    history, *_ = ui.decide(True)(case_id, "", "Agree with AI imaging findings", "Soon", "dr.sharma", [], radiologist)
    assert "Read submitted" in history[-1]["content"]

    coordinator = ui.start("coord.thomas")[0]
    history, *_ = ui.decide(True)(case_id, "", "", "Urgent", "dr.menon", [], coordinator)
    assert "Dr. Kavya Menon" in history[-1]["content"]
    assert ui.sidebar.queue_choices(ui.start("dr.sharma")[0]) == [("✓ No cases assigned to you", "")]

    menon = ui.start("dr.menon")[0]
    history, *_ = ui.decide(True)(case_id, "All clear.", "", "", "", [], menon)
    assert "approved and sent to" in history[-1]["content"]
    delivered = ui.deliver_updates(patient, [])
    assert delivered and "confirmed by Dr. Kavya Menon" in delivered[-1]["content"]
    assert ui.deliver_updates(patient, []) is None  # delivered only once
