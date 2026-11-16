"""Page layout: places every Gradio component and connects its events to the UIController (no logic here)."""

from __future__ import annotations

import gradio as gr

from clinicalchat.domain import workflow as wf
from clinicalchat.domain.personas import PERSONAS, clinicians
from clinicalchat.domain.roles import policy
from clinicalchat.ingestion.uploads import SUPPORTED_UPLOADS
from clinicalchat.web import components
from clinicalchat.web.content import DEFAULT_PLACEHOLDER, N_CHIPS
from clinicalchat.web.controller import UIController, attach_files, update_attachments
from clinicalchat.web.controller import DEFAULT_USER
from clinicalchat.web.theme import FORCE_LIGHT_JS

POLL_SECONDS = 10
SECONDARY = "cc-sec"


def build_ui(ui: UIController, llm_label: str, auth_enabled: bool = True) -> gr.Blocks:
    with gr.Blocks(title="ClinicalChat", js=FORCE_LIGHT_JS, fill_width=True) as demo:
        session = gr.State(None)

        # ---------------- header
        with gr.Row(elem_classes="cc-top", equal_height=True):
            gr.HTML(components.brand_html(llm_label), scale=3, padding=False)
            persona_pick = gr.Dropdown([(f"{p.display_name} — {policy(p.role).label}", p.username) for p in PERSONAS.values()],
                                       value=DEFAULT_USER, label="Dev persona", visible=not auth_enabled, scale=2, container=False)
            user_box = gr.HTML(scale=3, padding=False)
            gr.Button("Log out", link="/logout", size="sm", scale=0, min_width=90, visible=auth_enabled)

        with gr.Row(elem_classes="cc-main", equal_height=False):
            # ---------------- sidebar tiles
            with gr.Column(scale=0, min_width=340, elem_classes="cc-side"):
                with gr.Column(elem_classes="cc-tile"):
                    patient_title = gr.HTML("<div class='cc-h'>Patient</div>", padding=False)
                    patient_pick = gr.Dropdown(show_label=False, filterable=True, visible=False, interactive=True, container=False)
                    card = gr.HTML(padding=False)
                with gr.Column(elem_classes="cc-tile", visible=False) as review_tile:
                    rev_title = gr.HTML(components.review_title(0), padding=False)
                    board = gr.HTML(padding=False, visible=False)
                    review_pick = gr.Dropdown(show_label=False, choices=[], value=None, interactive=True, container=False,
                                              filterable=False, elem_id="cc-queue")
                    verdict_pick = gr.Dropdown(list(wf.RADIOLOGY_VERDICTS), value=wf.RADIOLOGY_VERDICTS[0], show_label=False,
                                               container=False, interactive=True, visible=False, filterable=False,
                                               elem_id="cc-verdict")
                    with gr.Row(elem_classes="cc-qrow", visible=False) as triage_row:
                        priority_pick = gr.Dropdown([(f"{wf.PRIORITY_ICON[x]} {x}", x) for x in wf.PRIORITIES], value="Soon",
                                                    show_label=False, container=False, interactive=True, filterable=False,
                                                    min_width=100, elem_id="cc-priority")
                        assign_pick = gr.Dropdown([(p.display_name, p.username) for p in clinicians()],
                                                  value=DEFAULT_USER, show_label=False, container=False, interactive=True,
                                                  filterable=False, min_width=140, elem_id="cc-assign")
                    review_note = gr.Textbox(show_label=False, placeholder="Note to patient / feedback for the agents (optional)",
                                             lines=2, max_lines=3, container=False, elem_id="cc-note")
                    with gr.Row(elem_classes="cc-rev-btns"):
                        approve_btn = gr.Button("✅ Approve & send", variant="primary", size="sm", elem_id="cc-act")
                        return_btn = gr.Button("↩️ Return", size="sm", elem_classes=SECONDARY, elem_id="cc-return")
                with gr.Column(elem_classes="cc-tile", visible=False) as cases_tile:
                    gr.HTML("<div class='cc-h'>My case updates</div>", padding=False)
                    cases_box = gr.HTML(padding=False)
                with gr.Column(elem_classes="cc-tile"):
                    gr.HTML("<div class='cc-h'>🛡️ Allergy Safety Check</div>", padding=False)
                    with gr.Row(elem_classes="cc-qrow"):
                        allergen_box = gr.Textbox(placeholder="Allergen, e.g. peanut, penicillin, latex", show_label=False,
                                                  container=False, scale=3, elem_id="cc-allergen")
                        allergen_btn = gr.Button("Check", size="sm", scale=0, min_width=80, elem_classes=SECONDARY)
                with gr.Column(elem_classes="cc-tile"):
                    gr.HTML("<div class='cc-h'>Try asking</div>", padding=False)
                    chips = [gr.Button("", elem_classes="cc-chip", size="sm", visible=False) for _ in range(N_CHIPS)]
            # ---------------- chat: agent strip + conversation + input
            with gr.Column(scale=1, elem_classes="cc-chatcol"):
                strip = gr.HTML(components.agent_strip(None), padding=False)
                chatbot = gr.Chatbot(elem_id="cc-chat", height="calc(100vh - 274px)", show_label=False, render_markdown=True,
                                     group_consecutive_messages=False,
                                     placeholder="Describe your symptoms, attach reports or ask a health question…")
                attached = gr.State([])
                with gr.Row(elem_classes="cc-input"):
                    msg = gr.Textbox(placeholder=DEFAULT_PLACEHOLDER, show_label=False, scale=8, autofocus=True, container=False)
                    attach_chip = gr.Dropdown(choices=[], value=[], multiselect=True, show_label=False, container=False,
                                              interactive=True, scale=4, min_width=160, elem_classes="cc-attach",
                                              visible=False, filterable=False)
                    send = gr.Button("Send ➤", variant="primary", scale=0, min_width=116)
                    upload_btn = gr.UploadButton("📎 Attach", file_types=list(SUPPORTED_UPLOADS), file_count="multiple",
                                                 scale=0, min_width=116, elem_classes=SECONDARY, visible=False)
                    export_btn = gr.DownloadButton("⬇️ Export", value=None, interactive=False, scale=0, min_width=116,
                                                   elem_classes=SECONDARY)
                    clear_btn = gr.Button("🧹 New chat", scale=0, min_width=116, elem_classes=SECONDARY)
        timer = gr.Timer(POLL_SECONDS)

        # ---------------- session bootstrap (identity from the authenticated username, server-side)
        load_outputs = [session, user_box, patient_title, patient_pick, card, strip, chatbot, export_btn, upload_btn,
                        review_tile, rev_title, review_pick, cases_tile, cases_box, *chips,
                        board, verdict_pick, triage_row, review_note, approve_btn, return_btn]

        def on_load(request: gr.Request):
            return ui.start(getattr(request, "username", None) if auth_enabled else DEFAULT_USER)

        demo.load(on_load, outputs=load_outputs)
        if not auth_enabled:
            persona_pick.change(ui.start, inputs=persona_pick, outputs=load_outputs)
        patient_pick.change(ui.pick_patient, [patient_pick, session], [session, card])

        # ---------------- chat
        post = [export_btn, patient_pick, card, cases_box, rev_title, review_pick]
        outputs = [chatbot, msg, send, strip]
        send_inputs = [msg, chatbot, session, attached]
        send_outputs = [*outputs, attached, attach_chip, msg]
        msg.submit(ui.respond_with_files, send_inputs, send_outputs, concurrency_limit=4).then(ui.after_turn, session, post)
        send.click(ui.respond_with_files, send_inputs, send_outputs, concurrency_limit=4).then(ui.after_turn, session, post)
        for chip in chips:
            chip.click(ui.respond, [chip, chatbot, session], outputs, concurrency_limit=4).then(ui.after_turn, session, post)
        upload_btn.upload(attach_files, [upload_btn, attached, session], [attached, attach_chip, msg])
        attach_chip.change(update_attachments, [attach_chip, attached], [attached, attach_chip, msg], show_progress="hidden")
        clear_btn.click(ui.clear, session, [chatbot, export_btn, strip])
        allergen_btn.click(ui.allergy_check, [allergen_box, chatbot, session], [chatbot, allergen_box])
        allergen_box.submit(ui.allergy_check, [allergen_box, chatbot, session], [chatbot, allergen_box])

        # ---------------- care-team work queues (radiologist → coordinator → clinician)
        review_pick.change(ui.open_case, [review_pick, chatbot, session], [chatbot, patient_pick, card, priority_pick, verdict_pick])
        rev_inputs = [review_pick, review_note, verdict_pick, priority_pick, assign_pick, chatbot, session]
        rev_outputs = [chatbot, rev_title, review_pick, review_note, board]
        approve_btn.click(ui.decide(True), rev_inputs, rev_outputs)
        return_btn.click(ui.decide(False), rev_inputs, rev_outputs)

        # ---------------- polling: patients receive reviewed answers, staff see new cases
        timer.tick(ui.poll, [session, chatbot], [chatbot, cases_box, rev_title, review_pick, board], show_progress="hidden")
    return demo
