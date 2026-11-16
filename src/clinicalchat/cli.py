"""Command line: `clinicalchat serve | ingest | ask | reset-patient`."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clinicalchat", description="Role-aware multi-agent clinical chatbot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="Run the web app (login + chat)")
    ing = sub.add_parser("ingest", help="Parse the dataset and (re)build the vector index")
    ing.add_argument("--rebuild", action="store_true", help="Ignore caches and re-parse everything")
    ask = sub.add_parser("ask", help="Ask one question from the terminal")
    ask.add_argument("question")
    ask.add_argument("--user", default="dr.sharma", help="Persona username (decides role and patient access)")
    ask.add_argument("--patient", default=None, help="Patient key, e.g. ananya-rao-54 (staff roles only)")
    reset = sub.add_parser("reset-patient", help="Delete a patient's uploads, cases and history (stop the server first)")
    reset.add_argument("patient", help="Patient key (e.g. robin-hood-31) or the patient's login username")
    reset.add_argument("--delete-login", action="store_true", help="Also delete the patient's login")
    reset.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    return {"serve": _serve, "ingest": _ingest, "ask": _ask, "reset-patient": _reset_patient}[args.cmd](args)


def _serve(_args) -> int:
    from clinicalchat.web.app import serve

    serve()
    return 0


def _ingest(args) -> int:
    from clinicalchat.config import get_settings
    from clinicalchat.infrastructure.vector_index import VectorIndex
    from clinicalchat.ingestion.corpus import corpus_stats, load_corpus
    from clinicalchat.services.knowledge_base import KnowledgeBase

    settings = get_settings()
    docs, registry = load_corpus(settings, rebuild=args.rebuild)
    index = VectorIndex(settings)
    if args.rebuild:
        index.index(docs, force=True)
    kb = KnowledgeBase(settings, docs, registry, index)
    print(json.dumps({**corpus_stats(docs, registry), **kb.stats()}, indent=2))
    return 0


def _ask(args) -> int:
    from clinicalchat.container import get_container
    from clinicalchat.domain.conversation import ChatSession

    container = get_container()
    persona = container.accounts.get_persona(args.user)
    if not persona:
        print(f"Unknown user {args.user}", file=sys.stderr)
        return 2
    session = ChatSession(username=persona.username, role=persona.role, display_name=persona.display_name,
                          own_patient=persona.patient_key, selected_patient=persona.patient_key or args.patient)
    for ev in container.chat.ask(args.question, session):
        if ev.kind == "status":
            print(f"[{'..' if ev.status == 'pending' else 'ok'}] {ev.title}: {ev.text}", file=sys.stderr)
        elif ev.kind != "partial":
            print(f"\n=== {ev.title} ===\n{ev.text}")
    return 0


def _server_running(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _reset_patient(args) -> int:
    from clinicalchat.config import get_settings

    settings = get_settings()
    if _server_running(settings.port):
        print(f"The server is running on port {settings.port}. Stop it first: it keeps patient data in memory "
              "and would write it back.", file=sys.stderr)
        return 2
    admin = _admin(settings)
    user = admin.find_user(args.patient)
    key = user["patient_key"] if user else args.patient
    counts = admin.summary(key)
    if not user and not any(counts.values()):
        print(f"Nothing stored for '{args.patient}'.", file=sys.stderr)
        return 1
    who = f"{user['display_name']} ({key})" if user else key
    print(f"{who}: {counts['uploads']} upload(s), {counts['cases']} case(s)"
          + ("; the login will be DELETED" if args.delete_login else "; the login is kept"))
    if not args.yes and input("Delete? [y/N] ").strip().lower() != "y":
        return 1
    print(json.dumps(admin.purge(key, delete_login=args.delete_login)))
    return 0


def _admin(settings):
    """Only the database, vector index and export folder — no need to load the dataset for a purge."""
    from clinicalchat.container import Repositories
    from clinicalchat.infrastructure.database import Database
    from clinicalchat.infrastructure.vector_index import VectorIndex
    from clinicalchat.services.admin import PatientDataAdmin
    from clinicalchat.services.export import SessionExporter

    r = Repositories.on(Database(settings.database_path))
    return PatientDataAdmin(r.users, r.uploads, r.facts, r.cases, r.feedback, r.insights, VectorIndex(settings),
                            SessionExporter(None, settings.export_dir))


if __name__ == "__main__":
    raise SystemExit(main())
