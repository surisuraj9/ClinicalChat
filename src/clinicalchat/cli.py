"""Command line: `clinicalchat serve | ingest | ask | reset-patient`."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv=None) -> int:
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.cmd == "serve":
        from clinicalchat.app import serve

        serve()
        return 0
    if args.cmd == "ingest":
        from clinicalchat.config import get_settings
        from clinicalchat.engine import ClinicalEngine
        from clinicalchat.ingestion.pipeline import corpus_stats, load_corpus

        settings = get_settings()
        docs, registry = load_corpus(settings, rebuild=args.rebuild)
        engine = ClinicalEngine(settings, docs, registry)
        if args.rebuild:
            engine.store.index(docs, force=True)
        print(json.dumps({**corpus_stats(docs, registry), "chunks": engine.store.count(),
                          "embedding": engine.store.embedding_name, "graph": engine.kg.stats()}, indent=2))
        return 0
    if args.cmd == "ask":
        from clinicalchat.engine import get_engine
        from clinicalchat.orchestrator import ChatSession, Orchestrator
        from clinicalchat.personas import get_persona

        persona = get_persona(args.user)
        if not persona:
            print(f"Unknown user {args.user}", file=sys.stderr)
            return 2
        session = ChatSession(username=persona.username, role=persona.role, display_name=persona.display_name,
                              own_patient=persona.patient_key, selected_patient=persona.patient_key or args.patient)
        for ev in Orchestrator(get_engine()).ask(args.question, session):
            if ev.kind == "status":
                print(f"[{'..' if ev.status == 'pending' else 'ok'}] {ev.title}: {ev.text}", file=sys.stderr)
            else:
                print(f"\n=== {ev.title} ===\n{ev.text}")
        return 0
    if args.cmd == "reset-patient":
        return _reset_patient(args)
    return 1


def _reset_patient(args) -> int:
    import socket

    from clinicalchat.config import get_settings
    from clinicalchat.store import Store
    from clinicalchat.vectorstore import ClinicalVectorStore

    settings = get_settings()
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", settings.port)) == 0:
            print(f"The server is running on port {settings.port}. Stop it first: it keeps patient data in memory "
                  "and would write it back.", file=sys.stderr)
            return 2
    store = Store()
    user = store.get_user(args.patient) or next((u for u in store.users() if u["patient_key"] == args.patient), None)
    key = user["patient_key"] if user else args.patient
    uploads, cases = store.uploads(key), store.cases(patient_key=key)
    if not user and not uploads and not cases:
        print(f"Nothing stored for '{args.patient}'.", file=sys.stderr)
        return 1
    who = f"{user['display_name']} ({key})" if user else key
    print(f"{who}: {len(uploads)} upload(s), {len(cases)} case(s)"
          + ("; the login will be DELETED" if args.delete_login else "; the login is kept"))
    if not args.yes and input("Delete? [y/N] ").strip().lower() != "y":
        return 1
    removed = store.purge_patient(key, delete_login=args.delete_login)
    vectors = ClinicalVectorStore(settings)
    removed["index_chunks"] = vectors.purge_uploads(key)
    exports = list(settings.export_dir.glob(f"clinicalchat-{user['username']}-*.md")) if user else []
    for path in exports:
        path.unlink()
    removed["exports"] = len(exports)
    print(json.dumps(removed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
