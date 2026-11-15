"""Command line: `clinicalchat serve | ingest | ask`."""

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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
