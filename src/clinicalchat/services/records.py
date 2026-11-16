"""Builds the compact record text the AI reads: selected sections + a little reference material (max 9000 chars)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from clinicalchat.domain.models import Domain, Section
from clinicalchat.infrastructure.vector_index import VectorIndex
from clinicalchat.services.catalog import DocumentCatalog

MAX_CONTEXT_CHARS = 9000


def _fmt_section(s: Section, source: str, with_labs: bool = False, limit: int = 2500) -> str:
    head = f"### {s.title} — {s.patient_name or 'reference'} [{source}]" + (f" ({s.report_date})" if s.report_date else "")
    body = "\n".join("- " + r.line() for r in s.lab_results) if with_labs and s.lab_results else s.text[:limit]
    return f"{head}\n{body}"


def _pack(chunks: Sequence[str]) -> str:
    out, used = [], 0
    for c in chunks:
        if used + len(c) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - used
            if remaining > 400:
                out.append(c[:remaining] + "…")
            break
        out.append(c)
        used += len(c)
    return "\n\n".join(out) if out else "No records retrieved."


class RecordRetriever:
    def __init__(self, catalog: DocumentCatalog, index: VectorIndex):
        self.catalog = catalog
        self.index = index

    def search_sections(self, question: str, domains: Optional[Sequence[Domain]] = None, k: int = 6,
                        max_sections: int = 3) -> List[Section]:
        hits = self.index.search(question, k=k, domains=domains)
        return self.catalog.sections_for((h.section_id for h in hits), max_sections=max_sections)

    def records_context(self, query: str, sections: Sequence[Section], domains: Sequence[Domain],
                        reference_k: int = 2) -> str:
        chunks = [_fmt_section(s, self.catalog.source_of(s), with_labs=bool(s.lab_results), limit=1800) for s in sections]
        if reference_k:
            chunks += self._reference(query, [d for d in domains if d != Domain.REFERENCE], k=reference_k)
        return _pack(chunks)

    def reference_context(self, query: str, k: int = 4) -> Tuple[str, List[str]]:
        """General (non-patient) reference material for a general question, and its source files."""
        hits = [h for h in self.index.search(query, k=k, patient_key="", include_general=False) if h.score > 0.1]
        return (_pack([f"### {h.metadata.get('title', '')} [{h.source}]\n{h.text}" for h in hits]),
                list(dict.fromkeys(h.source for h in hits)))

    def _reference(self, query: str, domains: Sequence[Domain], k: int = 3) -> List[str]:
        hits = self.index.search(query, k=k, domains=[*domains, Domain.REFERENCE], patient_key="", include_general=False)
        return [f"### Reference: {h.metadata.get('title', '')} [{h.source}]\n{h.text}" for h in hits if h.score > 0.15]
