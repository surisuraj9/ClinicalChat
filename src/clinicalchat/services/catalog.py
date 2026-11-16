"""In-memory catalogue of every document and section (dataset + uploads): what a section is, where it came from."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from clinicalchat.domain.models import ClinicalDocument, Domain, Section


class DocumentCatalog:
    def __init__(self, documents: Iterable[ClinicalDocument] = ()):
        self.documents: Dict[str, ClinicalDocument] = {}
        self.sections: Dict[str, Section] = {}
        for doc in documents:
            self.add(doc)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.documents

    def add(self, doc: ClinicalDocument) -> None:
        self.documents[doc.doc_id] = doc
        for s in doc.sections:
            self.sections[s.section_id] = s

    def remove_uploads(self, patient_key: str) -> None:
        gone = {d for d, doc in self.documents.items()
                if doc.origin == "upload" and any(sec.patient_key == patient_key for sec in doc.sections)}
        self.documents = {d: doc for d, doc in self.documents.items() if d not in gone}
        self.sections = {k: v for k, v in self.sections.items() if v.doc_id not in gone}

    def all(self) -> List[ClinicalDocument]:
        return list(self.documents.values())

    def document_of(self, section: Section) -> Optional[ClinicalDocument]:
        return self.documents.get(section.doc_id)

    def domain_of(self, section: Section) -> Optional[Domain]:
        doc = self.document_of(section)
        return doc.domain if doc else None

    def source_of(self, section: Section) -> str:
        doc = self.document_of(section)
        return doc.filename if doc else ""

    def patient_sections(self, patient_key: str, domains: Optional[Sequence[Domain]] = None) -> List[Section]:
        wanted = set(domains) if domains else None
        return [s for s in self.sections.values() if s.patient_key == patient_key
                and (wanted is None or self.domain_of(s) in wanted)]

    def sections_for(self, section_ids: Iterable[str], max_sections: int = 4) -> List[Section]:
        """Unique sections in the given (search-ranked) order — "small-to-big": chunks are searched, sections returned."""
        ordered = list(dict.fromkeys(section_ids))[:max_sections]
        return [self.sections[s] for s in ordered if s in self.sections]
