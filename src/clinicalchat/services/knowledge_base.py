"""The knowledge base: every record the app knows, held in memory and kept in sync across its four views —
the document catalogue, the patient directory, the knowledge graph and the vector index."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Iterable, List, Optional

from clinicalchat.config import Settings
from clinicalchat.domain.models import ClinicalDocument, PatientProfile
from clinicalchat.ingestion.corpus import load_corpus
from clinicalchat.ingestion.registry import PatientRegistry
from clinicalchat.infrastructure.vector_index import VectorIndex
from clinicalchat.knowledge.graph import MedicalKnowledgeGraph
from clinicalchat.services.catalog import DocumentCatalog
from clinicalchat.services.patients import PatientDirectory
from clinicalchat.services.records import RecordRetriever
from clinicalchat.services.safety import SafetyChecker

log = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, settings: Settings, documents: List[ClinicalDocument], registry: PatientRegistry,
                 index: VectorIndex):
        self.settings = settings
        self._lock = threading.RLock()
        self.catalog = DocumentCatalog(documents)
        self.kg = MedicalKnowledgeGraph(registry.all())
        self.patients = PatientDirectory(registry, self.kg, self.catalog, self._lock)
        self.index = index
        self.index.index(documents)
        self.records = RecordRetriever(self.catalog, index)
        self.safety = SafetyChecker(self.kg, self.patients)

    @classmethod
    def load(cls, settings: Settings, persist: bool = True) -> "KnowledgeBase":
        """Dataset from the processed corpus cache (parsed on first run) + the vector index."""
        documents, registry = load_corpus(settings)
        return cls(settings, documents, registry, VectorIndex(settings, persist=persist))

    def add_document(self, doc: ClinicalDocument) -> List[str]:
        """Index a new (uploaded) report everywhere; returns the patients whose profile changed."""
        with self._lock:
            if doc.doc_id in self.catalog:
                return []
            self.catalog.add(doc)
            self.index.add_document(doc)
            touched = self.patients.registry.add_document(doc)
            for key in touched:
                self.patients.refresh(key)
            return touched

    def load_user_data(self, users: Iterable[Dict], facts: Iterable[Dict], uploads: Iterable[Dict]) -> None:
        """Re-create self-registered patients, their facts and uploads from the database at start-up."""
        users = list(users)
        registry = self.patients.registry
        for u in users:
            if u["patient_key"] not in registry.patients:
                registry.patients[u["patient_key"]] = PatientProfile(
                    patient_key=u["patient_key"], name=u["display_name"], age=u["age"], gender=u["gender"])
        for f in facts:
            self.patients.apply_fact(f["patient_key"], f["kind"], f["value"], refresh=False)
        for row in uploads:
            try:
                self.add_document(ClinicalDocument.model_validate_json(row["doc_json"]))
            except Exception:  # a corrupt upload must not stop the app
                log.exception("Could not reload upload %s", row["doc_id"])
        for u in users:
            self.patients.refresh(u["patient_key"])

    def purge_uploads(self, patient_key: str) -> int:
        """Forget a patient's uploaded reports (vector chunks + catalogue); returns the chunks removed."""
        with self._lock:
            self.catalog.remove_uploads(patient_key)
            return self.index.purge_uploads(patient_key)

    def stats(self, extra: Optional[Dict] = None) -> Dict:
        return {"chunks": self.index.count(), "embedding": self.index.embedding_name, "graph": self.kg.stats(),
                **(extra or {})}
