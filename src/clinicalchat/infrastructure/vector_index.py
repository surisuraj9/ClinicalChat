"""ChromaDB vector index: finds the report passages most similar in meaning to a question.

Retrieval is "small-to-big": short chunks (sized for MiniLM's 256-token window) are embedded
and searched, then the agents receive the full parent section for complete clinical context.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import chromadb

from clinicalchat.config import Settings, get_settings
from clinicalchat.domain.models import ClinicalDocument, Domain, Section

log = logging.getLogger(__name__)

COLLECTION = "clinical_documents"
CHUNK_WORDS = 150
CHUNK_OVERLAP = 30


def build_embedding_function(settings: Settings):
    """sentence-transformers when available, otherwise the ONNX export of the same MiniLM model."""
    provider = settings.embedding_provider
    if provider in {"auto", "sentence-transformers"}:
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            ef = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
            ef(["warm-up"])
            return ef, f"sentence-transformers/{settings.embedding_model}"
        except Exception as exc:
            if provider == "sentence-transformers":
                raise
            log.info("sentence-transformers unavailable (%s); using ONNX MiniLM export", type(exc).__name__)
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    return ONNXMiniLM_L6_V2(), "onnx/all-MiniLM-L6-v2"


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, object]

    @property
    def section_id(self) -> str:
        return str(self.metadata.get("section_id", ""))

    @property
    def source(self) -> str:
        return str(self.metadata.get("filename", ""))


def chunk_section(section: Section, doc: ClinicalDocument) -> List[str]:
    header = f"[{doc.title} | {section.title}" + (f" | Patient: {section.patient_name}" if section.patient_name else "") + "]"
    words = section.text.split()
    if not words:
        return []
    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for start in range(0, len(words), step):
        chunks.append(f"{header}\n{' '.join(words[start:start + CHUNK_WORDS])}")
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def corpus_fingerprint(documents: Sequence[ClinicalDocument], model_name: str) -> str:
    h = hashlib.sha256(model_name.encode())
    for doc in documents:
        for s in doc.sections:
            h.update(s.section_id.encode())
            h.update(s.text.encode())
    return h.hexdigest()[:16]


class VectorIndex:
    def __init__(self, settings: Optional[Settings] = None, persist: bool = True, collection_name: str = COLLECTION):
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.embedding_function, self.embedding_name = build_embedding_function(self.settings)
        if persist:
            self.client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        else:
            self.client = chromadb.EphemeralClient()
        self.collection = None

    def index(self, documents: Sequence[ClinicalDocument], force: bool = False) -> int:
        """(Re)build the collection unless the stored one was built from exactly these documents."""
        fingerprint = corpus_fingerprint(documents, self.embedding_name)

        existing = None
        try:
            existing = self.client.get_collection(self.collection_name, embedding_function=self.embedding_function)
        except Exception:
            existing = None
        if existing is not None and not force and (existing.metadata or {}).get("fingerprint") == fingerprint:
            self.collection = existing
            return existing.count()
        if existing is not None:
            self.client.delete_collection(self.collection_name)

        self.collection = self.client.create_collection(
            self.collection_name, embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "fingerprint": fingerprint, "embedding": self.embedding_name},
        )
        count = self._add(documents)
        log.info("Indexed %d chunks with %s", count, self.embedding_name)
        return count

    def _add(self, documents: Sequence[ClinicalDocument]) -> int:
        ids, texts, metas = [], [], []
        for doc in documents:
            for section in doc.sections:
                for i, chunk in enumerate(chunk_section(section, doc)):
                    ids.append(f"{section.section_id}-C{i:02d}")
                    texts.append(chunk)
                    metas.append({
                        "doc_id": doc.doc_id, "section_id": section.section_id, "filename": doc.filename,
                        "category": doc.category, "domain": doc.domain.value, "title": section.title,
                        "patient_key": section.patient_key or "", "patient_name": section.patient_name or "",
                        "chunk_index": i, "origin": doc.origin, "report_date": section.report_date or "",
                    })
        for start in range(0, len(ids), 128):
            self.collection.upsert(ids=ids[start:start + 128], documents=texts[start:start + 128],
                                   metadatas=metas[start:start + 128])
        return len(ids)

    def add_document(self, doc: ClinicalDocument) -> int:
        """Incrementally index one (uploaded) document."""
        if self.collection is None:
            raise RuntimeError("Vector store not indexed yet")
        return self._add([doc])

    def purge_uploads(self, patient_key: str) -> int:
        """Delete a patient's *uploaded* chunks (dataset chunks are never touched). Returns chunks removed."""
        collection = self.collection or self.client.get_collection(self.collection_name,
                                                                   embedding_function=self.embedding_function)
        where = {"$and": [{"patient_key": patient_key}, {"origin": "upload"}]}
        ids = collection.get(where=where, include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    @staticmethod
    def _where(domains: Optional[Sequence[Domain]], patient_key: Optional[str], include_general: bool) -> Optional[dict]:
        clauses = []
        if domains:
            values = [d.value for d in domains]
            clauses.append({"domain": {"$in": values}} if len(values) > 1 else {"domain": values[0]})
        if patient_key is not None:  # "" = general/reference material only
            if include_general and patient_key:
                clauses.append({"$or": [{"patient_key": patient_key}, {"patient_key": ""}]})
            else:
                clauses.append({"patient_key": patient_key})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, query: str, k: int = 6, domains: Optional[Sequence[Domain]] = None,
               patient_key: Optional[str] = None, include_general: bool = True) -> List[SearchHit]:
        if self.collection is None:
            raise RuntimeError("Vector store not indexed yet")
        result = self.collection.query(
            query_texts=[query], n_results=k, where=self._where(domains, patient_key, include_general),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                         result["metadatas"][0], result["distances"][0], strict=True):
            hits.append(SearchHit(chunk_id=cid, text=text, score=round(1 - float(dist), 4), metadata=dict(meta)))
        return hits

    def count(self) -> int:
        return self.collection.count() if self.collection is not None else 0
