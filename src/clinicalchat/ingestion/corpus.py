"""Dataset ingestion (runs once, result cached): discover -> load (DOCX/PDF/OCR/XLSX) -> section per patient -> registry.

The processed corpus is cached as JSON in `data/processed/corpus.json` so the app starts fast.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from clinicalchat.config import Settings, get_settings
from clinicalchat.domain.models import ClinicalDocument, Domain, PatientProfile, Section
from clinicalchat.ingestion.dates import date_from_fields
from clinicalchat.ingestion.loaders import load_document
from clinicalchat.ingestion.registry import PatientRegistry
from clinicalchat.ingestion.sectioning import split_sections

log = logging.getLogger(__name__)

SUPPORTED = {".docx", ".pdf", ".xlsx"}

# path keyword -> (category, domain); first match wins
CATEGORY_RULES: List[Tuple[str, str, Domain]] = [
    ("food allergy", "allergy_food", Domain.ALLERGY),
    ("medication allergy", "allergy_medication", Domain.ALLERGY),
    ("environment allergy", "allergy_environment", Domain.ALLERGY),
    ("skin allergy", "allergy_skin", Domain.ALLERGY),
    ("radiolog", "radiology", Domain.RADIOLOGY),
    ("clinician", "lab", Domain.LAB),
    ("doctor", "lab", Domain.LAB),
    ("training", "reference", Domain.REFERENCE),
    ("patient", "patient_record", Domain.PATIENT_RECORD),
]


def categorise(relative_path: str) -> Tuple[str, Domain]:
    low = relative_path.lower()
    for keyword, category, domain in CATEGORY_RULES:
        if keyword in low:
            return category, domain
    return "other", Domain.REFERENCE


def discover(data_dir: Path) -> List[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
                  and not p.name.startswith((".", "~$")))


def _title(path: Path) -> str:
    stem = path.stem.replace(".zz", "").replace(" zz", "").replace("_", " ")
    return " ".join(stem.split())


def _doc_id(relative_path: str, category: str) -> str:
    return f"{category.upper()}-{hashlib.sha1(relative_path.encode()).hexdigest()[:8]}"


def _glossary_sections(doc_id: str, rows: List[List[str]]) -> List[Section]:
    """One section per ~25 glossary terms keeps chunks semantically tight."""
    body = rows[1:]  # row 0 is the header
    sections = []
    for start in range(0, len(body), 25):
        chunk = body[start:start + 25]
        text = "\n".join(f"{r[0]} ({', '.join(c for c in r[1:3] if c)}): {r[3] if len(r) > 3 else ''}" for r in chunk)
        sections.append(Section(section_id=f"{doc_id}-S{len(sections):02d}", doc_id=doc_id,
                                title=f"Medical glossary {start + 1}-{start + len(chunk)}", text=text))
    return sections


def ingest(settings: Optional[Settings] = None) -> Tuple[List[ClinicalDocument], PatientRegistry]:
    settings = settings or get_settings()
    documents: List[ClinicalDocument] = []
    registry = PatientRegistry()

    for path in discover(settings.data_dir):
        rel = str(path.relative_to(settings.data_dir))
        category, domain = categorise(rel)
        doc_id = _doc_id(rel, category)
        title = _title(path)
        try:
            blocks = load_document(path, ocr_enabled=settings.ocr_enabled, cache_dir=settings.processed_dir)
        except Exception as exc:  # one bad file must not stop ingestion
            log.warning("Failed to load %s: %s", rel, exc)
            continue
        doc = ClinicalDocument(doc_id=doc_id, filename=path.name, relative_path=rel, category=category, domain=domain,
                               title=title, extraction_method="ocr" if any(b.source == "ocr" for b in blocks) else "text")

        if path.suffix.lower() == ".xlsx" and blocks:
            doc.sections = _glossary_sections(doc_id, blocks[0].rows)
        else:
            drafts = split_sections(blocks, title)
            if domain == Domain.LAB and not any(d.name for d in drafts) and any(d.lab_results for d in drafts):
                # scanned lab panel without a readable patient header: keep it queryable as its own record
                for d in drafts:
                    d.name = f"Unnamed Patient ({title})"
            for i, draft in enumerate(drafts):
                key = registry.register(draft.name, draft.age, draft.gender, draft.identifiers) if draft.name else None
                doc.sections.append(Section(
                    section_id=f"{doc_id}-S{i:02d}", doc_id=doc_id, title=draft.title, text=draft.text,
                    patient_key=key, patient_name=draft.name, fields=draft.fields, lab_results=draft.lab_results,
                    report_date=date_from_fields(draft.fields),
                ))
        documents.append(doc)

    registry.build(documents)
    return documents, registry


def corpus_path(settings: Settings) -> Path:
    return settings.processed_dir / "corpus.json"


def save_corpus(documents: List[ClinicalDocument], registry: PatientRegistry, settings: Settings) -> Path:
    path = corpus_path(settings)
    payload = {
        "documents": [d.model_dump(mode="json") for d in documents],
        "patients": [p.model_dump(mode="json") for p in registry.all()],
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_corpus(settings: Optional[Settings] = None, rebuild: bool = False) -> Tuple[List[ClinicalDocument], PatientRegistry]:
    settings = settings or get_settings()
    path = corpus_path(settings)
    if path.exists() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = [ClinicalDocument.model_validate(d) for d in payload["documents"]]
        registry = PatientRegistry()
        for p in payload["patients"]:
            profile = PatientProfile.model_validate(p)
            registry.patients[profile.patient_key] = profile
        return documents, registry
    documents, registry = ingest(settings)
    save_corpus(documents, registry, settings)
    return documents, registry


def corpus_stats(documents: List[ClinicalDocument], registry: PatientRegistry) -> Dict[str, object]:
    by_category: Dict[str, int] = {}
    for d in documents:
        by_category[d.category] = by_category.get(d.category, 0) + 1
    return {
        "documents": len(documents),
        "sections": sum(len(d.sections) for d in documents),
        "patients": len(registry.patients),
        "lab_results": sum(len(s.lab_results) for d in documents for s in d.sections),
        "positive_allergies": sum(len(p.allergies) for p in registry.all()),
        "by_category": by_category,
    }
