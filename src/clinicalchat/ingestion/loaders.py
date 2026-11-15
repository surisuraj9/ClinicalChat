"""Low-level loaders for DOCX / PDF (incl. OCR for scanned pages) / XLSX clinical documents.

Every loader returns an ordered list of `Block`s (paragraph or table) so that tables stay
next to the patient section they belong to.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
UPLOAD_SUFFIXES = {".docx", ".pdf", ".xlsx", ".txt", ".md", *IMAGE_SUFFIXES}


@dataclass
class Block:
    kind: str  # "para" | "table"
    text: str
    rows: List[List[str]] = field(default_factory=list)
    source: str = "text"  # text | ocr


def _clean(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def table_to_text(rows: List[List[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row if cell) for row in rows if any(row))


def load_docx(path: Path) -> List[Block]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    blocks: List[Block] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = _clean(Paragraph(child, doc).text)
            if text:
                blocks.append(Block("para", text))
        elif tag == "tbl":
            rows: List[List[str]] = []
            for row in Table(child, doc).rows:
                cells: List[str] = []
                for cell in row.cells:
                    value = _clean(cell.text)
                    # merged cells repeat; keep the first occurrence only
                    if not cells or value != cells[-1]:
                        cells.append(value)
                if any(cells):
                    rows.append(cells)
            if rows:
                blocks.append(Block("table", table_to_text(rows), rows))
    return blocks


def _ocr_page_rows(image) -> List[List[str]]:
    """OCR a rendered page and regroup detected text boxes into visual rows."""
    from rapidocr_onnxruntime import RapidOCR

    engine = _ocr_engine()
    result, _ = engine(image)
    if not result:
        return []
    boxes = []
    for box, text, _score in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        boxes.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs), _clean(text)))
    boxes.sort(key=lambda b: (b[0], b[2]))
    heights = sorted(b[1] for b in boxes)
    tolerance = max(4.0, heights[len(heights) // 2] * 0.6)
    rows: List[List[tuple]] = []
    for box in boxes:
        if rows and abs(rows[-1][0][0] - box[0]) <= tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [[b[3] for b in sorted(row, key=lambda b: b[2])] for row in rows]


_OCR = None


def _ocr_engine():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
    return _OCR


def load_pdf(path: Path, ocr_enabled: bool = True, cache_dir: Optional[Path] = None) -> List[Block]:
    import pdfplumber

    blocks: List[Block] = []
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                if _clean(line):
                    blocks.append(Block("para", _clean(line)))
            for table in page.extract_tables() or []:
                rows = [[_clean(c or "") for c in row] for row in table if row]
                if rows:
                    blocks.append(Block("table", table_to_text(rows), rows))

    text_chars = sum(len(b.text) for b in blocks)
    if text_chars >= 40 * max(page_count, 1) or not ocr_enabled:
        return blocks

    # Scanned PDF -> OCR (cached because it is the slowest ingestion step)
    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        cache_file = cache_dir / "ocr" / f"{digest}.txt"
        if cache_file.exists():
            return _blocks_from_ocr_text(cache_file.read_text(encoding="utf-8"))

    log.info("OCR: %s (%d pages)", path.name, page_count)
    import pypdfium2 as pdfium

    page_texts = []
    pdf = pdfium.PdfDocument(str(path))
    for index in range(len(pdf)):
        rows = _ocr_page_rows(pdf[index].render(scale=2).to_numpy())
        page_texts.append("\n".join(" | ".join(r) for r in rows))
    ocr_text = "\n\f\n".join(page_texts)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(ocr_text, encoding="utf-8")
    return _blocks_from_ocr_text(ocr_text)


def _blocks_from_ocr_text(text: str) -> List[Block]:
    blocks: List[Block] = []
    for page in text.split("\f"):
        lines = [l.strip() for l in page.strip().splitlines() if l.strip()]
        table_rows = [l.split(" | ") for l in lines if " | " in l]
        for line in lines:
            if " | " not in line:
                blocks.append(Block("para", line, source="ocr"))
        if table_rows:
            blocks.append(Block("table", table_to_text(table_rows), table_rows, source="ocr"))
    return blocks


def load_text(path: Path) -> List[Block]:
    """Plain-text / markdown report. Lines containing '|' (or tabs) are treated as table rows."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    blocks: List[Block] = []
    table: List[List[str]] = []

    def flush() -> None:
        if table:
            blocks.append(Block("table", table_to_text(table), list(table)))
            table.clear()

    for line in raw.splitlines():
        line = _clean(line.replace("\t", " | ") if "\t" in line else line)
        if line and re.fullmatch(r"\|?[\s:|-]+\|?", line):  # markdown table rule (keeps header + rows together)
            continue
        if not line:
            flush()
            continue
        if line.count("|") >= 2:
            table.append([_clean(c) for c in line.strip("|").split("|")])
        else:
            flush()
            blocks.append(Block("para", line))
    flush()
    return blocks


def load_image(path: Path) -> List[Block]:
    """Photo/scan of a report -> OCR rows (rows with several cells become a table)."""
    import numpy as np
    from PIL import Image

    rows = _ocr_page_rows(np.array(Image.open(path).convert("RGB")))
    text = "\n".join(" | ".join(r) for r in rows)
    return _blocks_from_ocr_text(text)


def load_xlsx(path: Path) -> List[Block]:
    import openpyxl

    blocks: List[Block] = []
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    for ws in wb.worksheets:
        rows = [[_clean(str(c)) if c is not None else "" for c in row] for row in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if any(r)]
        if rows:
            blocks.append(Block("table", table_to_text(rows), rows))
    return blocks


def load_document(path: Path, ocr_enabled: bool = True, cache_dir: Optional[Path] = None) -> List[Block]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".pdf":
        return load_pdf(path, ocr_enabled=ocr_enabled, cache_dir=cache_dir)
    if suffix == ".xlsx":
        return load_xlsx(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)
    if suffix in IMAGE_SUFFIXES:
        return load_image(path)
    raise ValueError(f"Unsupported document format: {path.suffix}")
