"""Date parsing for clinical reports (report dates drive historical comparison)."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Dict, Optional

_FORMATS = ("%d-%b-%Y", "%d %b %Y", "%d-%B-%Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d-%m-%Y", "%d/%m/%Y",
            "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d", "%d-%b-%y", "%d/%m/%y")
_DATE_TOKEN = re.compile(r"(\d{1,2}[-/. ](?:[A-Za-z]{3,9}|\d{1,2})[-/. ,]+\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|[A-Za-z]{3,9} \d{1,2}, \d{4})")

# Field names that carry the date the report/test refers to, most specific first
DATE_FIELDS = ("Report Date", "Date of Report", "Date of Test", "Test Date", "Collection Date", "Sample Date",
               "Study Date", "Date of Scan", "Date of Measurement", "Date of Study", "Date", "Discharge Date",
               "Admission Date")


def parse_date(text: Optional[str]) -> Optional[str]:
    """Return an ISO date (YYYY-MM-DD) from free text, or None."""
    if not text:
        return None
    if isinstance(text, (date, datetime)):
        return text.strftime("%Y-%m-%d")
    candidates = [text.strip()] + _DATE_TOKEN.findall(text)
    for cand in candidates:
        cand = re.sub(r"\s+", " ", cand.strip().rstrip("."))
        for fmt in _FORMATS:
            try:
                return datetime.strptime(cand, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


def date_from_fields(fields: Dict[str, str]) -> Optional[str]:
    lowered = {k.lower(): v for k, v in fields.items()}
    for key in DATE_FIELDS:
        value = lowered.get(key.lower())
        parsed = parse_date(value)
        if parsed:
            return parsed
    return None


_DATE_LINE = re.compile(r"^\s*(?:report|test|collection|sample|scan|study|measurement)?\s*date(?:\s*of\s*(?:test|report|scan|study|measurement|collection))?\s*[:\-]", re.I)


def date_from_text(text: str) -> Optional[str]:
    """Fallback for OCR text where field names lose their spaces (e.g. 'DateofTest:05-Mar-2026')."""
    for line in text.splitlines():
        squashed = re.sub(r"(?<=[a-z])(?=of)|(?<=of)(?=[A-Z])", " ", line)
        if _DATE_LINE.search(squashed) or re.match(r"^\s*date", line, re.I):
            parsed = parse_date(line.split(":", 1)[-1])
            if parsed:
                return parsed
    return None
