"""De-identification: strip anything that could identify a patient before text leaves the server (web search)."""

from __future__ import annotations

import re
from typing import Iterable

_SCRUB = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                                  # e-mail
    re.compile(r"(?i)\b(patient\s+name|name)\s*[:=]\s*\S+(\s+\S+)?"),           # "Name: Robin Hood"
    re.compile(r"(?i)\b(mrn|dob|uhid|id|case|patient\s+id)\b\s*[:=#]?\s*[\w/.-]*\d[\w/.-]*"),
    re.compile(r"\+?\d[\d\s()-]{7,}\d"),                                     # phone numbers
    re.compile(r"\b\d{6,}\b"),                                               # long ids
    re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b"),                      # dates
    re.compile(r"\b[A-Z]{1,4}-[A-Z0-9]{4,}\b"),                               # case ids (C-5A0A12FB)
    re.compile(r"\b[a-z]+(?:-[a-z]+)+-\d{1,3}(?:-\d+)?\b"),                  # patient keys (robin-hood-31)
]


def deidentify(text: str, names: Iterable[str] = ()) -> str:
    for rx in _SCRUB:
        text = rx.sub(" ", text)
    parts = {p for n in names for p in re.split(r"[\s.'-]+", (n or "").lower()) if len(p) >= 3}
    if parts:
        text = re.sub(r"(?i)\b(" + "|".join(map(re.escape, sorted(parts, key=len, reverse=True))) + r")(?:'s)?\b",
                      " ", text)
    text = re.sub(r"\(\s*[,;]?\s*\)|\s+[,;](?=[\s,;)])", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,.;:-()")[:380]
