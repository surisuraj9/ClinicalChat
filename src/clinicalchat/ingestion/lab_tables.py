"""Lab table parsing with reference-range extraction and deterministic status computation.

The LLM agents receive these pre-computed flags so that "is this value abnormal?" is never
left to model judgement alone (clinical accuracy + reproducibility).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from clinicalchat.domain.models import LabResult, LabStatus

_NUM = r"\d+(?:\.\d+)?"
_RANGE_RE = re.compile(rf"({_NUM})\s*(?:-|–|—|~|to)\s*({_NUM})", re.IGNORECASE)
_LT_RE = re.compile(rf"^(?:<|≤|<=|less than|up to)\s*=?\s*({_NUM})", re.IGNORECASE)
_GT_RE = re.compile(rf"^(?:>|≥|>=|more than|greater than)\s*=?\s*({_NUM})", re.IGNORECASE)
_VALUE_RE = re.compile(rf"^\s*({_NUM})\s*(.*)$")

_HEADER_VALUE = re.compile(r"\b(result|value|observed)\b", re.IGNORECASE)
_HEADER_RANGE = re.compile(r"(range|reference|ref\.?|interval|normal)", re.IGNORECASE)
_HEADER_STATUS = re.compile(r"\b(status|flag|interpretation)\b", re.IGNORECASE)
_HEADER_UNIT = re.compile(r"\bi?unit\b", re.IGNORECASE)

# Panic values that require immediate clinician notification (parameter keyword -> (low, high)).
CRITICAL_LIMITS: List[Tuple[Tuple[str, ...], Optional[float], Optional[float]]] = [
    (("hemoglobin", "haemoglobin", "hb "), 7.0, 20.0),
    (("fasting glucose", "blood sugar", "plasma glucose", "glucose"), 40.0, 500.0),
    (("potassium",), 2.5, 6.5),
    (("sodium",), 120.0, 160.0),
    (("platelet",), 50.0, 1000.0),  # in 10^3/uL (converted below when needed)
    (("troponin",), None, 0.4),
    (("inr",), None, 5.0),
]


def _strip_thousands(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def parse_reference_range(text: str) -> Tuple[Optional[float], Optional[float], str]:
    """Return (low, high, trailing_unit) for strings like '4.5–5.9', '<0.35 kUA/L', '>40 mg/dL'."""
    cleaned = _strip_thousands(text.replace("|", " ").strip())
    m = _RANGE_RE.search(cleaned)
    if m:
        return float(m.group(1)), float(m.group(2)), cleaned[m.end():].strip()
    m = _LT_RE.search(cleaned)
    if m:
        return None, float(m.group(1)), cleaned[m.end():].strip()
    m = _GT_RE.search(cleaned)
    if m:
        return float(m.group(1)), None, cleaned[m.end():].strip()
    return None, None, ""


def parse_value(text: str) -> Tuple[Optional[float], str]:
    cleaned = _strip_thousands(text.strip())
    if re.search(r"\d\s*/\s*\d", cleaned):  # blood pressure style values: not a single number
        return None, ""
    m = _VALUE_RE.match(cleaned)
    if not m:
        return None, ""
    return float(m.group(1)), m.group(2).strip()


def _normalise_platelets(param: str, value: float, high: Optional[float]) -> float:
    # Datasets mix 245,000 /uL and 245 x10^3/uL; compare critical limits in 10^3/uL.
    if "platelet" in param.lower() and value > 5000:
        return value / 1000.0
    return value


def compute_status(param: str, value: Optional[float], low: Optional[float], high: Optional[float]) -> LabStatus:
    if value is None or (low is None and high is None):
        return LabStatus.UNKNOWN
    status = LabStatus.NORMAL
    if low is not None and value < low:
        status = LabStatus.LOW
    elif high is not None and (value > high if low is not None else value >= high):
        status = LabStatus.HIGH
    p = f"{param.lower()} "
    for keywords, crit_low, crit_high in CRITICAL_LIMITS:
        if any(k in p for k in keywords):
            v = _normalise_platelets(param, value, high)
            if crit_low is not None and v < crit_low and status == LabStatus.LOW:
                return LabStatus.CRITICAL_LOW
            if crit_high is not None and v > crit_high and status == LabStatus.HIGH:
                return LabStatus.CRITICAL_HIGH
            break
    return status


def _reported_matches(reported: str, status: LabStatus) -> bool:
    r = reported.lower()
    if not r or status == LabStatus.UNKNOWN:
        return True
    if any(w in r for w in ("normal", "good", "optimal", "desirable", "negative", "non-reactive")):
        return status == LabStatus.NORMAL
    if any(w in r for w in ("low", "deficien", "decreased")):
        return status in (LabStatus.LOW, LabStatus.CRITICAL_LOW)
    if any(w in r for w in ("high", "elevated", "raised", "positive", "increased", "border", "pre", "abnormal")):
        return status in (LabStatus.HIGH, LabStatus.CRITICAL_HIGH)
    return True


def _find_header(rows: List[List[str]]) -> Optional[int]:
    for idx, row in enumerate(rows[:3]):
        joined = " ".join(row)
        if _HEADER_VALUE.search(joined) and _HEADER_RANGE.search(joined):
            return idx
    return None


def _read_cells(row: List[str], status_col: Optional[int]) -> Tuple[Optional[float], str, str, Optional[float],
                                                                     Optional[float], str]:
    """First numeric cell = the value (+ unit), first range-like cell = the reference range."""
    value, value_text, unit = None, "", ""
    low = high = None
    range_text = ""
    for i, cell in enumerate(row[1:], start=1):
        if i == status_col:
            continue
        if value is None:
            v, rest = parse_value(cell)
            if v is not None and not _RANGE_RE.search(_strip_thousands(cell)):
                value, value_text, unit = v, cell, rest
                continue
        if not range_text:
            lo, hi, trailing = parse_reference_range(cell)
            if lo is not None or hi is not None:
                low, high, range_text = lo, hi, cell.replace("|", " ").strip()
                unit = unit or trailing
    return value, value_text, unit, low, high, range_text


def _cell(row: List[str], col: Optional[int]) -> str:
    return row[col] if col is not None and col < len(row) else ""


def parse_lab_table(rows: List[List[str]]) -> List[LabResult]:
    """A report table with a value column and a reference-range column -> one LabResult per test row."""
    header_idx = _find_header(rows)
    if header_idx is None:
        return []
    header = rows[header_idx]
    status_col = next((i for i, h in enumerate(header) if _HEADER_STATUS.search(h) and not _HEADER_RANGE.search(h)), None)
    unit_col = next((i for i, h in enumerate(header) if _HEADER_UNIT.search(h) and len(h) <= 6), None)

    results: List[LabResult] = []
    for row in rows[header_idx + 1:]:
        if len(row) < 2 or not row[0]:
            continue
        param = row[0]
        if param.lower() in {"investigations", "parameter", "test", "result"} or parse_value(param)[0] is not None:
            continue
        value, value_text, unit, low, high, range_text = _read_cells(row, status_col)
        if value is None:
            continue
        unit = unit or _cell(row, unit_col)
        reported = _cell(row, status_col)
        status = compute_status(param, value, low, high)
        results.append(LabResult(
            parameter=param, value_text=value_text or row[1], value=value, unit=unit.strip(" |"),
            reference_range=range_text, ref_low=low, ref_high=high, status=status,
            reported_status=reported, status_discrepancy=not _reported_matches(reported, status)))
    return results


def parse_pipe_lines(lines: Iterable[str]) -> List[LabResult]:
    """Tables pasted as text ('Parameter | Result | Normal Range' lines inside a paragraph)."""
    rows = [[c.strip() for c in line.split("|")] for line in lines if line.count("|") >= 2]
    return parse_lab_table(rows) if rows else []


def make_lab_result(parameter: str, value_text: str, unit: str = "", reference_range: str = "", reported: str = "",
                    origin: str = "parser") -> Optional[LabResult]:
    """Build a LabResult from loose strings (e.g. values recovered from free text by the extraction agent)."""
    value, trailing = parse_value(str(value_text))
    if value is None:
        return None
    low, high, range_unit = parse_reference_range(str(reference_range)) if reference_range else (None, None, "")
    status = compute_status(parameter, value, low, high)
    return LabResult(parameter=parameter.strip(), value_text=str(value_text).strip(), value=value,
                     unit=(unit or trailing or range_unit).strip(), reference_range=str(reference_range).strip(),
                     ref_low=low, ref_high=high, status=status, reported_status=reported,
                     status_discrepancy=not _reported_matches(reported, status), origin=origin)
