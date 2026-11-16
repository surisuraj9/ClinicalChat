"""The agents ask the model for one JSON object; this module describes the expected keys and parses the reply leniently."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Tuple, Type

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
FENCE = re.compile(r"^```(?:json)?|```$", re.M)


def schema_hint(model: Type[BaseModel]) -> str:
    lines = []
    for name, f in model.model_fields.items():
        ann = str(f.annotation)
        kind = "list of strings" if "List" in ann else "number" if "float" in ann else "string"
        lines.append(f'- "{name}" ({kind}){": " + f.description if f.description else ""}')
    return "\n".join(lines)


def _as_text_list(value: Any) -> Any:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [x if isinstance(x, str) else " — ".join(map(str, x.values())) if isinstance(x, dict) else str(x)
                for x in value]
    return value


def _as_fraction(value: Any) -> float:
    """0.8, "80%", 80 -> 0.8 (clamped to 0..1). Raises ValueError when not a number."""
    number = float(str(value).strip("%"))
    return max(0.0, min(1.0, number / (100 if number > 1 else 1)))


def _normalise(data: Dict[str, Any], model: Type[BaseModel]) -> Dict[str, Any]:
    for k, v in list(data.items()):
        if k not in model.model_fields:
            continue
        ann = str(model.model_fields[k].annotation)
        if "List" in ann:
            data[k] = _as_text_list(v)
        elif "float" in ann:
            try:
                data[k] = _as_fraction(v)
            except ValueError:
                data.pop(k)
    return data


def parse_output(raw: str, model: Type[BaseModel], text_field: str) -> Tuple[BaseModel, bool]:
    """Lenient JSON extraction; falls back to the raw text (never a second LLM call). Returns (output, parsed_ok)."""
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return model.model_validate(_normalise(data, model)), True
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning("Could not parse agent JSON: %s", str(exc)[:200])
    return model(**{text_field: text.strip("`") or "No answer was produced."}), False
