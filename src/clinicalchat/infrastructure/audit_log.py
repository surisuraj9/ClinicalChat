"""Always-on audit trail: one JSON line per question, upload and care-team action (data/audit/audit-<date>.jsonl)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class AuditLog:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.directory / f"audit-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"

    def record(self, event: Dict[str, Any]) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
