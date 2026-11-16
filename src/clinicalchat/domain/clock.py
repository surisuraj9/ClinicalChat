"""Timestamps in the two formats the app stores: local minutes (records, cases) and UTC (uploads)."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def local_minute() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def utc_minute() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
