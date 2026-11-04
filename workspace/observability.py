"""Helpers for timestamped, structured application logs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log_structured(logger: logging.Logger, event: str, **fields: Any) -> None:
    record = {"timestamp": utc_timestamp(), "event": event, **fields}
    logger.info("%s", json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
