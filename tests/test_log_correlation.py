"""Diagnostic logs must correlate every event of one answer by turn/request id.

Run:  python tests/test_log_correlation.py
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    from llm_service.log_context import (
        bind, configure_logging, log_event, reset,
    )

    configure_logging()
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter(
        "%(session_id)s|%(turn_id)s|%(request_id)s|%(message)s"))
    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        root.handlers[:] = [handler]
        token = bind(session="sess-1", turn=7, request="req-9")
        logger = logging.getLogger("corr.test")
        log_event(logger, "generate_start", chars=42)
        log_event(logger, "tool_call", name="web_search", query="patch 7.41")
        reset(token)
        log_event(logger, "connection_closed", active_sessions=0)
    finally:
        root.handlers[:] = saved

    output = buffer.getvalue()
    check("turn/request/session ids are attached to each line",
          "sess-1|7|req-9|event=generate_start chars=42" in output
          and "sess-1|7|req-9|event=tool_call name=web_search" in output,
          output.splitlines()[:2])
    check("events outside a request fall back to placeholders",
          "-|-|-|event=connection_closed" in output, output.splitlines()[-1:])
    check("search query is visible in the correlated trace",
          "query=patch 7.41" in output, output)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
