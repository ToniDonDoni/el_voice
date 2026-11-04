"""Correlated diagnostic logging for the LLM service.

Every log record is tagged with the current session, turn and request id so the
whole answer to one question can be traced: provider calls, tool decisions,
web searches, returned sources and the final answer all share one ``req`` id
and the client's ``turn`` id. Ids are routing/diagnostic metadata; transcript
content is not logged, only event names, counts and (for local diagnostics)
the search query and source URLs.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

_CONTEXT: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "llm_log_context", default={}
)
_DEFAULTS = {"session": "-", "turn": "-", "request": "-"}
_original_factory = None


def current() -> dict[str, str]:
    return {**_DEFAULTS, **_CONTEXT.get()}


def bind(**fields: Any) -> contextvars.Token:
    values = current()
    values.update({key: str(value) for key, value in fields.items()})
    return _CONTEXT.set(values)


def reset(token: contextvars.Token) -> None:
    _CONTEXT.reset(token)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    parts = [f"event={event}"]
    for key, value in fields.items():
        text = str(value).replace("\n", " ").replace("\r", " ")
        parts.append(f"{key}={text}")
    logger.log(level, " ".join(parts))


class ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = current()
        record.session_id = context["session"]
        record.turn_id = context["turn"]
        record.request_id = context["request"]
        return super().format(record)


def configure_logging() -> None:
    """Install the context formatter on the root logger (and uvicorn logs)."""
    global _original_factory
    if _original_factory is None:
        _original_factory = logging.getLogRecordFactory()

        def factory(*args, **kwargs):
            record = _original_factory(*args, **kwargs)
            context = current()
            record.session_id = context["session"]
            record.turn_id = context["turn"]
            record.request_id = context["request"]
            return record

        logging.setLogRecordFactory(factory)

    formatter = ContextFormatter(
        "%(asctime)s %(levelname)s %(name)s "
        "[session=%(session_id)s turn=%(turn_id)s req=%(request_id)s] %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not getattr(root, "_el_voice_context_handler", False):
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root._el_voice_context_handler = True
