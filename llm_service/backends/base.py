"""Backend contract shared by the local and remote LLM implementations.

A backend is an async iterable of event dictionaries for one generation:

* ``{"type": "delta", "text": "..."}`` - final-answer text only;
* ``{"type": "tool", "name": "web_search", "query": "..."}`` - a tool call was
  requested and is about to run;
* ``{"type": "sources", "sources": [{"title", "url", "snippet"}]}`` - bounded
  search results that the UI may display but must not speak.

Terminal conditions are an exception (``BackendError``) or the end of the
iterator.  The cancel token is a ``threading.Event`` so blocking CPU inference
and asyncio network code can observe the same flag between tokens/chunks.
"""

from __future__ import annotations

import threading
from typing import AsyncIterator, Protocol


class BackendError(RuntimeError):
    """A generation failure with a redacted, user-facing explanation."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class Backend(Protocol):
    mode: str

    async def start(self) -> None:
        """Load or validate the backend. Raises BackendError/ValueError."""

    async def close(self) -> None:
        """Release model/network resources."""

    def status(self) -> dict:
        """Content-free readiness information for ``/status``."""

    def stream(
        self, messages: list[dict], cancel: threading.Event
    ) -> AsyncIterator[dict]:
        """Yield events for one prompt. Must observe ``cancel``."""
