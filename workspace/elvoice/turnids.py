"""Application-wide turn identity allocation.

Turn ids must not restart at 1 when a browser reconnects and a new voice
session begins, otherwise audio and flush events from different sessions can
share the same id. One allocator per voice application hands out unique,
strictly increasing ids for the whole process lifetime.
"""

from __future__ import annotations

import itertools
import threading


class TurnIdSource:
    def __init__(self, start: int = 1):
        self._counter = itertools.count(int(start))
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            return next(self._counter)
