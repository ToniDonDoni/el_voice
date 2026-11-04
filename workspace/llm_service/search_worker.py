"""Isolated DDGS worker; the parent can terminate the process on cancel/timeout."""

from __future__ import annotations

import json
import sys

from .search import search_sync


def main() -> int:
    try:
        request = json.loads(sys.stdin.buffer.read(8192))
        results = search_sync(
            request["query"], request["backend"], int(request["max_results"]),
            int(request["max_snippet_chars"]), float(request["timeout_seconds"]),
        )
        sys.stdout.write(json.dumps({"results": results}, ensure_ascii=False))
        return 0
    except Exception as exc:
        name = type(exc).__name__.lower()
        error = "search_rate_limited" if "rate" in name or "limit" in str(exc).lower() else "search_unavailable"
        sys.stdout.write(json.dumps({"error": error}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
