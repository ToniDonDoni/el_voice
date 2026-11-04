"""DuckDuckGo search checks for SPEC-002.

Two clearly separated groups:

1. Mocked/provider-independent checks of source bounding and URL safety. These
   always run.
2. An optional live ``ddgs`` search. It is skipped unless ``--live`` or
   ``EL_VOICE_LIVE_SEARCH=1`` is set, because it requires outbound HTTPS.

Run:  python tests/test_search_live.py
      python tests/test_search_live.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_service.backends.base import BackendError  # noqa: E402
from llm_service.backends.search import (  # noqa: E402
    ddg_search,
    normalize_sources,
    safe_url,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def mocked_checks() -> None:
    print("== mocked / provider-independent checks ==")
    rows = [
        {"title": "OK", "href": "https://example.com/a", "body": "  snippet  " * 100},
        {"title": "Loopback", "href": "http://127.0.0.1/private", "body": "x"},
        {"title": "IPv6 loopback", "href": "http://[::1]/private", "body": "x"},
        {"title": "Metadata", "href": "http://169.254.169.254/latest/meta-data", "body": "x"},
        {"title": "File", "href": "file:///etc/passwd", "body": "x"},
        {"title": "Credentials", "href": "https://user:pass@example.com/", "body": "x"},
        {"title": "Private", "href": "http://10.0.0.5/admin", "body": "x"},
        {"title": "Second", "href": "https://example.org/b", "body": "ok"},
    ]
    sources = normalize_sources(rows, max_results=3, snippet_chars=50)
    check("source count is bounded", len(sources) <= 3, f"{len(sources)} sources")
    check("only public HTTP(S) URLs are kept",
          all(source["url"].startswith("https://") for source in sources),
          repr([source["url"] for source in sources]))
    check("snippets are bounded and whitespace-normalised",
          all(len(source["snippet"]) <= 50 for source in sources)
          and "\n" not in sources[0]["snippet"],
          repr([source["snippet"][:20] for source in sources]))
    numeric = [safe_url("http://127.1/x"), safe_url("http://2130706433/x"),
               safe_url("http://0x7f000001/x")]
    check("numeric private address forms are rejected",
          all(candidate is None for candidate in numeric), repr(numeric))
    check("loopback/private/metadata/file/credential URLs are rejected",
          safe_url("http://127.0.0.1/x") is None
          and safe_url("http://[::1]/x") is None
          and safe_url("http://169.254.169.254/") is None
          and safe_url("file:///etc/passwd") is None
          and safe_url("https://user:pass@example.com/") is None
          and safe_url("http://192.168.1.2/") is None,
          "public-url filter")


async def live_checks() -> None:
    print("== live ddgs search (outbound HTTPS) ==")
    started = time.perf_counter()
    sources: list[dict] = []
    last_error: BackendError | None = None
    # DuckDuckGo occasionally answers a valid query with zero rows while it is
    # throttling. Retry a bounded number of times and report honestly either way.
    for attempt in range(3):
        try:
            sources = await ddg_search(
                "Dota 2 latest patch notes",
                max_results=3,
                snippet_chars=200,
                timeout=15.0,
            )
        except BackendError as exc:
            last_error = exc
            sources = []
        if sources:
            break
        await asyncio.sleep(2 * (attempt + 1))
    elapsed = time.perf_counter() - started
    if not sources:
        detail = f"{last_error.code}: {last_error.message}" if last_error else "0 results after 3 attempts"
        check("live DuckDuckGo search succeeds", False, detail)
        return
    check("live DuckDuckGo search returns bounded results",
          len(sources) <= 3, f"{len(sources)} results in {elapsed:.2f} s")
    if sources:
        check("live results have clickable public URLs, titles and bounded snippets",
              all(source["url"].startswith(("http://", "https://")) for source in sources)
              and all(source["title"] for source in sources)
              and all(len(source["snippet"]) <= 200 for source in sources),
              repr(sources[0]))
        print(f"  first live result: {sources[0]['title']} — {sources[0]['url']}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="SPEC-002 search checks")
    parser.add_argument("--live", action="store_true",
                        help="also run a real outbound DuckDuckGo search")
    args = parser.parse_args()
    mocked_checks()
    live = args.live or os.environ.get("EL_VOICE_LIVE_SEARCH") == "1"
    if live:
        await live_checks()
    else:
        print("SKIP: live ddgs search (pass --live or set EL_VOICE_LIVE_SEARCH=1)")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
