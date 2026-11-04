"""Typed web search with bounded results and cancellable worker processes."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import sys
import time
import uuid
from urllib.parse import urlparse


class SearchFailure(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def public_https_url(value: str) -> bool:
    parsed = urlparse(value)
    host = parsed.hostname
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        return False
    host = host.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return "." in host and not host.endswith(".test")


def search_sync(query: str, backend: str, max_results: int,
                max_snippet_chars: int, timeout_seconds: float) -> list[dict[str, str]]:
    """Run bounded DDGS search; called only inside the disposable worker process."""
    from ddgs import DDGS

    raw = DDGS(timeout=max(1, int(timeout_seconds))).text(
        query[:240], region="ru-ru", safesearch="moderate",
        max_results=max_results, backend=backend,
    )
    output = []
    for item in raw[:max_results]:
        url = str(item.get("href") or item.get("url") or "")
        if not public_https_url(url):
            continue
        output.append({
            "title": str(item.get("title") or "Search result")[:200],
            "url": url[:2048],
            "snippet": str(item.get("body") or item.get("snippet") or "")[:max_snippet_chars],
        })
    return output


class WebSearch:
    def __init__(self, backend: str | None = None, max_results: int = 5,
                 timeout_seconds: float | None = None, max_snippet_chars: int | None = None,
                 max_concurrent: int = 2):
        self.backend = backend or os.getenv("WEB_SEARCH_BACKEND", "auto")
        if self.backend not in {"auto", "duckduckgo", "bing", "brave", "google", "mojeek", "startpage", "yandex", "yahoo", "wikipedia"}:
            raise ValueError("WEB_SEARCH_BACKEND is not a supported search provider")
        timeout_seconds = timeout_seconds if timeout_seconds is not None else float(
            os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "10")
        )
        max_snippet_chars = max_snippet_chars if max_snippet_chars is not None else int(
            os.getenv("WEB_SEARCH_MAX_SNIPPET_CHARS", "500")
        )
        self.max_results = min(5, max(1, max_results))
        self.timeout_seconds = min(120.0, max(1.0, timeout_seconds))
        self.max_snippet_chars = min(20_000, max(100, max_snippet_chars))
        self.worker_slots = asyncio.Semaphore(min(4, max(1, max_concurrent)))

    async def _spawn_worker(self):
        env = {
            name: os.environ[name]
            for name in ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH", "SSL_CERT_FILE",
                         "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")
            if name in os.environ
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return await asyncio.create_subprocess_exec(
            sys.executable, "-m", "llm_service.search_worker",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=env,
        )

    async def _terminate_worker(self, process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _run_worker(self, query: str) -> list[dict[str, str]]:
        process = await self._spawn_worker()
        request = json.dumps({
            "query": query[:240], "backend": self.backend,
            "max_results": self.max_results,
            "max_snippet_chars": self.max_snippet_chars,
            "timeout_seconds": self.timeout_seconds,
        }).encode()
        try:
            stdout, _ = await process.communicate(input=request)
        except asyncio.CancelledError:
            await self._terminate_worker(process)
            raise
        except Exception:
            await self._terminate_worker(process)
            raise
        if process.returncode != 0:
            raise SearchFailure("search_unavailable")
        try:
            payload = json.loads(stdout)
        except (TypeError, ValueError) as exc:
            raise SearchFailure("search_unavailable") from exc
        if payload.get("error"):
            raise SearchFailure(payload["error"])
        results = payload.get("results")
        if not isinstance(results, list):
            raise SearchFailure("search_unavailable")
        return results[:self.max_results]

    async def search(self, query: str) -> list[dict[str, str]]:
        query = query.strip()
        if not query or len(query) > 240:
            raise SearchFailure("invalid_search_query")
        try:
            async with self.worker_slots:
                return await asyncio.wait_for(self._run_worker(query), self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise SearchFailure("search_timeout") from exc
        except SearchFailure:
            raise
        except Exception as exc:
            name = type(exc).__name__.lower()
            code = "search_rate_limited" if "rate" in name or "limit" in str(exc).lower() else "search_unavailable"
            raise SearchFailure(code) from exc

    async def tool_payload(self, query: str) -> str:
        results = await self.search(query)
        bounded = []
        max_payload_chars = max(6000, self.max_snippet_chars + 1500)
        for item in results[:self.max_results]:
            candidate = {"title": str(item.get("title", "Search result"))[:150],
                         "url": str(item.get("url", ""))[:512],
                         "snippet": str(item.get("snippet", ""))[:self.max_snippet_chars]}
            if not public_https_url(candidate["url"]):
                continue
            trial = json.dumps({"query": query[:240], "results": [*bounded, candidate]}, ensure_ascii=False)
            if len(trial) > max_payload_chars:
                break
            bounded.append(candidate)
        return json.dumps({"query": query[:240], "results": bounded}, ensure_ascii=False)


def make_langchain_search_tool(search: WebSearch, observer=None):
    from langchain.tools import ToolRuntime, tool

    @tool("web_search")
    async def web_search(query: str, runtime: ToolRuntime) -> str:
        """Search public web pages for current information. Input one concise search query."""
        invocation_id = uuid.uuid4().hex
        call_id = runtime.tool_call_id
        started_at = time.monotonic()
        if observer:
            observer("tool_call_started", tool_name="web_search", tool_invocation_id=invocation_id,
                     tool_call_id=call_id, backend=getattr(search, "backend", "configured"))
        try:
            payload = await search.tool_payload(query)
            try:
                result_count = len(json.loads(payload).get("results", []))
            except (TypeError, ValueError, AttributeError):
                result_count = 0
            if observer:
                observer("tool_call_completed", tool_name="web_search",
                         tool_invocation_id=invocation_id, tool_call_id=call_id,
                         result_count=result_count,
                         duration_ms=round((time.monotonic() - started_at) * 1000, 1))
            return payload
        except Exception as exc:
            if observer:
                code = getattr(exc, "code", None)
                error_code = str(code if isinstance(code, str) else type(exc).__name__)[:100]
                observer("tool_call_failed", tool_name="web_search",
                         tool_invocation_id=invocation_id, tool_call_id=call_id,
                         error_code=error_code,
                         duration_ms=round((time.monotonic() - started_at) * 1000, 1))
            raise

    return web_search
