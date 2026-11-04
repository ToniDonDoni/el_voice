"""Voice-side client for the internal LLM service contract."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, AsyncIterator

class LLMServiceError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class LLMServiceClient:
    def __init__(self, base_url: str | None = None, transport: Any = None):
        self.base_url = (base_url or os.getenv("LLM_SERVICE_URL", "http://llm:8001")).rstrip("/")
        self.transport = transport

    async def wait_ready(self, timeout_seconds: float = 30) -> dict[str, Any]:
        import httpx

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        last_error = "llm_service_unavailable"
        async with httpx.AsyncClient(timeout=2, follow_redirects=False, transport=self.transport) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    response = await client.get(f"{self.base_url}/healthz")
                    data = response.json()
                    if response.status_code == 200 and data.get("ready"):
                        return data
                    candidate = data.get("error")
                    if candidate in {"provider_configuration_error", "provider_authentication_failed",
                                     "provider_unavailable_or_incompatible", "provider_timeout"}:
                        last_error = str(candidate)
                        if candidate == "provider_configuration_error":
                            raise LLMServiceError(last_error)
                except LLMServiceError:
                    raise
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(0.25)
        raise LLMServiceError(last_error)

    async def stream(self, session_id: str, turn_id: str, text: str,
                     cancel_event: threading.Event) -> AsyncIterator[dict[str, Any]]:
        import httpx

        endpoint = f"{self.base_url}/v1/sessions/{session_id}/turns/{turn_id}/stream"
        timeout = httpx.Timeout(connect=3, read=None, write=5, pool=3)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=self.transport) as client:
            try:
                async with client.stream(
                    "POST", endpoint, json={"text": text, "trace_id": turn_id},
                ) as response:
                    if response.status_code != 200:
                        raise LLMServiceError(f"llm_service_http_{response.status_code}")
                    lines = response.aiter_lines().__aiter__()
                    completed = False
                    while not cancel_event.is_set():
                        next_line = asyncio.create_task(anext(lines))
                        try:
                            while not next_line.done() and not cancel_event.is_set():
                                await asyncio.wait({next_line}, timeout=0.05)
                            if cancel_event.is_set():
                                break
                            line = next_line.result()
                        except StopAsyncIteration:
                            break
                        finally:
                            if not next_line.done():
                                next_line.cancel()
                                await asyncio.gather(next_line, return_exceptions=True)
                        if cancel_event.is_set():
                            break
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise LLMServiceError("invalid_llm_service_event") from exc
                        response_trace_id = event.get("trace_id")
                        if response_trace_id is not None and response_trace_id != turn_id:
                            raise LLMServiceError("llm_service_trace_id_mismatch")
                        if event.get("type") == "error":
                            raise LLMServiceError(str(event.get("code") or "generation_failed"))
                        if event.get("type") == "done":
                            completed = True
                            break
                        yield event
                    if not cancel_event.is_set() and not completed:
                        raise LLMServiceError("truncated_llm_service_stream")
            finally:
                if cancel_event.is_set():
                    try:
                        await client.post(
                            f"{self.base_url}/v1/sessions/{session_id}/turns/{turn_id}/cancel",
                            timeout=2,
                        )
                    except httpx.HTTPError:
                        pass

    async def clear_session(self, session_id: str) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=2, follow_redirects=False, transport=self.transport) as client:
            try:
                await client.delete(f"{self.base_url}/v1/sessions/{session_id}")
            except httpx.HTTPError:
                # Disconnect cleanup is best effort; the service also bounds session count.
                pass
