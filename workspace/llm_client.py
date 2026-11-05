"""Voice-side client for the internal LLM service contract."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, AsyncIterator

from observability import log_structured

logger = logging.getLogger("el_voice.llm_client")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


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
                     cancel_event: threading.Event,
                     search_enabled: bool = False) -> AsyncIterator[dict[str, Any]]:
        import httpx

        endpoint = f"{self.base_url}/v1/sessions/{session_id}/turns/{turn_id}/stream"
        timeout = httpx.Timeout(connect=3, read=None, write=5, pool=3)
        event_count = 0
        delta_count = 0
        response_chars = 0
        log_structured(logger, "llm_client_stream_started", session_id=session_id,
                       turn_id=turn_id, trace_id=turn_id)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=self.transport) as client:
            try:
                async with client.stream(
                    "POST", endpoint, json={"text": text, "trace_id": turn_id,
                                             "search_enabled": search_enabled},
                ) as response:
                    log_structured(logger, "llm_client_response_headers", session_id=session_id,
                                   turn_id=turn_id, trace_id=turn_id,
                                   status_code=response.status_code,
                                   content_type=response.headers.get("content-type", "").split(";", 1)[0])
                    if response.status_code != 200:
                        log_structured(logger, "llm_client_http_error", session_id=session_id,
                                       turn_id=turn_id, trace_id=turn_id,
                                       status_code=response.status_code)
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
                            log_structured(logger, "llm_client_event_parse_failed",
                                           session_id=session_id, turn_id=turn_id,
                                           trace_id=turn_id, event_index=event_count + 1,
                                           error_type=type(exc).__name__)
                            raise LLMServiceError("invalid_llm_service_event") from exc
                        event_count += 1
                        response_trace_id = event.get("trace_id")
                        if response_trace_id is not None and response_trace_id != turn_id:
                            log_structured(logger, "llm_client_trace_mismatch",
                                           session_id=session_id, turn_id=turn_id,
                                           trace_id=turn_id, response_trace_id=response_trace_id,
                                           event_index=event_count)
                            raise LLMServiceError("llm_service_trace_id_mismatch")
                        if event.get("type") == "error":
                            code = str(event.get("code") or "generation_failed")
                            log_structured(logger, "llm_client_error_event_received",
                                           session_id=session_id, turn_id=turn_id,
                                           trace_id=turn_id, event_index=event_count,
                                           error_code=code)
                            raise LLMServiceError(code)
                        if event.get("type") == "done":
                            completed = True
                            log_structured(logger, "llm_client_stream_completed",
                                           session_id=session_id, turn_id=turn_id,
                                           trace_id=turn_id, event_count=event_count,
                                           delta_count=delta_count,
                                           response_chars=response_chars,
                                           received_done=True)
                            break
                        text_chars = len(event.get("text", "")) if isinstance(event.get("text"), str) else 0
                        if event.get("type") == "delta":
                            delta_count += 1
                            response_chars += text_chars
                        log_structured(logger, "llm_client_event_yielded",
                                       session_id=session_id, turn_id=turn_id,
                                       trace_id=turn_id, event_index=event_count,
                                       event_type=event.get("type"), text_chars=text_chars,
                                       delta_count=delta_count)
                        yield event
                    if not cancel_event.is_set() and not completed:
                        log_structured(logger, "llm_client_stream_truncated",
                                       session_id=session_id, turn_id=turn_id,
                                       trace_id=turn_id, event_count=event_count,
                                       delta_count=delta_count,
                                       response_chars=response_chars)
                        raise LLMServiceError("truncated_llm_service_stream")
            except asyncio.CancelledError:
                log_structured(logger, "llm_client_stream_cancelled", session_id=session_id,
                               turn_id=turn_id, trace_id=turn_id,
                               event_count=event_count, delta_count=delta_count,
                               response_chars=response_chars)
                raise
            except Exception as exc:
                log_structured(logger, "llm_client_stream_failed", session_id=session_id,
                               turn_id=turn_id, trace_id=turn_id,
                               event_count=event_count, delta_count=delta_count,
                               response_chars=response_chars,
                               error_code=getattr(exc, "code", type(exc).__name__))
                raise
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
