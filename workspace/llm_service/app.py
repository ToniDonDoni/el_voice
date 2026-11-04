"""Internal streaming API shared by Local Voice and Remote Voice."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .context import ConversationStore
from .local_provider import LocalGemmaProvider
from .remote_provider import OpenCodeLangChainProvider, ProviderConfigurationError
from .search import SearchFailure

logger = logging.getLogger("el_voice.llm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

IDENTITY_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_USER_TEXT_CHARS = 8000


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_USER_TEXT_CHARS)


class BackendUnavailable(RuntimeError):
    pass


def make_provider(mode: str):
    if mode == "local":
        return LocalGemmaProvider()
    if mode == "remote":
        return OpenCodeLangChainProvider()
    raise ProviderConfigurationError("EL_VOICE_MODE must be local or remote")


def safe_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if code in {"search_timeout", "search_rate_limited", "search_unavailable", "invalid_search_query"}:
        return code
    allowed = {
        "provider_authentication_failed", "provider_rate_limited", "provider_timeout",
        "provider_unavailable_or_incompatible", "provider_request_failed",
        "provider_tool_loop_failed", "local_inference_failed", "search_tool_call_missing", "search_unavailable",
        "empty_generation",
    }
    if str(error) in allowed:
        return str(error)
    if isinstance(error, ProviderConfigurationError):
        return "provider_configuration_error"
    if isinstance(error, SearchFailure):
        return error.code
    return "generation_failed"


class LLMService:
    def __init__(self, provider: Any, mode: str, token_budget: int = 2400, max_sessions: int = 64):
        self.provider = provider
        self.mode = mode
        self.contexts = ConversationStore(token_budget=token_budget, max_sessions=max_sessions)
        self.ready = provider is not None
        self.error = None if self.ready else "provider_not_configured"

    async def shutdown(self) -> None:
        for session_id in tuple(self.contexts.sessions):
            await self.clear_session(session_id)
        if self.provider is not None:
            close = getattr(self.provider, "close", None)
            if close:
                await close()

    async def clear_session(self, session_id: str) -> None:
        state = self.contexts.clear(session_id)
        if state is None:
            return
        if state.cancel_event is not None:
            state.cancel_event.set()
        task = state.active_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task


def create_app(service: LLMService | None = None) -> FastAPI:
    selected_mode = os.getenv("EL_VOICE_MODE", "local").lower()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if service is not None:
            _app.state.service = service
        else:
            try:
                _app.state.service = LLMService(
                    make_provider(selected_mode), selected_mode,
                    token_budget=int(os.getenv("LLM_CONTEXT_TOKEN_BUDGET", "2400")),
                )
            except Exception as exc:
                _app.state.service = LLMService(None, selected_mode)
                _app.state.service.ready = False
                _app.state.service.error = safe_error_code(exc)
                logger.error("LLM provider configuration failed: %s", _app.state.service.error)
        try:
            yield
        finally:
            await _app.state.service.shutdown()

    application = FastAPI(title="EL Voice LLM Service", docs_url=None, redoc_url=None, lifespan=lifespan)

    @application.get("/healthz")
    async def healthz():
        current: LLMService | None = getattr(application.state, "service", service)
        ready = bool(current and current.ready)
        mode = current.mode if current else selected_mode
        return JSONResponse({"status": "ok" if ready else "not_ready", "ready": ready,
                             "mode": mode, "error": None if ready else (current.error if current else "starting")},
                            status_code=200 if ready else 503)

    @application.get("/api/status")
    async def status():
        current: LLMService | None = getattr(application.state, "service", service)
        mode = current.mode if current else selected_mode
        return {"ready": bool(current and current.ready), "mode": mode,
                "provider": "Gemma local" if mode == "local" else "OpenCode Go",
                "model": "gemma-3-1b-it-Q4_K_M" if mode == "local" else os.getenv("OPENCODE_GO_MODEL", "deepseek-v4.1-flash"),
                "error": None if current and current.ready else (current.error if current else "starting")}

    def require_service() -> LLMService:
        current: LLMService | None = getattr(application.state, "service", service)
        if current is None or not current.ready:
            raise HTTPException(503, "LLM provider is not ready")
        return current

    @application.post("/v1/sessions/{session_id}/turns/{turn_id}/stream")
    async def stream_turn(session_id: str, turn_id: str, request: TurnRequest):
        if not IDENTITY_RE.fullmatch(session_id) or not IDENTITY_RE.fullmatch(turn_id):
            raise HTTPException(400, "Invalid session or turn identity")
        current = require_service()
        try:
            state = current.contexts.get(session_id)
        except OverflowError as exc:
            raise HTTPException(429, "The LLM service has reached its active session limit") from exc
        state.reserved = True

        async def response_body() -> AsyncIterator[bytes]:
            async with state.lock:
                state.reserved = False
                cancelled = asyncio.Event()
                state.active_turn_id = turn_id
                state.cancel_event = cancelled
                queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=12)
                producer_finished = asyncio.Event()
                reply_parts: list[str] = []

                async def produce() -> None:
                    try:
                        history = current.contexts.trimmed_history(state.history)
                        async for event in current.provider.stream(request.text, history, cancelled):
                            if cancelled.is_set():
                                break
                            event_type = event.get("type")
                            if event_type == "delta":
                                piece = str(event.get("text", ""))[:4000]
                                if piece:
                                    reply_parts.append(piece)
                                    await queue.put(("event", {"type": "delta", "text": piece}))
                            elif event_type == "sources":
                                sources = event.get("sources", [])
                                await queue.put(("event", {"type": "sources", "sources": sources[:5]}))
                            elif event_type == "tool_call":
                                await queue.put(("event", {"type": "tool_call", "name": "web_search",
                                                             "result_count": min(5, max(0, int(event.get("result_count", 0))))}))
                        if not cancelled.is_set() and reply_parts:
                            current.contexts.remember(state, request.text, "".join(reply_parts))
                            await queue.put(("event", {"type": "done"}))
                        elif not cancelled.is_set():
                            await queue.put(("event", {"type": "error", "code": "empty_generation",
                                                         "message": "The selected language provider returned no answer."}))
                    except asyncio.CancelledError:
                        cancelled.set()
                    except Exception as exc:
                        code = safe_error_code(exc)
                        logger.warning("LLM turn failed: code=%s mode=%s", code, current.mode)
                        await queue.put(("event", {"type": "error", "code": code,
                                                     "message": "The selected language provider could not complete this turn."}))
                    finally:
                        producer_finished.set()

                task = asyncio.create_task(produce(), name=f"llm-{session_id}-{turn_id}")
                state.active_task = task
                try:
                    while True:
                        try:
                            kind, event = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            if producer_finished.is_set():
                                break
                            continue
                        yield (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    await task
                finally:
                    cancelled.set()
                    if not task.done():
                        task.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await task
                    state.active_turn_id = None
                    state.cancel_event = None
                    state.active_task = None

        return StreamingResponse(response_body(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @application.post("/v1/sessions/{session_id}/turns/{turn_id}/cancel", status_code=202)
    async def cancel_turn(session_id: str, turn_id: str):
        if not IDENTITY_RE.fullmatch(session_id) or not IDENTITY_RE.fullmatch(turn_id):
            raise HTTPException(400, "Invalid session or turn identity")
        current = require_service()
        state = current.contexts.sessions.get(session_id)
        if state and state.active_turn_id == turn_id:
            if state.cancel_event is not None:
                state.cancel_event.set()
            if state.active_task is not None and not state.active_task.done():
                state.active_task.cancel()
        return {"cancelled": bool(state and state.active_turn_id == turn_id)}

    @application.delete("/v1/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        if not IDENTITY_RE.fullmatch(session_id):
            raise HTTPException(400, "Invalid session identity")
        current = require_service()
        await current.clear_session(session_id)
        return None

    return application


app = create_app()
