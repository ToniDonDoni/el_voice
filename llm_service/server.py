"""FastAPI/WebSocket server for the internal EL Voice LLM service.

One WebSocket connection is one conversation. The server owns a bounded
``ConversationHistory`` per connection, builds every prompt from the workspace
system prompt, and allows one active generation per connection. Cancelling stops
the backend iterator and the blocking local worker or remote HTTP stream.

Content-free: prompts, answers, tool arguments and search snippets are not
logged. ``/status`` reports only modes, readiness, limits and prompt statistics.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import threading
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from .backends.base import Backend, BackendError
from .config import ServiceConfig, load_config, workspace_dir

# ``python -m llm_service`` must work without an externally configured
# PYTHONPATH; the workspace settings module is imported from the mounted source.
_workspace = workspace_dir()
if str(_workspace) not in sys.path:
    sys.path.insert(0, str(_workspace))

from elvoice.context import ConversationHistory, build_messages  # noqa: E402

logger = logging.getLogger("llm_service.server")

MAX_PROMPT_TEXT_CHARS = 12000
CANCEL_GRACE_SECONDS = 5.0


@dataclass
class ConnectionState:
    history: ConversationHistory
    websocket: WebSocket
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task | None = None
    cancel: threading.Event | None = None
    turn_id: int | None = None
    answer: list[str] = field(default_factory=list)


class ServiceState:
    def __init__(self, config: ServiceConfig, backend: Backend):
        self.config = config
        self.backend = backend
        self.ready = False
        self.error: str | None = None
        self.last_prompt: dict = {}
        self.active_sessions = 0
        self.start_task: asyncio.Task | None = None

    async def start_backend(self) -> None:
        try:
            await self.backend.start()
            self.ready = True
            self.error = None
            logger.info("backend ready in mode %s", self.config.mode)
        except BackendError as exc:
            self.error = exc.message
            self.ready = False
            logger.error("backend failed to start: %s", exc.code)
        except Exception as exc:  # noqa: BLE001 - reported through /status
            message = f"{type(exc).__name__}: {exc}"
            try:
                from .backends.remote import redact

                message = redact(message, self.config.remote.api_key)
            except Exception:  # noqa: BLE001 - redaction is best effort
                message = message[:200]
            self.error = message
            self.ready = False
            logger.error("backend failed to start: %s", type(exc).__name__)

    def status_payload(self) -> dict:
        public = self.config.public_status()
        backend_status = {}
        try:
            backend_status = self.backend.status()
        except Exception as exc:  # noqa: BLE001 - status must not crash
            backend_status = {"error": type(exc).__name__}
        return {
            "mode": self.config.mode,
            "ready": self.ready,
            "error": self.error,
            "context": public["context"],
            "limits": public["limits"],
            "search": public["search"],
            "remote": public["remote"],
            "last_prompt": dict(self.last_prompt),
            "active_sessions": self.active_sessions,
            "backend": backend_status,
            "content_logging": False,
        }


def build_backend(config: ServiceConfig) -> Backend:
    if config.mode == "local":
        from .backends.local import LocalBackend

        return LocalBackend(config)
    from .backends.remote import RemoteBackend

    return RemoteBackend(config)


def create_app(backend: Backend | None = None, config: ServiceConfig | None = None) -> FastAPI:
    """Create the service app; tests inject a deterministic backend."""
    if config is None:
        config = load_config()
    if backend is None:
        backend = build_backend(config)
    state = ServiceState(config, backend)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        state.start_task = asyncio.create_task(state.start_backend())
        try:
            yield
        finally:
            if state.start_task is not None and not state.start_task.done():
                state.start_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await state.start_task
            with contextlib.suppress(Exception):
                await state.backend.close()

    app = FastAPI(
        title="EL Voice LLM service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.service = state

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse(state.status_payload())

    @app.websocket("/generate")
    async def generate_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        state.active_sessions += 1
        conn = ConnectionState(
            history=ConversationHistory(config.history_turns),
            websocket=websocket,
        )
        logger.info("ws: connection opened (active=%d)", state.active_sessions)
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text = message.get("text")
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await _send(conn, {"type": "error", "code": "invalid_json",
                                       "message": "The service received invalid JSON.",
                                       "retryable": False})
                    continue
                if not isinstance(payload, dict):
                    continue
                kind = payload.get("type")
                if kind == "generate":
                    await _handle_generate(app, state, conn, payload)
                elif kind == "cancel":
                    await _handle_cancel(conn, payload)
                elif kind == "reset":
                    await _stop_active(conn)
                    conn.history.reset()
                    state.last_prompt = {}
                    await _send(conn, {"type": "reset"})
                else:
                    await _send(conn, {
                        "type": "error", "code": "unknown_message",
                        "message": f"Unknown message type: {kind!r}",
                        "retryable": False,
                    })
        except WebSocketDisconnect:
            pass
        finally:
            await _stop_active(conn)
            conn.history.reset()
            state.active_sessions = max(0, state.active_sessions - 1)
            logger.info("ws: connection closed (active=%d)", state.active_sessions)

    return app


async def _send(conn: ConnectionState, payload: dict) -> None:
    async with conn.send_lock:
        if conn.websocket.client_state == WebSocketState.DISCONNECTED:
            return
        try:
            await conn.websocket.send_json(payload)
        except (RuntimeError, WebSocketDisconnect):
            return


async def _handle_generate(
    app: FastAPI, state: ServiceState, conn: ConnectionState, payload: dict
) -> None:
    turn_id = payload.get("turn_id")
    text = payload.get("text")
    if not isinstance(turn_id, int) or isinstance(turn_id, bool):
        await _send(conn, {
            "type": "error", "code": "invalid_turn_id",
            "message": "generate requires an integer turn_id.", "retryable": False,
        })
        return
    if not isinstance(text, str) or not text.strip():
        await _send(conn, {
            "type": "error", "code": "invalid_text", "turn_id": turn_id,
            "message": "generate requires non-empty text.", "retryable": False,
        })
        return
    if len(text) > MAX_PROMPT_TEXT_CHARS:
        await _send(conn, {
            "type": "error", "code": "text_too_long", "turn_id": turn_id,
            "message": "The user text exceeds the service limit.", "retryable": False,
        })
        return

    # The voice client is expected to cancel before starting a new turn; if it
    # does not, keep the single-active-generation invariant rather than running
    # two generations on one model.
    if conn.task is not None and not conn.task.done():
        if conn.cancel is not None:
            conn.cancel.set()
        conn.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(conn.task, timeout=CANCEL_GRACE_SECONDS)

    cancel = threading.Event()
    conn.cancel = cancel
    conn.turn_id = turn_id
    conn.answer = []
    conn.task = asyncio.create_task(
        _run_generation(state, conn, turn_id, text.strip(), cancel)
    )


async def _handle_cancel(conn: ConnectionState, payload: dict) -> None:
    turn_id = payload.get("turn_id")
    if conn.task is None or conn.task.done():
        await _send(conn, {
            "type": "cancelled", "turn_id": turn_id,
            "reason": "no_active_generation",
        })
        return
    if isinstance(turn_id, int) and conn.turn_id is not None and turn_id != conn.turn_id:
        await _send(conn, {
            "type": "cancelled", "turn_id": turn_id,
            "reason": "stale_turn_id",
        })
        return
    if conn.cancel is not None:
        conn.cancel.set()
    conn.task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(conn.task, timeout=CANCEL_GRACE_SECONDS)


async def _stop_active(conn: ConnectionState) -> None:
    if conn.cancel is not None:
        conn.cancel.set()
    if conn.task is not None and not conn.task.done():
        conn.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(conn.task, timeout=CANCEL_GRACE_SECONDS)
    await asyncio.sleep(0)


async def _pump(iterator, queue: asyncio.Queue) -> None:
    try:
        async for event in iterator:
            queue.put_nowait(("event", event))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - handed to the generation task
        queue.put_nowait(("error", exc))
    finally:
        queue.put_nowait(("end", None))


async def _run_generation(
    state: ServiceState,
    conn: ConnectionState,
    turn_id: int,
    text: str,
    cancel: threading.Event,
) -> None:
    pump: asyncio.Task | None = None
    cancelled_sent = False
    assistant: list[str] = []
    try:
        if not state.ready:
            await _send(conn, {
                "type": "error", "turn_id": turn_id, "code": "not_ready",
                "message": state.error or "The LLM service is not ready.",
                "retryable": True,
            })
            return

        messages, stats = build_messages(
            state.config.system_prompt,
            conn.history.as_list(),
            text,
            state.config.max_prompt_chars,
        )
        state.last_prompt = dict(stats)
        await _send(conn, {"type": "start", "turn_id": turn_id, "prompt": stats})

        iterator = state.backend.stream(messages, cancel)
        queue: asyncio.Queue = asyncio.Queue()
        pump = asyncio.create_task(_pump(iterator, queue))

        while True:
            if cancel.is_set():
                break
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if kind == "end":
                break
            if kind == "error":
                raise value
            event = value
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if cancel.is_set():
                break
            if kind == "delta":
                chunk = event.get("text")
                if not isinstance(chunk, str) or not chunk:
                    continue
                assistant.append(chunk)
                await _send(conn, {"type": "delta", "turn_id": turn_id, "text": chunk})
            elif kind == "tool":
                await _send(conn, {
                    "type": "tool", "turn_id": turn_id,
                    "name": event.get("name", ""), "query": event.get("query", ""),
                })
            elif kind == "sources":
                await _send(conn, {
                    "type": "sources", "turn_id": turn_id,
                    "sources": list(event.get("sources") or []),
                })
            else:
                logger.warning("backend produced an unknown event type: %r", kind)

        if cancel.is_set():
            await _send(conn, {"type": "cancelled", "turn_id": turn_id})
            cancelled_sent = True
            return

        answer = "".join(assistant)
        if answer.strip():
            conn.history.add_exchange(text, answer)
        await _send(conn, {
            "type": "done", "turn_id": turn_id, "chars": len(answer),
        })
    except asyncio.CancelledError:
        if not cancelled_sent:
            with contextlib.suppress(Exception):
                await asyncio.shield(_send(conn, {"type": "cancelled", "turn_id": turn_id}))
        raise
    except BackendError as exc:
        logger.error("generation failed: %s", exc.code)
        if cancel.is_set():
            await _send(conn, {"type": "cancelled", "turn_id": turn_id})
        else:
            await _send(conn, {
                "type": "error", "turn_id": turn_id, "code": exc.code,
                "message": exc.message, "retryable": exc.retryable,
            })
    except Exception as exc:  # noqa: BLE001 - never crash the connection
        logger.error("generation failed: %s", type(exc).__name__)
        if cancel.is_set():
            await _send(conn, {"type": "cancelled", "turn_id": turn_id})
        else:
            await _send(conn, {
                "type": "error", "turn_id": turn_id, "code": "internal_error",
                "message": "The LLM service could not finish this turn.",
                "retryable": True,
            })
    finally:
        if pump is not None and not pump.done():
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
        if conn.task is asyncio.current_task():
            conn.task = None
            conn.cancel = None
            conn.turn_id = None
