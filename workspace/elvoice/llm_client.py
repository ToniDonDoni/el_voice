"""WebSocket client used by the voice application in both launch modes.

The voice pipeline no longer builds conversation prompts or loads Gemma. It
opens one connection to the internal LLM service per browser session, sends only
the current final utterance, forwards text deltas and search sources to the
browser, and cancels by ``turn_id`` on barge-in or stop.

Only the service is contacted; the voice container never gets an OpenCode Go key
or direct provider egress.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("elvoice.llm_client")

DEFAULT_LLM_URL = "ws://llm:8100"
EVENT_TIMEOUT = 120.0
CONNECT_TIMEOUT = 10.0


class LlmClientError(RuntimeError):
    """A generation could not be completed by the LLM service."""

    def __init__(self, message: str, code: str = "llm_error", retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def status_url(ws_url: str) -> str:
    """Derive the HTTP ``/status`` URL from the configured WebSocket URL."""
    parsed = urllib.parse.urlsplit(ws_url)
    scheme = "https" if parsed.scheme in ("wss", "https") else "http"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, "/status", "", ""))


def fetch_status_sync(ws_url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(status_url(ws_url), timeout=timeout) as response:
        return json.loads(response.read())


async def fetch_status(ws_url: str, timeout: float = 2.0) -> dict:
    """Read the service's content-free status without blocking the event loop."""
    return await asyncio.to_thread(fetch_status_sync, ws_url, timeout)


class LlmClient:
    """One WebSocket conversation with the internal LLM service."""

    def __init__(self, url: str | None = None):
        base = (url or DEFAULT_LLM_URL).rstrip("/")
        self.url = base if base.endswith("/generate") else base + "/generate"
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._queues: dict[int, asyncio.Queue] = {}
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        self._closed = False
        self._loop = asyncio.get_running_loop()
        await self._connect()
        await self._send({"type": "reset"})

    async def _connect(self) -> None:
        async with self._connect_lock:
            if self._ws is not None and not self._is_closed(self._ws):
                return
            import websockets

            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.url,
                        open_timeout=CONNECT_TIMEOUT,
                        max_size=None,
                        ping_interval=20,
                        ping_timeout=20,
                    ),
                    timeout=CONNECT_TIMEOUT,
                )
            except Exception as exc:  # noqa: BLE001 - reported as readiness failure
                self._ws = None
                raise LlmClientError(
                    "The internal LLM service is not reachable. Start the service or "
                    "check EL_VOICE_LLM_URL.",
                    code="llm_unreachable",
                    retryable=True,
                ) from exc
            self._reader = asyncio.create_task(self._recv_loop(self._ws))

    @staticmethod
    def _is_closed(websocket) -> bool:
        state = getattr(websocket, "state", None)
        if state is not None:
            name = getattr(state, "name", str(state))
            return name not in ("OPEN", "CONNECTING")
        return bool(getattr(websocket, "closed", False))

    async def close(self) -> None:
        self._closed = True
        websocket = self._ws
        self._ws = None
        if websocket is not None:
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                pass
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):
                pass
        self._fail_queues("The LLM service connection was closed.")

    async def _recv_loop(self, websocket) -> None:
        try:
            async for raw in websocket:
                if not isinstance(raw, str):
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                turn_id = payload.get("turn_id")
                queue = self._queues.get(turn_id) if isinstance(turn_id, int) else None
                if queue is not None:
                    queue.put_nowait(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - connection loss is reported to callers
            logger.warning("LLM service connection lost: %s", type(exc).__name__)
        finally:
            if self._ws is websocket:
                self._ws = None
            self._fail_queues("The LLM service connection was lost.")

    def _fail_queues(self, message: str) -> None:
        for queue in list(self._queues.values()):
            queue.put_nowait({
                "type": "transport_error",
                "code": "llm_connection_lost",
                "message": message,
                "retryable": True,
            })

    async def _send(self, payload: dict) -> None:
        websocket = self._ws
        if websocket is None or self._is_closed(websocket):
            raise LlmClientError(
                "The internal LLM service connection is not open.",
                code="llm_not_connected",
                retryable=True,
            )
        try:
            await websocket.send(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001 - surfaced as a connection error
            raise LlmClientError(
                "The internal LLM service connection failed while sending.",
                code="llm_send_failed",
                retryable=True,
            ) from exc

    # -- conversation ------------------------------------------------------
    async def generate(self, turn_id: int, text: str):
        """Yield service events for one turn, stopping at its terminal event."""
        if self._ws is None or self._is_closed(self._ws):
            await self._connect()
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[turn_id] = queue
        try:
            await self._send({"type": "generate", "turn_id": turn_id, "text": text})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=EVENT_TIMEOUT)
                except asyncio.TimeoutError as exc:
                    raise LlmClientError(
                        "The internal LLM service did not answer in time.",
                        code="llm_timeout",
                        retryable=True,
                    ) from exc
                kind = event.get("type")
                if kind == "transport_error":
                    raise LlmClientError(
                        event.get("message") or "The LLM service connection was lost.",
                        code=event.get("code", "llm_connection_lost"),
                        retryable=True,
                    )
                if kind == "error":
                    raise LlmClientError(
                        event.get("message") or "The LLM service reported an error.",
                        code=event.get("code", "llm_error"),
                        retryable=bool(event.get("retryable", True)),
                    )
                yield event
                if kind in ("done", "cancelled"):
                    return
        finally:
            self._queues.pop(turn_id, None)

    async def cancel(self, turn_id: int) -> None:
        if self._ws is None or self._is_closed(self._ws):
            return
        try:
            await self._send({"type": "cancel", "turn_id": turn_id})
        except LlmClientError:
            return
