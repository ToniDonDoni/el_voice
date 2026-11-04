"""Shared helpers for running the internal LLM service in tests.

The service is started as a real uvicorn server on a loopback port and driven
through its public WebSocket/HTTP boundary. No internal method is called to
produce the events under test.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "workspace"))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fetch_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def start_server(app) -> tuple[uvicorn.Server, threading.Thread, int]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port,
        log_level="warning", access_log=False, lifespan="on",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread, port


def wait_http(url: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            return fetch_json(url, timeout=2)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = str(exc)
            time.sleep(0.05)
    raise RuntimeError(f"service did not answer {url}: {last_error}")


def wait_status(base_url: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    status: dict = {}
    while time.monotonic() < deadline:
        status = wait_http(base_url + "/status", timeout=2)
        if status.get("ready"):
            return status
        time.sleep(0.05)
    raise RuntimeError(f"service did not become ready: {status}")


def stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


class WsClient:
    def __init__(self, websocket):
        self.websocket = websocket
        self.queue: asyncio.Queue = asyncio.Queue()
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            while True:
                raw = await self.websocket.recv()
                if isinstance(raw, str):
                    await self.queue.put(json.loads(raw))
        except Exception:  # noqa: BLE001 - normal close ends the reader
            await self.queue.put(None)

    async def send(self, payload: dict) -> None:
        await self.websocket.send(json.dumps(payload))

    async def next(self, timeout: float = 10.0) -> dict | None:
        message = await asyncio.wait_for(self.queue.get(), timeout=timeout)
        return message

    async def until(self, predicate, timeout: float = 10.0) -> list[dict]:
        seen: list[dict] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out; saw {seen!r}")
            message = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            if message is None:
                raise ConnectionError(f"connection closed; saw {seen!r}")
            seen.append(message)
            if predicate(message):
                return seen

    async def close(self) -> None:
        self.reader.cancel()
        try:
            await self.reader
        except (asyncio.CancelledError, Exception):
            pass
