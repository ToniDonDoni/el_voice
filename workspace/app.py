"""EL Voice browser application.

The browser captures microphone audio on the Mac and streams 16 kHz PCM frames
through the existing Nginx WebSocket proxy to this service. The service runs a
streaming VAD, final Whisper transcription, local Gemma generation and Silero
TTS, and returns tagged audio chunks for incremental playback. The voice service
itself is never published directly; only the loopback-bound proxy port is.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from elvoice import settings
from elvoice.engines import EngineError, EngineManager
from elvoice.llm_client import LlmClient, fetch_status as fetch_llm_status
from elvoice.pipeline import Metrics, VoiceSession

SOURCE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SOURCE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "connect-src 'self' ws: wss:; media-src 'self' blob:; img-src 'self' data:; "
    "base-uri 'none'; frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self)",
    "Cache-Control": "no-store",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("elvoice.app")


class AppState:
    """Process-wide objects: one engine set and at most one active session."""

    def __init__(self) -> None:
        self.cfg: dict | None = None
        self.settings_error: str | None = None
        try:
            self.cfg = settings.load_settings()
        except Exception as exc:  # noqa: BLE001 - reported through the status endpoint
            self.settings_error = f"{type(exc).__name__}: {exc}"
            logger.error("settings error: %s", self.settings_error)
        self.manager = EngineManager(self.cfg, settings.model_root()) if self.cfg else None
        self.llm_url = settings.llm_service_url()
        self.metrics = Metrics()
        self.active: dict | None = None


STATE = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if STATE.manager is not None:
        STATE.manager.start()
    try:
        yield
    finally:
        active = STATE.active
        STATE.active = None
        if active is not None:
            try:
                await active["session"].stop()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass


app = FastAPI(
    title="EL Voice",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        if name not in response.headers:
            response.headers[name] = value
    return response


async def _llm_status() -> dict:
    try:
        return await fetch_llm_status(STATE.llm_url, timeout=2.0)
    except Exception as exc:  # noqa: BLE001 - readiness must stay available
        return {
            "mode": "unknown",
            "ready": False,
            "error": f"The internal LLM service is not reachable ({type(exc).__name__}).",
        }


async def _status_payload() -> dict:
    manager = STATE.manager
    active = STATE.active
    llm = await _llm_status()
    llm_ready = bool(llm.get("ready"))
    components = manager.status() if manager is not None else {
        "ready": False,
        "error": STATE.settings_error,
        "components": {},
    }
    components = {
        **components,
        "components": {
            **dict(components.get("components") or {}),
            "llm": {
                "status": "ready" if llm_ready else ("error" if llm.get("error") else "pending"),
                "detail": llm.get("error") or "",
                "mode": llm.get("mode"),
            },
        },
    }
    session_state = active["session"].state if active is not None else "idle"
    payload = {
        "stage": "voice",
        "settings_error": STATE.settings_error,
        "pipeline_ready": bool(
            manager is not None and manager.is_ready and llm_ready
        ),
        "components": components,
        "llm": {
            "mode": llm.get("mode", "unknown"),
            "ready": llm_ready,
            "error": llm.get("error"),
            "context": llm.get("context"),
            "search": llm.get("search"),
            "limits": llm.get("limits"),
        },
        "session": {
            "active": active is not None,
            "state": session_state,
            "last_prompt": dict(active["session"].last_prompt) if active is not None else {},
        },
        "metrics": STATE.metrics.snapshot(),
        "config": {
            "headphones_required": True,
            "echo_cancellation": "best_effort",
            "audio_persisted": False,
            "transcript_persisted": False,
        },
    }
    if STATE.cfg is not None:
        payload["config"].update({
            "input_sample_rate": STATE.cfg["audio"]["input_sample_rate"],
            "output_sample_rate": STATE.cfg["audio"]["output_sample_rate"],
            "vad": {
                "threshold": STATE.cfg["vad"]["threshold"],
                "neg_threshold": STATE.cfg["vad"]["neg_threshold"],
                "min_speech_ms": STATE.cfg["vad"]["min_speech_ms"],
                "end_silence_ms": STATE.cfg["vad"]["end_silence_ms"],
                "pre_roll_ms": STATE.cfg["audio"]["pre_roll_ms"],
            },
            "conversation": {
                "history_turns": STATE.cfg["conversation"]["history_turns"],
                "max_prompt_chars": STATE.cfg["conversation"]["max_prompt_chars"],
            },
            "stt_language": STATE.cfg["stt"]["language"],
            "tts_speaker": STATE.cfg["tts"]["speaker"],
        })
    return payload


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """The HTTP process is alive; this says nothing about model readiness."""
    return JSONResponse({"status": "ok"}, headers=SECURITY_HEADERS)


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(await _status_payload(), headers=SECURITY_HEADERS)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE, media_type="text/html; charset=utf-8")


@app.get("/audio/tts-test.wav")
async def tts_test_audio() -> Response:
    sample = settings.data_root() / "tts-test.wav"
    if not sample.is_file():
        return JSONResponse({"error": "no test sample yet"}, status_code=404)
    return FileResponse(sample, media_type="audio/wav")


@app.websocket("/ws")
async def websocket_session(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("ws: client connected")
    send_lock = asyncio.Lock()
    session: VoiceSession | None = None

    async def send_json(payload: dict) -> None:
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass

    async def send_audio(payload: bytes) -> None:
        async with send_lock:
            try:
                await websocket.send_bytes(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is not None:
                if session is not None:
                    await session.feed(data)
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            if kind == "start":
                logger.info("ws: start requested")
                if STATE.cfg is None or STATE.manager is None:
                    await send_json({
                        "type": "error", "code": "config_error",
                        "message": "Application settings are invalid.", "retryable": False,
                    })
                    continue
                previous = STATE.active
                if previous is not None and previous["socket"] is not websocket:
                    logger.info("ws: replacing stale or busy session")
                    if previous["socket"].client_state == WebSocketState.DISCONNECTED:
                        try:
                            await previous["session"].stop()
                        except Exception:  # noqa: BLE001
                            pass
                        STATE.active = None
                    else:
                        await send_json({
                            "type": "error", "code": "busy",
                            "message": "Another voice session is already active.",
                            "retryable": True,
                        })
                        continue
                if not STATE.manager.is_ready:
                    await send_json({
                        "type": "error", "code": "not_ready",
                        "message": "Local speech models are still loading.",
                        "retryable": True,
                    })
                    continue
                llm = await _llm_status()
                if not llm.get("ready"):
                    await send_json({
                        "type": "error", "code": "llm_not_ready",
                        "message": llm.get("error")
                        or "The internal LLM service is still loading.",
                        "retryable": True,
                    })
                    continue
                session = VoiceSession(
                    str(uuid.uuid4()), STATE.cfg, STATE.manager, STATE.metrics,
                    send_json, send_audio,
                    llm=LlmClient(STATE.llm_url),
                )
                try:
                    ready = await session.start()
                except EngineError as exc:
                    logger.warning("session start failed: %s", type(exc).__name__)
                    session = None
                    await send_json({
                        "type": "error", "code": "not_ready",
                        "message": str(exc) or "The voice pipeline is not ready.",
                        "retryable": True,
                    })
                    continue
                STATE.active = {"session": session, "socket": websocket}
                logger.info("ws: session %s active", session.session_id)
                await send_json(ready)
            elif kind == "stop":
                logger.info("ws: stop requested")
                stopped = session
                if stopped is not None:
                    await stopped.stop()
                if STATE.active is not None and STATE.active["session"] is stopped:
                    STATE.active = None
                session = None
            elif kind == "playback_done" and session is not None:
                await session.playback_done(int(payload.get("turn_id", -1)))
            elif kind == "playback_stopped" and session is not None:
                session.client_playback_stopped(
                    int(payload.get("turn_id", -1)), float(payload.get("elapsed_ms", -1)),
                )
            elif kind == "ping":
                await send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("ws: client disconnected")
        if session is not None:
            try:
                await session.stop()
            except Exception:  # noqa: BLE001
                pass
        if STATE.active is not None and STATE.active["session"] is session:
            STATE.active = None


def main() -> None:
    parser = argparse.ArgumentParser(description="EL Voice browser conversation service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
