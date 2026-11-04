"""Browser-facing HTTP and WebSocket server for the local voice assistant."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from voice_pipeline import (
    MODEL_ROOT,
    SAMPLE_RATE,
    VAD_FRAME_SAMPLES,
    Outbound,
    Turn,
    VoiceRuntime,
)

logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logger = logging.getLogger("el_voice")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def log_event(event: str, **fields: Any) -> None:
    """Write one structured local diagnostic record; never include audio payloads."""
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False, separators=(",", ":")))

ROOT = Path(__file__).resolve().parent
runtime = VoiceRuntime()
boot_task: asyncio.Task | None = None
active_session: "VoiceSession | None" = None
session_lock = asyncio.Lock()


@dataclass
class VoiceSession:
    session_id: str
    websocket: WebSocket
    vad: Any
    outgoing: asyncio.Queue[Outbound] = field(default_factory=lambda: asyncio.Queue(maxsize=8))
    current_turn: Turn | None = None
    turns: dict[str, Turn] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    tasks: set[asyncio.Task] = field(default_factory=set)
    interrupt_pending: dict[str, float] = field(default_factory=dict)
    stopped: bool = False

    def is_current(self, turn: Turn) -> bool:
        return not self.stopped and self.current_turn is turn and not turn.cancel_event.is_set()

    def remember_turn(self, turn: Turn) -> None:
        self.turns[turn.turn_id] = turn
        if len(self.turns) > 64:
            oldest = next(iter(self.turns))
            self.turns.pop(oldest, None)


def schedule_model_load() -> None:
    global boot_task
    if boot_task is None or boot_task.done():
        boot_task = asyncio.create_task(runtime.initialize(), name="load-local-models")


async def stop_runtime() -> None:
    global active_session
    if active_session:
        active_session.stopped = True
        for turn in active_session.turns.values():
            turn.cancel_event.set()
        for task in tuple(active_session.tasks):
            task.cancel()
    if boot_task and not boot_task.done():
        boot_task.cancel()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    schedule_model_load()
    try:
        yield
    finally:
        await stop_runtime()


app = FastAPI(title="EL Voice", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/healthz")
async def healthz() -> JSONResponse:
    # Process health lets Compose start the proxy; readiness is reported separately.
    return JSONResponse({"status": "ok", "ready": runtime.status["ready"], "stage": runtime.status["stage"]})


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse(runtime.status_snapshot())


@app.post("/api/client-error")
async def client_error(request: Request) -> Response:
    """Accept small local browser diagnostics when a WebSocket is unavailable."""
    try:
        event = await request.json()
        message = str(event.get("message", "Unknown browser error"))[:500]
    except Exception:
        message = "Malformed browser diagnostic"
    log_event("browser_error", error=message)
    return Response(status_code=204)


async def queue_event(session: VoiceSession, event: dict[str, Any], turn_id: str | None = None) -> None:
    if session.stopped:
        return
    await session.outgoing.put(Outbound(event=event, turn_id=turn_id))


def clear_outgoing(session: VoiceSession) -> None:
    while True:
        try:
            session.outgoing.get_nowait()
            session.outgoing.task_done()
        except asyncio.QueueEmpty:
            return


async def websocket_sender(session: VoiceSession) -> None:
    websocket = session.websocket
    while not session.stopped:
        message = await session.outgoing.get()
        try:
            if message.turn_id:
                turn = session.turns.get(message.turn_id)
                if turn is None or not session.is_current(turn):
                    continue
            if message.event:
                await websocket.send_text(json.dumps(message.event, ensure_ascii=False))
            if message.audio:
                if message.turn_id:
                    turn = session.turns.get(message.turn_id)
                    if turn is None or not session.is_current(turn):
                        continue
                await websocket.send_bytes(message.audio)
        except (WebSocketDisconnect, RuntimeError):
            session.stopped = True
            return
        finally:
            session.outgoing.task_done()


def _next(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _speakable_chunks(buffer: str, max_chars: int = 80) -> tuple[list[str], str]:
    chunks: list[str] = []
    remaining = buffer
    while remaining:
        boundary = None
        for index, character in enumerate(remaining):
            if character in ".!?…。！？" and index >= 18:
                boundary = index + 1
                while boundary < len(remaining) and remaining[boundary].isspace():
                    boundary += 1
                break
        if boundary is None and len(remaining) > max_chars:
            boundary = remaining.rfind(" ", 0, max_chars + 1)
            if boundary < 35:
                boundary = max_chars
        if boundary is None:
            break
        piece = remaining[:boundary].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[boundary:].lstrip()
    return chunks, remaining


async def _llm_items(session: VoiceSession, turn: Turn, user_text: str):
    """Consume a single-threaded llama.cpp stream through a bounded async queue."""
    loop = asyncio.get_running_loop()
    items: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=4)

    def push(kind: str, value: Any) -> bool:
        future = asyncio.run_coroutine_threadsafe(items.put((kind, value)), loop)
        try:
            future.result(timeout=30)
            return True
        except Exception:
            future.cancel()
            return False

    def worker() -> None:
        try:
            for delta in runtime.llm_stream(user_text, turn.cancel_event, session.history):
                if turn.cancel_event.is_set() or not push("delta", delta):
                    break
            push("done", None)
        except Exception as exc:
            push("error", (type(exc).__name__, str(exc)[:500]))

    worker_task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            kind, value = await items.get()
            if kind == "done":
                break
            if kind == "error":
                error_name, detail = value
                raise RuntimeError(f"{error_name}: {detail}")
            yield value
    finally:
        if turn.cancel_event.is_set() or session.stopped:
            while not worker_task.done():
                with suppress(asyncio.QueueEmpty):
                    items.get_nowait()
                await asyncio.sleep(0.005)
        await asyncio.shield(worker_task)


async def process_turn(session: VoiceSession, turn: Turn, audio: np.ndarray) -> None:
    answer_parts: list[str] = []
    transcript: str | None = None
    reply_logged = False
    try:
        text = await asyncio.to_thread(runtime.transcribe, audio)
        transcript = text
        if not session.is_current(turn):
            log_event("turn_discarded_after_transcription", session_id=session.session_id, turn_id=turn.turn_id)
            return
        if not text:
            log_event("empty_transcription", session_id=session.session_id, turn_id=turn.turn_id)
            await queue_event(session, {"type": "empty_turn", "turn_id": turn.turn_id}, turn.turn_id)
            return

        log_event("user_transcript", session_id=session.session_id, turn_id=turn.turn_id, text=text)

        await queue_event(
            session, {"type": "transcript", "turn_id": turn.turn_id, "text": text}, turn.turn_id
        )
        await queue_event(session, {"type": "generation_started", "turn_id": turn.turn_id}, turn.turn_id)
        text_buffer = ""
        async for delta in _llm_items(session, turn, text):
            if not session.is_current(turn):
                return
            answer_parts.append(delta)
            text_buffer += delta
            await queue_event(
                session, {"type": "text_delta", "turn_id": turn.turn_id, "text": delta}, turn.turn_id
            )
            chunks, text_buffer = _speakable_chunks(text_buffer)
            for chunk in chunks:
                if not session.is_current(turn):
                    return
                await synthesize_and_queue(session, turn, chunk)

        if not session.is_current(turn):
            return
        if text_buffer.strip():
            await synthesize_and_queue(session, turn, text_buffer.strip())
        answer = "".join(answer_parts).strip()
        if answer and session.is_current(turn):
            session.history.extend(
                [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
            )
            session.history = session.history[-8:]
            log_event("assistant_reply", session_id=session.session_id, turn_id=turn.turn_id, text=answer)
            reply_logged = True
        await queue_event(session, {"type": "turn_complete", "turn_id": turn.turn_id}, turn.turn_id)
    except asyncio.CancelledError:
        if answer_parts:
            log_event("assistant_reply_interrupted", session_id=session.session_id, turn_id=turn.turn_id,
                      text="".join(answer_parts).strip())
            reply_logged = True
        raise
    except Exception as exc:
        log_event("turn_error", session_id=session.session_id, turn_id=turn.turn_id,
                  stage="pipeline", error=type(exc).__name__, detail=str(exc)[:500])
        if session.is_current(turn):
            await queue_event(
                session,
                {"type": "error", "turn_id": turn.turn_id, "message": "A local voice component failed.", "code": type(exc).__name__},
                turn.turn_id,
            )
    finally:
        if answer_parts and not reply_logged and not session.is_current(turn):
            log_event("assistant_reply_interrupted", session_id=session.session_id, turn_id=turn.turn_id,
                      text="".join(answer_parts).strip())


async def synthesize_and_queue(session: VoiceSession, turn: Turn, text: str) -> None:
    if not session.is_current(turn):
        return
    audio, sample_rate = await asyncio.to_thread(runtime.synthesize, text)
    if not session.is_current(turn):
        return
    log_event("tts_chunk_ready", session_id=session.session_id, turn_id=turn.turn_id,
              characters=len(text), audio_bytes=len(audio), sample_rate=sample_rate)
    await session.outgoing.put(
        Outbound(
            event={
                "type": "audio_chunk",
                "turn_id": turn.turn_id,
                "sample_rate": sample_rate,
                "encoding": "pcm_s16le_mono",
                "text": text,
            },
            audio=audio,
            turn_id=turn.turn_id,
        )
    )


async def receive_control(session: VoiceSession, payload: str) -> None:
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        await queue_event(session, {"type": "error", "message": "Invalid control message."})
        return
    kind = event.get("type")
    turn_id = event.get("turn_id")
    if kind == "client_error":
        log_event("browser_error", session_id=session.session_id,
                  error=str(event.get("message", "Unknown browser error"))[:500])
    elif kind == "playback_started" and turn_id in session.turns:
        turn = session.turns[turn_id]
        turn.playback_started = True
        if turn.speech_end_at is not None:
            elapsed = max(0.0, (time.monotonic() - turn.speech_end_at) * 1000)
            runtime.metrics.append({"response_ms": elapsed})
            log_event("playback_started", session_id=session.session_id, turn_id=turn_id,
                      speech_end_to_playback_ms=round(elapsed, 1))
    elif kind == "playback_stopped" and turn_id in session.interrupt_pending:
        started = session.interrupt_pending.pop(turn_id)
        elapsed = max(0.0, (time.monotonic() - started) * 1000)
        runtime.metrics.append({"interrupt_ms": elapsed})
        log_event("playback_stopped", session_id=session.session_id, turn_id=turn_id,
                  interruption_to_stop_ms=round(elapsed, 1))
    elif kind == "playback_finished" and session.current_turn and turn_id == session.current_turn.turn_id:
        log_event("playback_finished", session_id=session.session_id, turn_id=turn_id)
        await queue_event(session, {"type": "listening"})
        session.current_turn = None
    elif kind == "clear_history":
        session.history.clear()


async def receive_audio(session: VoiceSession, payload: bytes) -> None:
    if len(payload) != VAD_FRAME_SAMPLES * 2:
        await queue_event(session, {"type": "error", "message": "Unexpected microphone frame size."})
        return
    frame = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    event, completed_audio, score = session.vad.accept(frame)
    if event == "start":
        previous = session.current_turn
        turn = Turn(turn_id=uuid.uuid4().hex)
        session.current_turn = turn
        session.remember_turn(turn)
        log_event("speech_started", session_id=session.session_id, turn_id=turn.turn_id,
                  vad_probability=round(score, 3))
        if previous:
            log_event("barge_in", session_id=session.session_id, old_turn_id=previous.turn_id,
                      new_turn_id=turn.turn_id, playback_was_active=previous.playback_started)
            previous.cancel_event.set()
            if previous.playback_started:
                session.interrupt_pending[previous.turn_id] = turn.speech_onset_at
            clear_outgoing(session)
            await queue_event(
                session,
                {"type": "interrupt", "turn_id": previous.turn_id, "next_turn_id": turn.turn_id},
            )
        await queue_event(
            session,
            {"type": "speech_started", "turn_id": turn.turn_id, "vad_probability": round(score, 3)},
            turn.turn_id,
        )
    elif event == "end" and completed_audio is not None and session.current_turn:
        turn = session.current_turn
        turn.speech_end_at = time.monotonic()
        log_event("speech_ended", session_id=session.session_id, turn_id=turn.turn_id,
                  duration_ms=round(completed_audio.size * 1000 / SAMPLE_RATE, 1))
        await queue_event(session, {"type": "speech_ended", "turn_id": turn.turn_id}, turn.turn_id)
        task = asyncio.create_task(process_turn(session, turn, completed_audio), name=f"turn-{turn.turn_id}")
        session.tasks.add(task)
        task.add_done_callback(session.tasks.discard)
    elif event == "empty" and session.current_turn:
        await queue_event(session, {"type": "empty_turn"})
        session.current_turn = None


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket) -> None:
    global active_session
    await websocket.accept()
    if not runtime.status["ready"]:
        await websocket.send_text(json.dumps({"type": "not_ready", "status": runtime.status_snapshot()}))
        await websocket.close(code=1013, reason="Voice models are still loading")
        return

    async with session_lock:
        if active_session is not None and not active_session.stopped:
            await websocket.send_text(json.dumps({"type": "busy"}))
            await websocket.close(code=1013, reason="A local voice session is already active")
            return
        session = VoiceSession(
            session_id=uuid.uuid4().hex,
            websocket=websocket,
            vad=runtime.new_vad(),
        )
        active_session = session
    log_event("session_started", session_id=session.session_id)

    sender = asyncio.create_task(websocket_sender(session), name=f"ws-sender-{session.session_id}")
    try:
        start_message = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        start = json.loads(start_message)
        if start.get("type") != "start" or start.get("sample_rate") != SAMPLE_RATE:
            await queue_event(session, {"type": "error", "message": "Unsupported audio format."})
            return
        await queue_event(
            session,
            {
                "type": "session_started",
                "session_id": session.session_id,
                "sample_rate": SAMPLE_RATE,
                "vad": "silero-streaming",
                "echo_cancellation": start.get("echo_cancellation"),
                "headphones_required": True,
            },
        )
        while not session.stopped:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await receive_audio(session, message["bytes"])
            elif message.get("text") is not None:
                await receive_control(session, message["text"])
    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        with suppress(Exception):
            await websocket.close(code=1008, reason="Session start timed out")
    except Exception as exc:
        log_event("websocket_error", session_id=session.session_id, error=type(exc).__name__,
                  detail=str(exc)[:500])
    finally:
        session.stopped = True
        for turn in session.turns.values():
            turn.cancel_event.set()
        if session.tasks:
            await asyncio.gather(*tuple(session.tasks), return_exceptions=True)
        sender.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await sender
        async with session_lock:
            if active_session is session:
                active_session = None
        log_event("session_stopped", session_id=session.session_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
