"""Browser-facing HTTP and WebSocket server for the local voice assistant."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import secrets
import struct
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from observability import log_structured, utc_timestamp

from voice_pipeline import (
    MODEL_ROOT,
    SAMPLE_RATE,
    VAD_FRAME_SAMPLES,
    Outbound,
    Turn,
    VoiceRuntime,
)
from speaker_profiles import ProfileStore
from speaker_activity_gate import SpeakerActivityGate
from image_attachment import MAX_IMAGE_DATA_CHARS, image_metadata

logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logger = logging.getLogger("el_voice")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def log_event(event: str, **fields: Any) -> None:
    """Write one structured local diagnostic record; never include audio payloads."""
    if fields.get("turn_id"):
        fields["trace_id"] = fields["turn_id"]
    elif fields.get("new_turn_id"):
        fields["trace_id"] = fields["new_turn_id"]
    log_structured(logger, event, **fields)

ROOT = Path(__file__).resolve().parent
GENERATION_FALLBACK = "Я так качался"
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
    interrupt_enabled: bool = True
    search_enabled: bool = False
    pending_image: str | None = None
    pending_image_id: str | None = None
    ignored_speech: bool = False
    turns: dict[str, Turn] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    tasks: set[asyncio.Task] = field(default_factory=set)
    interrupt_pending: dict[str, float] = field(default_factory=dict)
    stopped: bool = False
    audio_sequence: dict[str, int] = field(default_factory=dict)
    audio_ack_sequence: dict[str, int] = field(default_factory=dict)
    audio_ack_buffered_ms: dict[str, float] = field(default_factory=dict)
    audio_ack_consumed_ms: dict[str, float] = field(default_factory=dict)
    audio_sent_sequence: dict[str, int] = field(default_factory=dict)
    audio_packet_durations: dict[str, dict[int, float]] = field(default_factory=dict)
    browser_buffered_ms: float = 0.0
    audio_capacity: asyncio.Event = field(default_factory=asyncio.Event)
    speaker_filter_enabled: bool = False
    speaker_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    speaker_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    speaker_worker: asyncio.Task | None = None
    speaker_candidate: list[np.ndarray] = field(default_factory=list)
    speaker_candidate_chunks: int = 0
    speaker_active: bool = False
    speaker_activity_gate: SpeakerActivityGate | None = None
    speaker_score_invocations: int = 0
    speaker_gate_skipped_batches: int = 0
    speaker_enrollment_kind: str | None = None
    speaker_enrollment_frames: list[np.ndarray] = field(default_factory=list)
    speaker_enrollment_received_samples: int = 0
    speaker_enrollment_speech_samples: int = 0
    speaker_enrollment_started: float = 0.0
    speaker_enrollment_last_frame_at: float | None = None
    speaker_enrollment_last_vad_probability: float | None = None
    speaker_enrollment_last_frame_is_speech: bool = False
    speaker_enrollment_progress_task: asyncio.Task | None = None
    speaker_control_times: deque = field(default_factory=deque)
    speaker_metrics: deque = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self) -> None:
        self.audio_capacity.set()

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
async def index(request: Request) -> HTMLResponse:
    token = secrets.token_urlsafe(32)
    document = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    document = document.replace("__EL_VOICE_CSRF_TOKEN__", token)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response = HTMLResponse(document, headers={"Cache-Control": "no-store"})
    response.set_cookie("el_voice_session", token, httponly=True, secure=secure,
                        samesite="strict", path="/", max_age=3600)
    return response


@app.get("/healthz")
async def healthz() -> JSONResponse:
    # Process health lets Compose start the proxy; readiness is reported separately.
    return JSONResponse({"status": "ok", "ready": runtime.status["ready"], "stage": runtime.status["stage"]})


@app.get("/api/status")
async def status() -> JSONResponse:
    # Session-owned queue metrics are supplied explicitly; VoiceRuntime has no app globals.
    return JSONResponse({
        **runtime.status_snapshot(session=active_session),
        "revision": os.environ.get("EL_VOICE_REVISION", "unknown"),
    })


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
    event = dict(event)
    event.setdefault("timestamp", utc_timestamp())
    if turn_id:
        event["turn_id"] = turn_id
        event["trace_id"] = turn_id
    await session.outgoing.put(Outbound(event=event, turn_id=turn_id))


async def clear_llm_context(session_id: str) -> None:
    clear = getattr(runtime, "clear_llm_session", None)
    if clear is not None:
        await clear(session_id)


def clear_outgoing(session: VoiceSession) -> None:
    while True:
        try:
            session.outgoing.get_nowait()
            session.outgoing.task_done()
        except asyncio.QueueEmpty:
            return


def clear_turn_outgoing(session: VoiceSession, turn_id: str) -> None:
    retained: list[Outbound] = []
    while True:
        try:
            message = session.outgoing.get_nowait()
        except asyncio.QueueEmpty:
            break
        session.outgoing.task_done()
        if message.turn_id != turn_id:
            retained.append(message)
    for message in retained:
        session.outgoing.put_nowait(message)

    session.browser_buffered_ms = 0.0
    audio_capacity = getattr(session, "audio_capacity", None)
    if audio_capacity is not None:
        audio_capacity.set()
    for name in ("audio_sequence", "audio_ack_sequence", "audio_ack_buffered_ms",
                 "audio_ack_consumed_ms", "audio_sent_sequence", "audio_packet_durations"):
        state = getattr(session, name, None)
        if state is not None:
            state.pop(turn_id, None)


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
                event_type = message.event.get("type")
                if event_type in {"text_delta", "response_replace", "response_audio_unavailable",
                                  "tool_call", "sources", "turn_complete", "error", "turn_failed"}:
                    text_value = message.event.get("text")
                    log_event("voice_websocket_event_sent", session_id=session.session_id,
                              turn_id=message.turn_id, event_type=event_type,
                              text_chars=len(text_value) if isinstance(text_value, str) else 0)
            if message.audio:
                if message.turn_id:
                    turn = session.turns.get(message.turn_id)
                    if turn is None or not session.is_current(turn):
                        continue
                await websocket.send_bytes(message.audio)
                if len(message.audio) >= 28 and message.audio[:4] == b"ELVA":
                    sent_turn_id = message.audio[4:20].hex()
                    sent_sequence = struct.unpack_from(">I", message.audio, 20)[0]
                    session.audio_sent_sequence[sent_turn_id] = sent_sequence
                    sample_rate = struct.unpack_from(">I", message.audio, 24)[0]
                    log_event("voice_websocket_audio_sent", session_id=session.session_id,
                              turn_id=sent_turn_id, sequence=sent_sequence,
                              sample_rate=sample_rate, audio_bytes=len(message.audio) - 28)
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


def _sanitize_speech(text: str) -> str:
    """Remove formatting and non-spoken tokens while preserving readable text."""
    import unicodedata

    text = re.sub(r"\[([^\]]+)\]\((?:https?://)?[^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"```.*?```|`([^`]*)`", lambda match: match.group(1) or " ", text, flags=re.DOTALL)
    text = re.sub(r"[*_~>#|]", " ", text)
    text = "".join(character if unicodedata.category(character) not in ("So", "Sk") else " "
                    for character in text)
    return re.sub(r"\s+", " ", text)


def _speakable_chunks(buffer: str, max_chars: int = 80, min_chars: int = 18) -> tuple[list[str], str]:
    chunks: list[str] = []
    raw = buffer
    incomplete_url = re.search(r"(?:https?://|www\.)\S*$", raw, flags=re.IGNORECASE)
    protected_tail = raw[incomplete_url.start():] if incomplete_url else ""
    remaining = _sanitize_speech(raw[:incomplete_url.start()] if incomplete_url else raw)
    while remaining:
        boundary = None
        for index, character in enumerate(remaining):
            if character in ".!?…。！？" and min_chars <= index < max_chars:
                boundary = index + 1
                while boundary < len(remaining) and remaining[boundary].isspace():
                    boundary += 1
                break
        if boundary is None and len(remaining) > max_chars:
            boundary = remaining.rfind(" ", 0, max_chars + 1)
            if boundary <= 0:
                boundary = max_chars
        if boundary is None:
            break
        piece = remaining[:boundary].strip()
        piece = _sanitize_speech(piece)
        if piece:
            chunks.append(piece)
        remaining = remaining[boundary:].lstrip()
    return chunks, (remaining + " " + protected_tail).strip() if protected_tail else remaining


async def _llm_items(session: VoiceSession, turn: Turn, user_text: str):
    """Consume a bounded text/tool/source event stream from the LLM service."""
    if getattr(runtime, "llm_client", None) is not None:
        options = {"image": turn.image} if turn.image is not None else {}
        stream = runtime.llm_stream(
            user_text, turn.cancel_event, session.history,
            session_id=session.session_id, turn_id=turn.turn_id,
            search_enabled=turn.search_enabled,
            **options,
        )
    else:
        stream = runtime.llm_stream(user_text, turn.cancel_event, session.history)
    if hasattr(stream, "__aiter__"):
        async for event in stream:
            if not session.is_current(turn):
                return
            yield {"type": "delta", "text": event} if isinstance(event, str) else event
        return

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
            for delta in stream:
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
            yield {"type": "delta", "text": value}
    finally:
        if turn.cancel_event.is_set() or session.stopped:
            while not worker_task.done():
                with suppress(asyncio.QueueEmpty):
                    items.get_nowait()
                await asyncio.sleep(0.005)
        await asyncio.shield(worker_task)


async def process_turn(session: VoiceSession, turn: Turn, audio: np.ndarray | None,
                       text_input: str | None = None) -> None:
    answer_parts: list[str] = []
    transcript: str | None = None
    reply_logged = False
    llm_delta_count = 0
    llm_response_chars = 0
    tts_chunk_count = 0
    try:
        stage = "stt"
        text = (text_input if text_input is not None else
                await asyncio.to_thread(runtime.transcribe, audio, turn.cancel_event))
        transcript = text
        if not session.is_current(turn):
            log_event("turn_discarded_after_transcription", session_id=session.session_id, turn_id=turn.turn_id)
            return
        if not text:
            log_event("empty_transcription", session_id=session.session_id, turn_id=turn.turn_id)
            await queue_event(session, {"type": "empty_turn", "turn_id": turn.turn_id}, turn.turn_id)
            return

        # Consume only after a nonempty question, so failed/empty STT keeps the image.
        # This atomic snapshot also prevents later pastes from changing a running turn.
        attachment_id = getattr(session, "pending_image_id", None)
        turn.image, session.pending_image = getattr(session, "pending_image", None), None
        session.pending_image_id = None
        metadata = image_metadata(turn.image) if turn.image else None
        log_event("user_transcript", session_id=session.session_id, turn_id=turn.turn_id,
                  text=text, input_source="text" if text_input is not None else "voice",
                  attachment_id=attachment_id, image=metadata)

        await queue_event(
            session, {"type": "transcript", "turn_id": turn.turn_id, "text": text,
                      "attachment_id": attachment_id}, turn.turn_id
        )
        await queue_event(session, {"type": "generation_started", "turn_id": turn.turn_id}, turn.turn_id)
        text_buffer = ""
        stage = "llm"
        log_event("voice_llm_stream_started", session_id=session.session_id,
                  turn_id=turn.turn_id,
                  client_managed=getattr(runtime, "llm_client", None) is not None)
        first_speech_chunk = True
        runtime_config = getattr(runtime, "config", {})
        speech_config = runtime_config.get("speech", {})
        max_chars = int(speech_config.get("max_chunk_chars", 80))
        minimum_chars = int(speech_config.get("minimum_chunk_chars", 18))
        first_max_chars = int(speech_config.get("first_chunk_max_chars", 60))
        async for delta in _llm_items(session, turn, text):
            if not session.is_current(turn):
                return
            if delta.get("type") == "sources":
                log_event("voice_llm_event_received", session_id=session.session_id,
                          turn_id=turn.turn_id, event_type="sources",
                          tool_call_id=delta.get("tool_call_id"),
                          tool_status=delta.get("status"),
                          error_code=delta.get("error_code"),
                          source_count=len(delta.get("sources", [])))
                await queue_event(session, {
                    "type": "sources", "turn_id": turn.turn_id,
                    "sources": delta.get("sources", [])[:5],
                    "tool_call_id": delta.get("tool_call_id", ""),
                    "status": delta.get("status", "success"),
                    "error_code": delta.get("error_code"),
                }, turn.turn_id)
                continue
            if delta.get("type") == "tool_call":
                log_event("voice_llm_event_received", session_id=session.session_id,
                          turn_id=turn.turn_id, event_type="tool_call",
                          tool_call_id=delta.get("tool_call_id"),
                          result_count=min(5, max(0, int(delta.get("result_count", 0)))))
                await queue_event(session, {
                    "type": "tool_call", "turn_id": turn.turn_id,
                    "name": delta.get("name", "web_search"),
                    "tool_call_id": delta.get("tool_call_id", ""),
                    "result_count": min(5, max(0, int(delta.get("result_count", 0)))),
                }, turn.turn_id)
                continue
            if delta.get("type") == "replace":
                fallback = str(delta.get("text") or GENERATION_FALLBACK)[:500]
                clear_turn_outgoing(session, turn.turn_id)
                answer_parts.clear()
                text_buffer = ""
                llm_response_chars = len(fallback)
                log_event("voice_turn_fallback_received", session_id=session.session_id,
                          turn_id=turn.turn_id,
                          fallback_reason=str(delta.get("fallback_reason", "generation_failed"))[:100],
                          response_chars=len(fallback))
                await queue_event(session, {
                    "type": "response_replace", "turn_id": turn.turn_id, "text": fallback,
                }, turn.turn_id)
                answer_parts.append(fallback)
                try:
                    stage = "tts"
                    tts_chunk_count += 1
                    await synthesize_and_queue(session, turn, fallback)
                except Exception as exc:
                    log_event("fallback_tts_failed", session_id=session.session_id,
                              turn_id=turn.turn_id, error=type(exc).__name__,
                              detail=str(exc)[:300])
                    await queue_event(session, {
                        "type": "response_audio_unavailable", "turn_id": turn.turn_id,
                    }, turn.turn_id)
                stage = "llm"
                continue
            if delta.get("type") != "delta":
                continue
            delta_text = str(delta.get("text", ""))
            if not delta_text:
                continue
            llm_delta_count += 1
            llm_response_chars += len(delta_text)
            log_event("voice_llm_delta_received", session_id=session.session_id,
                      turn_id=turn.turn_id, delta_index=llm_delta_count,
                      text_chars=len(delta_text))
            answer_parts.append(delta_text)
            text_buffer += delta_text
            await queue_event(
                session, {"type": "text_delta", "turn_id": turn.turn_id, "text": delta_text}, turn.turn_id
            )
            chunks, text_buffer = _speakable_chunks(
                text_buffer,
                max_chars=first_max_chars if first_speech_chunk else max_chars,
                min_chars=minimum_chars,
            )
            for chunk in chunks:
                if not session.is_current(turn):
                    return
                stage = "tts"
                tts_chunk_count += 1
                await synthesize_and_queue(session, turn, chunk)
                first_speech_chunk = False
                stage = "llm"

        if not session.is_current(turn):
            return
        final_chunk = _sanitize_speech(text_buffer)
        if final_chunk:
            stage = "tts"
            tts_chunk_count += 1
            await synthesize_and_queue(session, turn, final_chunk)
            stage = "llm"
        log_event("voice_turn_completed", session_id=session.session_id,
                  turn_id=turn.turn_id, llm_delta_count=llm_delta_count,
                  llm_response_chars=llm_response_chars,
                  tts_chunk_count=tts_chunk_count)
        answer = "".join(answer_parts).strip()
        if answer and session.is_current(turn):
            # The production LLM service owns context; legacy test runtimes use this fallback.
            if getattr(runtime, "llm_client", None) is None:
                session.history.extend([
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer},
                ])
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
                  stage=stage, error=type(exc).__name__, detail=str(exc)[:500] or repr(exc)[:500],
                  llm_delta_count=llm_delta_count,
                  llm_response_chars=llm_response_chars,
                  tts_chunk_count=tts_chunk_count)
        if session.is_current(turn):
            clear_turn_outgoing(session, turn.turn_id)
            answer_parts[:] = [GENERATION_FALLBACK]
            llm_response_chars = len(GENERATION_FALLBACK)
            log_event("voice_turn_fallback_emitted", session_id=session.session_id,
                      turn_id=turn.turn_id,
                      fallback_reason=stage,
                      response_chars=len(GENERATION_FALLBACK))
            await queue_event(session, {
                "type": "response_replace", "turn_id": turn.turn_id,
                "text": GENERATION_FALLBACK,
            }, turn.turn_id)
            try:
                tts_chunk_count += 1
                await synthesize_and_queue(session, turn, GENERATION_FALLBACK)
            except Exception as fallback_exc:
                log_event("fallback_tts_failed", session_id=session.session_id,
                          turn_id=turn.turn_id, error=type(fallback_exc).__name__,
                          detail=str(fallback_exc)[:300])
                await queue_event(session, {
                    "type": "response_audio_unavailable", "turn_id": turn.turn_id,
                }, turn.turn_id)
            answer = GENERATION_FALLBACK
            if getattr(runtime, "llm_client", None) is None:
                session.history.extend([
                    {"role": "user", "content": transcript or ""},
                    {"role": "assistant", "content": answer},
                ])
                session.history = session.history[-8:]
            log_event("assistant_reply", session_id=session.session_id,
                      turn_id=turn.turn_id, text=answer, is_fallback=True)
            reply_logged = True
            await queue_event(session, {"type": "turn_complete", "turn_id": turn.turn_id}, turn.turn_id)
    finally:
        turn.image = None
        if answer_parts and not reply_logged and not session.is_current(turn):
            log_event("assistant_reply_interrupted", session_id=session.session_id, turn_id=turn.turn_id,
                      text="".join(answer_parts).strip())


async def synthesize_and_queue(session: VoiceSession, turn: Turn, text: str) -> None:
    if not session.is_current(turn):
        return
    log_event("tts_synthesis_started", session_id=session.session_id,
              turn_id=turn.turn_id, characters=len(text))
    audio, sample_rate = await asyncio.to_thread(runtime.synthesize, text, turn.cancel_event)
    if not session.is_current(turn):
        return
    runtime_config = getattr(runtime, "config", {})
    audio_config = runtime_config.get("audio", {})
    max_buffered_ms = float(audio_config.get("max_buffered_audio_ms", 2000))
    packet_duration_ms = float(audio_config.get("packet_duration_ms", 800))
    if not hasattr(session, "browser_buffered_ms"):
        session.browser_buffered_ms = 0.0
    if not hasattr(session, "audio_capacity"):
        session.audio_capacity = asyncio.Event()
        session.audio_capacity.set()
    if not hasattr(session, "audio_sequence"):
        session.audio_sequence = {}
    if not hasattr(session, "audio_ack_sequence"):
        session.audio_ack_sequence = {}
    if not hasattr(session, "audio_sent_sequence"):
        session.audio_sent_sequence = {}
    if not hasattr(session, "audio_packet_durations"):
        session.audio_packet_durations = {}
    packet_bytes = max(2, int(sample_rate * packet_duration_ms / 1000) * 2)
    packet_bytes -= packet_bytes % 2
    log_event("tts_chunk_ready", session_id=session.session_id, turn_id=turn.turn_id,
              characters=len(text), audio_bytes=len(audio), sample_rate=sample_rate)
    for offset in range(0, len(audio), packet_bytes):
        if not session.is_current(turn):
            return
        payload = audio[offset:offset + packet_bytes]
        duration_ms = len(payload) * 500 / sample_rate
        while session.is_current(turn) and session.browser_buffered_ms + duration_ms > max_buffered_ms:
            session.audio_capacity.clear()
            try:
                await asyncio.wait_for(session.audio_capacity.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass
        if not session.is_current(turn):
            return
        sequence = session.audio_sequence.get(turn.turn_id, 0)
        session.audio_sequence[turn.turn_id] = sequence + 1
        try:
            packet = struct.pack(">4s16sII", b"ELVA", bytes.fromhex(turn.turn_id), sequence, sample_rate) + payload
        except ValueError as exc:
            raise ValueError("Turn identity must be a 128-bit hexadecimal UUID") from exc
        session.audio_packet_durations.setdefault(turn.turn_id, {})[sequence] = duration_ms
        session.browser_buffered_ms += duration_ms
        await session.outgoing.put(
            Outbound(
                event={
                    "type": "audio_chunk",
                    "turn_id": turn.turn_id,
                    "sequence": sequence,
                    "sample_rate": sample_rate,
                    "encoding": "pcm_s16le_mono",
                    "text": text if offset == 0 else "",
                },
                audio=packet,
                turn_id=turn.turn_id,
            )
        )


async def receive_control(session: VoiceSession, payload: str) -> None:
    if len(payload) > MAX_IMAGE_DATA_CHARS + 1024:
        await queue_event(session, {"type": "error", "message": "Control message is too large."})
        return
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        await queue_event(session, {"type": "error", "message": "Invalid control message."})
        return
    if not isinstance(event, dict):
        await queue_event(session, {"type": "error", "message": "Control message must be an object."})
        return
    kind = event.get("type")
    turn_id = event.get("turn_id")
    if kind != "image_attachment" and len(payload.encode("utf-8", errors="ignore")) > 32768:
        await queue_event(session, {"type": "error", "message": "Control message is too large."})
        return
    if kind not in {"image_attachment", "text_query"} and len(payload.encode("utf-8", errors="ignore")) > 4096:
        await queue_event(session, {"type": "error", "message": "Control message is too large."})
        return
    if not isinstance(kind, str):
        await queue_event(session, {"type": "error", "message": "Control type must be a string."})
        return
    if kind == "image_attachment":
        if getattr(runtime, "mode", "local") != "remote":
            await queue_event(session, {"type": "image_error", "message": "Images require Remote Voice."})
            return
        attachment_id = event.get("attachment_id")
        if not isinstance(attachment_id, str) or not re.fullmatch(r"[0-9a-f]{32}", attachment_id):
            await queue_event(session, {"type": "image_error", "message": "Invalid image identity."})
            return
        try:
            metadata = image_metadata(event.get("image"))
        except ValueError as exc:
            await queue_event(session, {"type": "image_error", "attachment_id": attachment_id,
                                        "message": str(exc)})
            return
        session.pending_image = event["image"]
        session.pending_image_id = attachment_id
        log_event("clipboard_image_received", session_id=session.session_id,
                  attachment_id=attachment_id, **metadata)
        await queue_event(session, {"type": "image_attachment", "attachment_id": attachment_id,
                                    "state": "ready", **metadata})
        return
    if kind == "image_remove":
        if event.get("attachment_id") == session.pending_image_id:
            session.pending_image = None
            session.pending_image_id = None
        return
    if kind == "text_query":
        text = event.get("text")
        if not isinstance(text, str) or len(text) > 8000:
            await queue_event(session, {"type": "error", "message": "Enter a question of at most 8000 characters."})
            return
        text = text.strip()
        if not text and session.pending_image:
            text = "Опиши, что на этой картинке."
        if not text:
            await queue_event(session, {"type": "error", "message": "Enter a question or paste an image."})
            return
        if session.speaker_enrollment_kind:
            await queue_event(session, {"type": "error", "message": "Finish speaker enrollment before sending a question."})
            return
        # A typed Send explicitly replaces the current reply, regardless of the
        # microphone-only interrupt preference. Discard any half-captured utterance.
        session.vad = runtime.new_vad()
        session.ignored_speech = False
        turn = await activate_turn(session)
        turn.speech_end_at = time.monotonic()
        task = asyncio.create_task(process_turn(session, turn, None, text_input=text),
                                   name=f"text-turn-{turn.turn_id}")
        session.tasks.add(task)
        task.add_done_callback(session.tasks.discard)
        return
    if kind and kind.startswith("speaker_"):
        now = time.monotonic()
        while session.speaker_control_times and now - session.speaker_control_times[0] > 60:
            session.speaker_control_times.popleft()
        limit = 60 if kind == "speaker_enrollment_end" else 20
        if len(session.speaker_control_times) >= limit:
            await queue_event(session, {"type": "speaker_error", "message": "Speaker controls are rate limited."})
            return
        session.speaker_control_times.append(now)
    if kind == "speaker_enrollment_begin":
        profile_kind = event.get("kind")
        if not isinstance(profile_kind, str) or profile_kind not in {"owner", "guest"}:
            await queue_event(session, {"type": "speaker_error", "message": "Choose owner or guest enrollment."})
            return
        if profile_kind == "guest" and event.get("consent") is not True:
            await queue_event(session, {"type": "speaker_error", "message": "Guest enrollment requires their explicit consent."})
            return
        if getattr(runtime, "speaker_model", None) is None:
            await queue_event(session, {"type": "speaker_error", "message": "Local speaker model is unavailable."})
            return
        if session.speaker_enrollment_kind:
            await queue_event(session, {"type": "speaker_error", "message": "Speaker enrollment is already active."})
            return
        previous = session.current_turn
        if previous:
            previous.cancel_event.set()
            clear_turn_outgoing(session, previous.turn_id)
            if previous.playback_started:
                session.interrupt_pending[previous.turn_id] = time.monotonic()
            await queue_event(session, {"type": "interrupt", "turn_id": previous.turn_id})
            session.current_turn = None
            log_event("speaker_enrollment_interrupted_turn", session_id=session.session_id,
                      turn_id=previous.turn_id, playback_was_active=previous.playback_started)
        while True:
            try:
                session.speaker_queue.get_nowait()
                session.speaker_queue.task_done()
            except asyncio.QueueEmpty:
                break
        session.speaker_candidate.clear()
        session.speaker_candidate_chunks = 0
        session.speaker_active = False
        session.speaker_enrollment_kind = profile_kind
        session.speaker_enrollment_frames.clear()
        session.speaker_enrollment_received_samples = 0
        session.speaker_enrollment_speech_samples = 0
        session.speaker_enrollment_started = time.monotonic()
        session.speaker_enrollment_last_frame_at = None
        session.speaker_enrollment_last_vad_probability = None
        session.speaker_enrollment_last_frame_is_speech = False
        session.enrollment_vad = runtime.new_vad()
        session.vad = runtime.new_vad()
        log_event("speaker_enrollment_button", session_id=session.session_id,
                  state="pressed", held=True, kind=profile_kind,
                  speaker_filter_enabled=session.speaker_filter_enabled)
        progress_task = asyncio.create_task(
            speaker_enrollment_progress(session), name=f"speaker-enrollment-progress-{session.session_id}"
        )
        session.speaker_enrollment_progress_task = progress_task
        session.tasks.add(progress_task)
        progress_task.add_done_callback(session.tasks.discard)
        speaker_config = runtime.config.get("speaker", {})
        await queue_event(session, {"type": "speaker_enrollment", "state": "capturing", "kind": profile_kind,
                                    "minimum_recording_seconds": int(speaker_config.get("enrollment_min_recording_seconds", 10)),
                                    "minimum_detected_speech_seconds": int(speaker_config.get("enrollment_min_detected_speech_seconds", 2))})
    elif kind in {"speaker_enrollment_end", "speaker_enrollment_cancel"}:
        if not session.speaker_enrollment_kind:
            await queue_event(session, {"type": "speaker_error", "message": "No speaker enrollment is active."})
            return
        if kind == "speaker_enrollment_cancel":
            elapsed = max(0.0, time.monotonic() - session.speaker_enrollment_started)
            log_event("speaker_enrollment_button", session_id=session.session_id,
                      state="cancelled", held=False, elapsed_seconds=round(elapsed, 1),
                      recorded_seconds=round(session.speaker_enrollment_received_samples / SAMPLE_RATE, 1),
                      speech_seconds=round(session.speaker_enrollment_speech_samples / SAMPLE_RATE, 1))
            await stop_speaker_enrollment_progress(session)
            session.speaker_enrollment_kind = None
            session.speaker_enrollment_frames.clear()
            session.speaker_enrollment_received_samples = 0
            session.speaker_enrollment_speech_samples = 0
            await queue_event(session, {"type": "speaker_enrollment", "state": "cancelled"})
            return
        profile_kind = session.speaker_enrollment_kind
        elapsed = max(0.0, time.monotonic() - session.speaker_enrollment_started)
        recorded_samples = session.speaker_enrollment_received_samples
        speech_samples = session.speaker_enrollment_speech_samples
        await stop_speaker_enrollment_progress(session)
        speaker_config = runtime.config.get("speaker", {})
        minimum = int(speaker_config.get("enrollment_min_recording_seconds", 10) * SAMPLE_RATE)
        minimum_speech = int(speaker_config.get("enrollment_min_detected_speech_seconds", 2) * SAMPLE_RATE)
        audio = np.concatenate(session.speaker_enrollment_frames) if session.speaker_enrollment_frames else np.zeros(0, dtype=np.float32)
        log_event("speaker_enrollment_button", session_id=session.session_id,
                  state="released", held=False, elapsed_seconds=round(elapsed, 1),
                  recorded_seconds=round(recorded_samples / SAMPLE_RATE, 1),
                  speech_seconds=round(speech_samples / SAMPLE_RATE, 1))
        session.speaker_enrollment_kind = None
        session.speaker_enrollment_frames.clear()
        session.speaker_enrollment_received_samples = 0
        session.speaker_enrollment_speech_samples = 0
        if audio.size < minimum:
            await queue_event(session, {"type": "speaker_error", "message": "Enrollment needs at least 10 seconds of recorded audio; the previous profile was kept."})
            return
        if speech_samples < minimum_speech:
            await queue_event(session, {"type": "speaker_error", "message": f"Enrollment needs at least {minimum_speech / SAMPLE_RATE:g} seconds of detected speech; the previous profile was kept."})
            return
        try:
            await queue_event(session, {"type": "speaker_enrollment", "state": "processing"})
            embedding = await asyncio.to_thread(runtime.speaker_model.enroll, audio)
            profile_id = "owner" if profile_kind == "owner" else None
            if profile_id is None:
                profile_id = f"guest-{secrets.token_hex(6)}"
            await asyncio.to_thread(runtime.speaker_model.warmup, profile_id, embedding)
            store = profile_store()
            profile_id = await asyncio.to_thread(store.upsert, profile_kind, embedding, profile_id)
            session.speaker_profiles = await asyncio.to_thread(store.load)
            session.speaker_filter_enabled = True
            if session.speaker_worker is None or session.speaker_worker.done():
                session.speaker_worker = asyncio.create_task(
                    speaker_audio_worker(session), name=f"speaker-filter-{session.session_id}"
                )
            await queue_event(session, {"type": "speaker_enrollment", "state": "saved", "profile": profile_id,
                                        "profiles": public_profiles(session.speaker_profiles),
                                        "filter_enabled": True})
        except Exception as exc:
            log_event("speaker_enrollment_failed", session_id=session.session_id, error=type(exc).__name__)
            await queue_event(session, {"type": "speaker_error", "message": "Could not save the speaker profile; the previous profile was kept."})
        return
    elif kind == "speaker_filter_setting":
        enabled = event.get("enabled")
        if type(enabled) is not bool:
            await queue_event(session, {"type": "speaker_error", "message": "Filtering setting must be a boolean."})
            return
        await queue_event(session, {"type": "speaker_error", "message": "End the session to change filtering; the setting is applied at the next session start.",
                                    "enabled": session.speaker_filter_enabled})
    elif kind == "speaker_profile_enabled":
        profile_id, enabled = event.get("profile_id"), event.get("enabled")
        if not isinstance(profile_id, str) or profile_id not in session.speaker_profiles or type(enabled) is not bool:
            await queue_event(session, {"type": "speaker_error", "message": "Invalid speaker profile setting."})
            return
        profiles = {key: dict(profile) for key, profile in session.speaker_profiles.items()}
        profiles[profile_id]["enabled"] = enabled
        try:
            await asyncio.to_thread(profile_store().save, profiles)
        except Exception:
            await queue_event(session, {"type": "speaker_error",
                                        "message": "Could not save the speaker setting; the existing profile settings were kept."})
            return
        session.speaker_profiles = profiles
        if not enabled:
            runtime.speaker_model.clear(profile_id)
        await queue_event(session, {"type": "speaker_profiles", "profiles": public_profiles(profiles),
                                    "filter_enabled": session.speaker_filter_enabled})
    elif kind == "speaker_profile_delete":
        profile_id = event.get("profile_id")
        if not isinstance(profile_id, str) or profile_id not in session.speaker_profiles:
            await queue_event(session, {"type": "speaker_error", "message": "Speaker profile was not found."})
            return
        profiles = {key: dict(profile) for key, profile in session.speaker_profiles.items()}
        profiles.pop(profile_id)
        try:
            if profiles:
                await asyncio.to_thread(profile_store().save, profiles)
            else:
                store = profile_store()
                store.path.unlink(missing_ok=True)
        except Exception:
            await queue_event(session, {"type": "speaker_error",
                                        "message": "Could not delete the speaker profile; existing profiles were kept."})
            return
        session.speaker_profiles = profiles
        runtime.speaker_model.clear(profile_id)
        await queue_event(session, {"type": "speaker_profiles", "profiles": public_profiles(profiles),
                                    "filter_enabled": session.speaker_filter_enabled})
    elif kind == "speaker_profiles_get":
        await queue_event(session, {"type": "speaker_profiles", "profiles": public_profiles(session.speaker_profiles),
                                    "filter_enabled": session.speaker_filter_enabled})
    if kind == "interrupt_setting":
        enabled = event.get("enabled")
        if type(enabled) is not bool:
            await queue_event(session, {
                "type": "error", "message": "Interrupt setting must be a boolean.",
            })
            return
        session.interrupt_enabled = enabled
        await queue_event(session, {"type": "interrupt_setting", "enabled": enabled})
    elif kind == "search_setting":
        enabled = event.get("enabled")
        if type(enabled) is not bool:
            await queue_event(session, {
                "type": "error", "message": "Search setting must be a boolean.",
            })
            return
        remote_mode = getattr(runtime, "mode", "local") == "remote"
        session.search_enabled = enabled if remote_mode else False
        await queue_event(session, {"type": "search_setting", "enabled": session.search_enabled})
    elif kind == "client_error":
        log_event("browser_error", session_id=session.session_id,
                  error=str(event.get("message", "Unknown browser error"))[:500])
    elif kind == "playback_queue":
        try:
            turn = session.turns.get(turn_id)
            sequence = event.get("sequence")
            buffered_ms = event.get("buffered_ms")
            consumed_ms = event.get("consumed_ms")
            max_buffered_ms = float(getattr(runtime, "config", {}).get("audio", {}).get("max_buffered_audio_ms", 2000))
            if (turn is None or turn is not session.current_turn or isinstance(sequence, bool)
                    or not isinstance(sequence, int) or sequence < 0
                    or isinstance(buffered_ms, bool) or not isinstance(buffered_ms, (int, float))
                    or not math.isfinite(buffered_ms) or not 0 <= buffered_ms <= max_buffered_ms
                    or isinstance(consumed_ms, bool) or not isinstance(consumed_ms, (int, float))
                    or not math.isfinite(consumed_ms) or not 0 <= consumed_ms <= max_buffered_ms):
                raise ValueError("invalid playback capacity acknowledgement")
            expected = session.audio_sent_sequence.get(turn_id, -1)
            previous_sequence = session.audio_ack_sequence.get(turn_id, -1)
            previous_buffered = session.audio_ack_buffered_ms.get(turn_id)
            previous_consumed = session.audio_ack_consumed_ms.get(turn_id, 0.0)
            if sequence > expected or sequence < previous_sequence:
                raise ValueError("out-of-order playback acknowledgement")
            if sequence == previous_sequence and (
                previous_buffered is None or float(buffered_ms) >= previous_buffered
                or float(consumed_ms) <= 0
            ):
                raise ValueError("duplicate playback acknowledgement made no progress")
            unacknowledged_ms = sum(
                duration
                for packet_sequence, duration in session.audio_packet_durations.get(turn_id, {}).items()
                if packet_sequence > sequence
            )
            effective_buffered = float(buffered_ms) + unacknowledged_ms
            if effective_buffered > max_buffered_ms:
                raise ValueError("playback acknowledgement exceeds reserved audio capacity")
            session.audio_ack_sequence[turn_id] = sequence
            session.audio_ack_buffered_ms[turn_id] = float(buffered_ms)
            session.audio_ack_consumed_ms[turn_id] = previous_consumed + float(consumed_ms)
            for packet_sequence in tuple(session.audio_packet_durations.get(turn_id, {})):
                if packet_sequence <= sequence:
                    session.audio_packet_durations[turn_id].pop(packet_sequence, None)
            session.browser_buffered_ms = effective_buffered
            session.audio_capacity.set()
        except (TypeError, ValueError) as exc:
            log_event("invalid_playback_ack", session_id=session.session_id,
                      turn_id=turn_id, error=str(exc))
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
        session.browser_buffered_ms = 0
        session.audio_capacity.set()
        for state in (session.audio_sequence, session.audio_ack_sequence,
                      session.audio_ack_buffered_ms, session.audio_ack_consumed_ms,
                      session.audio_sent_sequence, session.audio_packet_durations):
            state.pop(turn_id, None)
    elif kind == "clear_history":
        session.pending_image = None
        session.pending_image_id = None
        session.history.clear()
        await clear_llm_context(session.session_id)


def allowed_origins() -> set[str]:
    configured = os.environ.get("EL_VOICE_ALLOWED_ORIGINS", "")
    return {item.strip().rstrip("/") for item in configured.split(",") if item.strip()} | {
        "http://localhost:8765", "http://127.0.0.1:8765",
    }


def profile_store() -> ProfileStore:
    store = getattr(runtime, "profile_store", None)
    if store is None:
        store = ProfileStore()
        runtime.profile_store = store
    return store


def public_profiles(profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": key, "kind": value["kind"], "enabled": value["enabled"]}
            for key, value in profiles.items()]


async def speaker_enrollment_progress(session: VoiceSession) -> None:
    """Report hold state and capture counters independently of microphone frame arrival."""
    max_seconds = float(runtime.config.get("speaker", {}).get("enrollment_max_seconds", 45))
    while session.speaker_enrollment_kind and not session.stopped:
        now = time.monotonic()
        elapsed = max(0.0, now - session.speaker_enrollment_started)
        recorded_seconds = round(session.speaker_enrollment_received_samples / SAMPLE_RATE, 1)
        speech_seconds = round(session.speaker_enrollment_speech_samples / SAMPLE_RATE, 1)
        frame_gap_ms = (None if session.speaker_enrollment_last_frame_at is None else
                        round(max(0.0, now - session.speaker_enrollment_last_frame_at) * 1000, 1))
        log_event("speaker_enrollment_progress", session_id=session.session_id,
                  button_held=True, elapsed_seconds=round(elapsed, 1),
                  recorded_seconds=recorded_seconds, speech_seconds=speech_seconds,
                  received_frames=session.speaker_enrollment_received_samples // VAD_FRAME_SAMPLES,
                  audio_frame_gap_ms=frame_gap_ms,
                  vad_probability=session.speaker_enrollment_last_vad_probability,
                  vad_speech_frame=session.speaker_enrollment_last_frame_is_speech)
        await queue_event(session, {"type": "speaker_enrollment", "state": "capturing",
                                    "recorded_seconds": recorded_seconds,
                                    "speech_seconds": speech_seconds,
                                    "elapsed_seconds": round(elapsed, 1),
                                    "audio_frame_gap_ms": frame_gap_ms})
        if elapsed >= max_seconds:
            session.speaker_enrollment_kind = None
            session.speaker_enrollment_frames.clear()
            session.speaker_enrollment_received_samples = 0
            session.speaker_enrollment_speech_samples = 0
            session.speaker_enrollment_progress_task = None
            log_event("speaker_enrollment_button", session_id=session.session_id,
                      state="timed_out", held=False, elapsed_seconds=round(elapsed, 1),
                      audio_frame_gap_ms=frame_gap_ms)
            await queue_event(session, {"type": "speaker_error", "message": "Enrollment timed out; the previous profile was kept."})
            return
        await asyncio.sleep(0.5)


async def stop_speaker_enrollment_progress(session: VoiceSession) -> None:
    task = session.speaker_enrollment_progress_task
    session.speaker_enrollment_progress_task = None
    if task and task is not asyncio.current_task() and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def activate_turn(session: VoiceSession) -> Turn:
    """Replace the active turn and cancel its queued speech before a new question."""
    previous = session.current_turn
    turn = Turn(turn_id=uuid.uuid4().hex, search_enabled=session.search_enabled)
    session.current_turn = turn
    session.remember_turn(turn)
    if previous:
        log_event("barge_in", session_id=session.session_id, old_turn_id=previous.turn_id,
                  new_turn_id=turn.turn_id, playback_was_active=previous.playback_started)
        previous.cancel_event.set()
        session.browser_buffered_ms = 0.0
        session.audio_capacity.set()
        for state_name in ("audio_sequence", "audio_ack_sequence", "audio_ack_buffered_ms",
                           "audio_ack_consumed_ms", "audio_sent_sequence", "audio_packet_durations"):
            getattr(session, state_name).pop(previous.turn_id, None)
        if previous.playback_started:
            session.interrupt_pending[previous.turn_id] = turn.speech_onset_at
        clear_outgoing(session)
        await queue_event(session, {"type": "interrupt", "turn_id": previous.turn_id,
                                    "next_turn_id": turn.turn_id})
    return turn


async def _process_audio_frame(session: VoiceSession, frame: np.ndarray) -> None:
    if session.speaker_enrollment_kind:
        return
    event, completed_audio, score = session.vad.accept(frame)
    if event == "start":
        previous = session.current_turn
        if previous and not session.interrupt_enabled:
            session.ignored_speech = True
            log_event("speech_ignored_during_reply", session_id=session.session_id,
                      turn_id=previous.turn_id, reason="interrupt_disabled")
            await queue_event(session, {"type": "speech_ignored", "turn_id": previous.turn_id})
            return
        session.ignored_speech = False
        turn = await activate_turn(session)
        log_event("speech_started", session_id=session.session_id, turn_id=turn.turn_id,
                  vad_probability=round(score, 3), speaker_filter=session.speaker_filter_enabled)
        await queue_event(session, {"type": "speech_started", "turn_id": turn.turn_id,
                                    "vad_probability": round(score, 3)}, turn.turn_id)
    elif event == "end" and completed_audio is not None and session.current_turn:
        if session.ignored_speech:
            session.ignored_speech = False
            return
        turn = session.current_turn
        turn.speech_end_at = time.monotonic()
        log_event("speech_ended", session_id=session.session_id, turn_id=turn.turn_id,
                  duration_ms=round(completed_audio.size * 1000 / SAMPLE_RATE, 1))
        await queue_event(session, {"type": "speech_ended", "turn_id": turn.turn_id}, turn.turn_id)
        task = asyncio.create_task(process_turn(session, turn, completed_audio), name=f"turn-{turn.turn_id}")
        session.tasks.add(task)
        task.add_done_callback(session.tasks.discard)
    elif event == "empty" and session.ignored_speech:
        session.ignored_speech = False
    elif event == "empty" and session.current_turn:
        await queue_event(session, {"type": "empty_turn"})
        session.current_turn = None


async def speaker_audio_worker(session: VoiceSession) -> None:
    """Score 160 ms chunks off the WebSocket loop; old/backlogged audio fails closed."""
    cfg = runtime.config.get("speaker", {})
    chunk_frames = 5
    threshold = float(cfg.get("start_threshold", 0.75))
    release = float(cfg.get("release_threshold", 0.50))
    needed = int(cfg.get("start_chunks", 2))
    max_age = float(cfg.get("max_frame_age_ms", 1000)) / 1000
    while not session.stopped:
        item = await session.speaker_queue.get()
        batch = [item]
        try:
            while len(batch) < chunk_frames:
                batch.append(await asyncio.wait_for(session.speaker_queue.get(), timeout=0.25))
        except asyncio.TimeoutError:
            for _ in batch:
                session.speaker_queue.task_done()
            session.speaker_candidate.clear()
            session.speaker_candidate_chunks = 0
            session.speaker_active = False
            continue
        gate_ended = False
        try:
            now = time.monotonic()
            if now - batch[0][1] > max_age or now - batch[-1][1] > max_age:
                session.speaker_metrics.append({"stale_drops": 1})
                session.speaker_candidate.clear()
                session.speaker_candidate_chunks = 0
                session.speaker_active = False
                if session.current_turn:
                    silence = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                    for _ in batch:
                        await _process_audio_frame(session, silence)
                continue
            profiles = {key: profile["embedding"] for key, profile in session.speaker_profiles.items()
                        if profile["enabled"]}
            if not session.speaker_filter_enabled:
                for frame, _received_at in batch:
                    await _process_audio_frame(session, frame)
                continue
            # Silero remains active, but the heavier speaker model runs only
            # while the activity gate detects speech.
            if session.speaker_activity_gate is not None:
                should_score, gate_started, gate_ended, pre_roll = (
                    session.speaker_activity_gate.observe([frame for frame, _ in batch])
                )
                if gate_started:
                    # Drop the previous utterance's rolling speaker state.
                    for profile_id in profiles:
                        runtime.speaker_model.clear(profile_id)
                    session.speaker_active = False
                    session.speaker_candidate.clear()
                    session.speaker_candidate_chunks = 0
                    # Keep pre-onset audio; release it only after Nomo verifies
                    # the enrolled speaker on the current utterance.
                    session.speaker_candidate.extend(
                        np.concatenate(prior_frames) for prior_frames in pre_roll
                    )
                if not should_score:
                    session.speaker_gate_skipped_batches += 1
                    session.speaker_active = False
                    session.speaker_candidate.clear()
                    session.speaker_candidate_chunks = 0
                    if session.current_turn:
                        silence = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                        for _frame, _received_at in batch:
                            await _process_audio_frame(session, silence)
                    continue

            chunk = np.concatenate([frame for frame, _received_at in batch])
            try:
                if profiles:
                    session.speaker_score_invocations += 1
                    score, elapsed = await asyncio.to_thread(runtime.speaker_model.score, profiles, chunk)
                else:
                    score, elapsed = 0.0, 0.0
            except Exception as exc:
                log_event("speaker_scoring_failed", session_id=session.session_id, error=type(exc).__name__)
                await queue_event(session, {"type": "speaker_error", "message": "Speaker filter failed closed."})
                session.speaker_candidate.clear()
                session.speaker_candidate_chunks = 0
                session.speaker_active = False
                if session.current_turn:
                    for frame, _ in batch:
                        await _process_audio_frame(session, np.zeros_like(frame))
                continue
            session.speaker_metrics.append({"inference_ms": elapsed})
            if session.speaker_active and score >= release:
                for offset in range(0, chunk.size, VAD_FRAME_SAMPLES):
                    frame = chunk[offset:offset + VAD_FRAME_SAMPLES]
                    if frame.size == VAD_FRAME_SAMPLES:
                        await _process_audio_frame(session, frame)
            elif not session.speaker_active and score >= threshold:
                if session.speaker_candidate_chunks < needed:
                    session.speaker_candidate.append(chunk)
                    session.speaker_candidate_chunks += 1
                if session.speaker_candidate_chunks >= needed:
                    session.speaker_active = True
                    buffered = np.concatenate(session.speaker_candidate)
                    session.speaker_candidate.clear()
                    for offset in range(0, buffered.size, VAD_FRAME_SAMPLES):
                        frame = buffered[offset:offset + VAD_FRAME_SAMPLES]
                        if frame.size == VAD_FRAME_SAMPLES:
                            await _process_audio_frame(session, frame)
            else:
                session.speaker_active = False
                session.speaker_candidate.clear()
                session.speaker_candidate_chunks = 0
                zeros = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                for _frame, _received_at in batch:
                    await _process_audio_frame(session, zeros)
        finally:
            if gate_ended:
                # The next utterance starts with fresh speaker-scoring context.
                session.speaker_active = False
                session.speaker_candidate.clear()
                session.speaker_candidate_chunks = 0
                for profile_id in session.speaker_profiles:
                    runtime.speaker_model.clear(profile_id)
            for _ in batch:
                session.speaker_queue.task_done()


async def receive_audio(session: VoiceSession, payload: bytes) -> None:
    if len(payload) != VAD_FRAME_SAMPLES * 2:
        await queue_event(session, {"type": "error", "message": "Unexpected microphone frame size."})
        return
    frame = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if session.speaker_enrollment_kind:
        max_seconds = float(runtime.config.get("speaker", {}).get("enrollment_max_seconds", 45))
        elapsed = time.monotonic() - session.speaker_enrollment_started
        if elapsed > max_seconds:
            session.speaker_enrollment_kind = None
            session.speaker_enrollment_frames.clear()
            session.speaker_enrollment_received_samples = 0
            session.speaker_enrollment_speech_samples = 0
            session.speaker_enrollment_progress_task = None
            log_event("speaker_enrollment_button", session_id=session.session_id,
                      state="timed_out", held=False, elapsed_seconds=round(elapsed, 1))
            await queue_event(session, {"type": "speaker_error", "message": "Enrollment timed out; the previous profile was kept."})
            return
        probability = session.enrollment_vad.probability(frame)
        speech_frame = probability >= session.enrollment_vad.threshold
        max_recorded_samples = int(max_seconds * SAMPLE_RATE)
        if session.speaker_enrollment_received_samples + frame.size > max_recorded_samples:
            session.speaker_enrollment_kind = None
            session.speaker_enrollment_frames.clear()
            session.speaker_enrollment_received_samples = 0
            session.speaker_enrollment_speech_samples = 0
            log_event("speaker_enrollment_button", session_id=session.session_id,
                      state="recording_limit_reached", held=False, elapsed_seconds=round(elapsed, 1))
            await queue_event(session, {"type": "speaker_error", "message": "Enrollment reached its maximum recording duration; the previous profile was kept."})
            return
        session.speaker_enrollment_frames.append(frame.copy())
        session.speaker_enrollment_received_samples += frame.size
        if speech_frame:
            session.speaker_enrollment_speech_samples += frame.size
        session.speaker_enrollment_last_frame_at = time.monotonic()
        session.speaker_enrollment_last_vad_probability = round(probability, 3)
        session.speaker_enrollment_last_frame_is_speech = speech_frame
        return
    if session.speaker_filter_enabled:
        try:
            session.speaker_queue.put_nowait((frame, time.monotonic()))
        except asyncio.QueueFull:
            dropped = 0
            while True:
                try:
                    session.speaker_queue.get_nowait()
                    session.speaker_queue.task_done()
                    dropped += 1
                except asyncio.QueueEmpty:
                    break
            session.speaker_candidate.clear()
            session.speaker_candidate_chunks = 0
            session.speaker_metrics.append({"queue_drops": 1})
            await queue_event(session, {"type": "speaker_error", "message": "Speaker inference fell behind; audio was discarded to fail closed."})
            if session.current_turn:
                silence = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
                for _ in range(dropped):
                    await _process_audio_frame(session, silence)
        return
    await _process_audio_frame(session, frame)


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket) -> None:
    global active_session
    origin = websocket.headers.get("origin", "").rstrip("/")
    if not origin or origin not in allowed_origins():
        await websocket.close(code=1008, reason="Unexpected WebSocket origin")
        return
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
            speaker_queue=asyncio.Queue(maxsize=int(getattr(runtime, "config", {}).get("speaker", {}).get("max_queue_frames", 32))),
        )
        active_session = session
    log_event("session_started", session_id=session.session_id)

    sender = asyncio.create_task(websocket_sender(session), name=f"ws-sender-{session.session_id}")
    try:
        start_message = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        start = json.loads(start_message)
        interrupt_enabled = start.get("interrupt_enabled", True)
        search_enabled = start.get("search_enabled", False)
        speaker_filter_enabled = start.get("speaker_filter_enabled", False)
        csrf_token = start.get("csrf_token")
        if (start.get("type") != "start" or start.get("sample_rate") != SAMPLE_RATE
                or type(interrupt_enabled) is not bool or type(search_enabled) is not bool
                or type(speaker_filter_enabled) is not bool
                or not isinstance(csrf_token, str) or not secrets.compare_digest(
                    csrf_token, websocket.cookies.get("el_voice_session", "")
                )):
            await queue_event(session, {"type": "error", "message": "Unsupported audio format."})
            return
        session.interrupt_enabled = interrupt_enabled
        remote_mode = getattr(runtime, "mode", "local") == "remote"
        session.search_enabled = search_enabled if remote_mode else False
        try:
            session.speaker_profiles = await asyncio.to_thread(profile_store().load)
        except Exception as exc:
            log_event("speaker_profiles_invalid", session_id=session.session_id, error=type(exc).__name__)
            session.speaker_profiles = {}
        if speaker_filter_enabled and (getattr(runtime, "speaker_model", None) is None
                or not any(item["enabled"] for item in session.speaker_profiles.values())):
            await queue_event(session, {"type": "speaker_error", "message": "Filtering needs a valid local model and an enabled speaker profile."})
            return
        session.speaker_filter_enabled = speaker_filter_enabled
        if session.speaker_filter_enabled:
            session.speaker_activity_gate = SpeakerActivityGate(runtime.new_speaker_vad())
            session.speaker_worker = asyncio.create_task(
                speaker_audio_worker(session), name=f"speaker-filter-{session.session_id}"
            )
        await queue_event(
            session,
            {
                "type": "session_started",
                "session_id": session.session_id,
                "sample_rate": SAMPLE_RATE,
                "vad": "silero-streaming",
                "echo_cancellation": start.get("echo_cancellation"),
                "headphones_required": True,
                "interrupt_enabled": session.interrupt_enabled,
                "search_enabled": session.search_enabled,
                "speaker_filter_enabled": session.speaker_filter_enabled,
                "speaker_filter_state": "enabled" if session.speaker_filter_enabled else "filtering off",
                "speaker_model_available": getattr(runtime, "speaker_model", None) is not None,
                "speaker_profiles": public_profiles(session.speaker_profiles),
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
        if session.speaker_enrollment_kind:
            elapsed = max(0.0, time.monotonic() - session.speaker_enrollment_started)
            log_event("speaker_enrollment_button", session_id=session.session_id,
                      state="session_ended", held=False, elapsed_seconds=round(elapsed, 1),
                      recorded_seconds=round(session.speaker_enrollment_received_samples / SAMPLE_RATE, 1),
                      speech_seconds=round(session.speaker_enrollment_speech_samples / SAMPLE_RATE, 1))
            session.speaker_enrollment_kind = None
            session.speaker_enrollment_frames.clear()
        session.stopped = True
        session.pending_image = None
        session.pending_image_id = None
        for turn in session.turns.values():
            turn.cancel_event.set()
        sender.cancel()
        if session.speaker_worker:
            session.speaker_worker.cancel()
        clear_outgoing(session)
        for task in tuple(session.tasks):
            task.cancel()
        if session.tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tuple(session.tasks), return_exceptions=True), timeout=2
                )
            except asyncio.TimeoutError:
                log_event("session_cleanup_timeout", session_id=session.session_id,
                          active_turn_tasks=sum(not task.done() for task in session.tasks))
        await clear_llm_context(session.session_id)
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await sender
        if session.speaker_worker:
            with suppress(asyncio.CancelledError):
                await session.speaker_worker
        speaker_model = getattr(runtime, "speaker_model", None)
        if speaker_model is not None:
            for profile_id in session.speaker_profiles:
                speaker_model.clear(profile_id)
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
