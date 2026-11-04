"""Per-connection conversation state machine with turn isolation and barge-in.

Audio frames arrive continuously from the browser for the whole session. The
streaming VAD decides when a user utterance starts and ends; each utterance is
one turn with a unique ``turn_id`` that is propagated through recognition,
generation, synthesis and browser events. A new speech onset cancels the active
turn immediately, flushes browser playback and starts the next turn. Delayed
results from an older turn are checked against the cancelled turn identity and
never reach the browser.

No transcript, audio sample or model output is written to disk or to logs here;
only non-sensitive timing counters are recorded.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import struct
import threading
import time

from .llm_client import LlmClientError
from .tts import SpeakableChunker
from .turnids import TurnIdSource
from .vad import FRAME_BYTES, SpeechSegmenter

_DEFAULT_TURN_IDS = TurnIdSource()

logger = logging.getLogger("elvoice.pipeline")

AUDIO_MAGIC = b"EVA1"
AUDIO_HEADER = struct.Struct("<4sIIHH")  # magic, turn_id, sequence, sample_rate, reserved


class Turn:
    """One user utterance and the assistant answer that belongs to it."""

    __slots__ = (
        "id", "cancel", "task", "state", "cancelled", "audio_sent", "answer_text",
        "speech_end_ts", "stt_done_ts", "first_token_ts", "first_audio_ts",
    )

    def __init__(self, turn_id: int):
        self.id = turn_id
        self.cancel = threading.Event()
        self.task: asyncio.Task | None = None
        self.state = "capturing"
        self.cancelled = False
        self.audio_sent = False
        self.answer_text = ""
        self.speech_end_ts: float | None = None
        self.stt_done_ts: float | None = None
        self.first_token_ts: float | None = None
        self.first_audio_ts: float | None = None


class Metrics:
    """In-memory, content-free timing samples for the status endpoint."""

    def __init__(self):
        self._response = collections.deque(maxlen=200)
        self._barge = collections.deque(maxlen=200)
        self._client_stop = collections.deque(maxlen=200)
        self._counters: collections.Counter[str] = collections.Counter()

    def response(self, milliseconds: float) -> None:
        self._response.append(float(milliseconds))

    def barge_in(self, milliseconds: float) -> None:
        self._barge.append(float(milliseconds))

    def client_stop(self, milliseconds: float) -> None:
        self._client_stop.append(float(milliseconds))

    def count(self, name: str) -> None:
        self._counters[name] += 1

    @staticmethod
    def _summary(values: collections.deque) -> dict:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        count = len(ordered)
        p50 = ordered[max(0, (count - 1) // 2)]
        p95 = ordered[min(count - 1, int(round(0.95 * (count - 1))))]
        return {
            "count": count,
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "max": round(ordered[-1], 1),
        }

    def snapshot(self) -> dict:
        return {
            "counters": dict(self._counters),
            "response_latency_ms": self._summary(self._response),
            "barge_in_dispatch_ms": self._summary(self._barge),
            "client_playback_stop_ms": self._summary(self._client_stop),
        }


class VoiceSession:
    """One browser microphone session with one active turn at a time."""

    def __init__(self, session_id: str, cfg: dict, engines, metrics: Metrics,
                 send_json, send_audio, next_turn_id=None, llm=None):
        self.session_id = session_id
        self._cfg = cfg
        self._engines = engines
        self._metrics = metrics
        self._send_json = send_json
        self._send_audio = send_audio
        self._segmenter = SpeechSegmenter(cfg["audio"], cfg["vad"])
        self._pending = bytearray()
        self._turn: Turn | None = None
        self._state = "idle"
        self._closed = False
        self._vad_failures = 0
        self._output_rate = int(cfg["audio"]["output_sample_rate"])
        self._next_turn_id = next_turn_id or _DEFAULT_TURN_IDS.next
        self._turn_ids: dict[int, int] = {}
        self._llm = llm
        self.last_prompt: dict = {}

    # -- session lifecycle -------------------------------------------------
    async def start(self) -> dict:
        logger.info("session %s: start requested", self.session_id)
        await self._engines.wait_ready()
        if self._llm is None:
            raise EngineError("the internal LLM service client is not configured")
        try:
            await self._llm.start()
        except LlmClientError as exc:
            raise EngineError(str(exc)) from exc
        self._engines.vad.reset_states()
        self._segmenter.reset()
        self._pending.clear()
        self._turn_ids.clear()
        self.last_prompt = {}
        self._state = "listening"
        logger.info("session %s: ready", self.session_id)
        vad_cfg = self._cfg["vad"]
        return {
            "type": "ready",
            "session_id": self.session_id,
            "input_sample_rate": int(self._cfg["audio"]["input_sample_rate"]),
            "output_sample_rate": self._output_rate,
            "headphones_required": True,
            "echo_cancellation": "best_effort",
            "vad": {
                "threshold": vad_cfg["threshold"],
                "neg_threshold": vad_cfg["neg_threshold"],
                "min_speech_ms": vad_cfg["min_speech_ms"],
                "end_silence_ms": vad_cfg["end_silence_ms"],
                "pre_roll_ms": self._cfg["audio"]["pre_roll_ms"],
            },
        }

    async def stop(self) -> None:
        logger.info("session %s: stop requested", self.session_id)
        self.last_prompt = {}
        self._turn_ids.clear()
        self._closed = True
        turn = self._turn
        if turn is not None:
            await self._cancel(turn, "stop")
            self._turn = None
        task = turn.task if turn is not None else None
        if task is not None and not task.done():
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014 - teardown must not raise
                pass
        if self._llm is not None:
            try:
                await self._llm.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
        await self._set_state("idle")

    # -- audio input and VAD ----------------------------------------------
    async def feed(self, data: bytes) -> None:
        if self._closed:
            return
        self._pending.extend(data)
        events: list[dict] = []
        while len(self._pending) >= FRAME_BYTES:
            frame = bytes(self._pending[:FRAME_BYTES])
            del self._pending[:FRAME_BYTES]
            probability = self._speech_probability(frame)
            events.extend(self._segmenter.feed(frame, probability))
        for event in events:
            if self._closed:
                break
            if event["type"] == "onset":
                await self._on_onset(event["turn_id"])
            elif event["type"] == "end":
                await self._on_utterance_end(event)

    def _speech_probability(self, frame: bytes) -> float:
        try:
            return self._engines.vad.speech_probability(frame)
        except Exception as exc:  # noqa: BLE001 - a VAD failure must not kill the session
            self._vad_failures += 1
            if self._vad_failures == 1:
                logger.warning("VAD inference failed: %s", type(exc).__name__)
            return 0.0

    async def _on_onset(self, sequence: int) -> None:
        onset_ts = time.perf_counter()
        turn_id = self._next_turn_id()
        self._turn_ids[sequence] = turn_id
        previous = self._turn.state if self._turn is not None else "none"
        logger.info("turn=%s: onset (previous=%s)", turn_id, previous)
        if self._turn is not None and not self._turn.cancelled:
            await self._cancel(self._turn, "barge_in", onset_ts)
        self._turn = Turn(turn_id)
        self._metrics.count("user_utterances")
        await self._send_json({"type": "user_speech_start", "turn_id": turn_id})
        await self._set_state("listening")

    async def _on_utterance_end(self, event: dict) -> None:
        turn = self._turn
        expected = self._turn_ids.pop(event["turn_id"], None)
        if turn is None or expected is None or turn.id != expected or turn.cancelled:
            return
        logger.info(
            "turn=%s: utterance end speech_ms=%.0f audio_ms=%.0f",
            turn.id, float(event["speech_ms"]), len(event["pcm"]) / 32.0,
        )
        if float(event["speech_ms"]) < float(self._cfg["vad"]["min_speech_ms"]):
            self._turn = None
            self._metrics.count("empty_turns")
            await self._send_json({"type": "empty_turn", "turn_id": turn.id})
            await self._set_state("listening")
            return
        turn.speech_end_ts = time.perf_counter()
        turn.state = "processing"
        turn.task = asyncio.create_task(self._run_turn(turn, event["pcm"]))
        await self._set_state("processing")

    # -- turn processing ---------------------------------------------------
    async def _run_turn(self, turn: Turn, pcm: bytes) -> None:
        try:
            logger.info("turn=%s: stt queued", turn.id)
            text = await self._engines.transcribe(pcm)
            turn.stt_done_ts = time.perf_counter()
            logger.info("turn=%s: stt done chars=%d", turn.id, len(text))
            if turn.cancelled:
                return
            await self._send_json({"type": "user_text", "turn_id": turn.id, "text": text})
            if not text:
                self._metrics.count("empty_turns")
                await self._send_json({"type": "empty_turn", "turn_id": turn.id})
                await self._finish_turn(turn)
                return
            await self._generate(turn, text)
        except asyncio.CancelledError:
            raise
        except LlmClientError as exc:
            turn.cancel.set()
            logger.warning("turn=%s: llm failed (%s)", turn.id, exc.code)
            if not turn.cancelled:
                await self._send_json({
                    "type": "error",
                    "code": exc.code,
                    "message": str(exc),
                    "retryable": exc.retryable,
                })
                await self._finish_turn(turn)
        except Exception as exc:  # noqa: BLE001 - report to the browser, keep the session
            turn.cancel.set()
            logger.exception("turn=%s: failed (%s)", turn.id, type(exc).__name__)
            if not turn.cancelled:
                await self._send_json({
                    "type": "error",
                    "code": "turn_failed",
                    "message": "The voice pipeline could not finish this turn.",
                    "retryable": True,
                })
                await self._finish_turn(turn)

    async def _generate(self, turn: Turn, text: str) -> None:
        """Stream one turn through the internal LLM service.

        The voice pipeline sends only the current final utterance. The service
        owns the bounded conversation history and the system prompt, and returns
        text deltas plus optional tool/source events.
        """
        if self._llm is None:
            raise EngineError("the internal LLM service client is not configured")
        logger.info("turn=%s: llm request chars=%d", turn.id, len(text))
        tts_cfg = self._cfg["tts"]
        chunker = SpeakableChunker(
            int(tts_cfg["min_chunk_chars"]),
            int(tts_cfg["soft_chunk_chars"]),
            int(tts_cfg["max_chunk_chars"]),
            int(tts_cfg.get("first_chunk_chars", 50)),
        )
        sequence = 0
        stream = self._llm.generate(turn.id, text)
        try:
            async for event in stream:
                if turn.cancelled:
                    return
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind == "start":
                    stats = event.get("prompt")
                    if isinstance(stats, dict):
                        self.last_prompt = dict(stats)
                elif kind == "delta":
                    delta = event.get("text") or ""
                    if not isinstance(delta, str) or not delta:
                        continue
                    if turn.first_token_ts is None:
                        turn.first_token_ts = time.perf_counter()
                        logger.info(
                            "turn=%s: first token after %.0f ms",
                            turn.id, (turn.first_token_ts - turn.stt_done_ts) * 1000.0,
                        )
                    turn.answer_text += delta
                    await self._send_json({
                        "type": "assistant_text", "turn_id": turn.id, "delta": delta,
                    })
                    for chunk in chunker.feed(delta):
                        if turn.cancelled:
                            return
                        sequence = await self._speak(turn, chunk, sequence)
                elif kind == "sources":
                    sources = event.get("sources")
                    if isinstance(sources, list):
                        await self._send_json({
                            "type": "assistant_sources",
                            "turn_id": turn.id,
                            "sources": sources,
                        })
                elif kind == "tool":
                    await self._send_json({
                        "type": "assistant_tool",
                        "turn_id": turn.id,
                        "name": event.get("name", ""),
                        "query": event.get("query", ""),
                    })
                elif kind == "cancelled":
                    return
                elif kind == "done":
                    break
                elif kind == "error":
                    raise LlmClientError(
                        event.get("message") or "The LLM service reported an error.",
                        code=event.get("code", "llm_error"),
                        retryable=bool(event.get("retryable", True)),
                    )
            for chunk in chunker.flush():
                if turn.cancelled:
                    return
                sequence = await self._speak(turn, chunk, sequence)
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:  # noqa: BLE001 - closing a cancelled stream is best effort
                    pass

        if turn.cancelled:
            return
        await self._send_json({"type": "turn_end", "turn_id": turn.id})
        if turn.first_audio_ts is not None and turn.speech_end_ts is not None:
            logger.info(
                "turn=%s: first audio dispatched after %.0f ms",
                turn.id, (turn.first_audio_ts - turn.speech_end_ts) * 1000.0,
            )
            response_ms = (turn.first_audio_ts - turn.speech_end_ts) * 1000.0
            self._metrics.response(response_ms)
            await self._send_json({
                "type": "timing", "turn_id": turn.id, "response_ms": round(response_ms, 1),
            })
            stt_ms = (turn.stt_done_ts - turn.speech_end_ts) * 1000.0 if turn.stt_done_ts else 0.0
            llm_ms = (turn.first_audio_ts - turn.first_token_ts) * 1000.0 if turn.first_token_ts else 0.0
            logger.info(
                "turn=%s stt_ms=%.0f llm_to_first_audio_ms=%.0f response_ms=%.0f",
                turn.id, stt_ms, llm_ms, response_ms,
            )
        if not turn.audio_sent:
            await self._finish_turn(turn)

    async def _speak(self, turn: Turn, chunk: str, sequence: int) -> int:
        pcm = await self._engines.synthesize(chunk)
        if turn.cancelled:
            return sequence
        sequence += 1
        header = AUDIO_HEADER.pack(AUDIO_MAGIC, turn.id, sequence, self._output_rate, 0)
        await self._send_audio(header + pcm)
        turn.audio_sent = True
        if turn.first_audio_ts is None:
            turn.first_audio_ts = time.perf_counter()
            turn.state = "speaking"
            logger.info("turn=%s: first tts chunk dispatched (%d bytes)", turn.id, len(pcm))
            await self._set_state("speaking")
        return sequence

    async def _cancel(self, turn: Turn, reason: str, onset_ts: float | None = None) -> None:
        if turn.cancelled:
            return
        logger.info(
            "turn=%s: cancel reason=%s state=%s audio_sent=%s",
            turn.id, reason, turn.state, turn.audio_sent,
        )
        turn.cancelled = True
        turn.state = "cancelled"
        turn.cancel.set()
        if self._llm is not None:
            try:
                await self._llm.cancel(turn.id)
            except Exception:  # noqa: BLE001 - cancellation is best effort
                pass
        task = turn.task
        if task is not None and not task.done():
            task.cancel()
        if turn.audio_sent:
            await self._send_json({"type": "flush", "turn_id": turn.id, "reason": reason})
            if onset_ts is not None:
                dispatch_ms = (time.perf_counter() - onset_ts) * 1000.0
                self._metrics.barge_in(dispatch_ms)
                logger.info("turn=%s barge_in_dispatch_ms=%.0f", turn.id, dispatch_ms)
        self._metrics.count("cancelled_turns")
        await self._set_state("listening")

    async def _finish_turn(self, turn: Turn) -> None:
        if self._turn is not turn or turn.cancelled:
            return
        turn.state = "done"
        logger.info("turn=%s: completed", turn.id)
        self._metrics.count("turns_completed")
        self._turn = None
        await self._set_state("listening")

    # -- client notifications ---------------------------------------------
    async def playback_done(self, turn_id: int) -> None:
        logger.info("turn=%s: playback done from browser", turn_id)
        turn = self._turn
        if turn is None or turn.id != turn_id or turn.cancelled:
            return
        if turn.state == "speaking":
            await self._finish_turn(turn)

    def client_playback_stopped(self, turn_id: int, milliseconds: float) -> None:
        logger.info("turn=%s: browser stopped playback in %.1f ms", turn_id, float(milliseconds))
        if 0.0 <= float(milliseconds) <= 10000.0:
            self._metrics.client_stop(float(milliseconds))

    # -- helpers -----------------------------------------------------------
    async def _set_state(self, state: str) -> None:
        if self._state == state:
            return
        self._state = state
        await self._send_json({"type": "state", "state": state})

    @property
    def state(self) -> str:
        return self._state
