"""Focused review regressions for the local voice response path.

SDDTDD SPEC: ../../tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-REVIEW.md

Run in the voice image with:
    python -m unittest discover -s /workspace/tests -p 'test_review_*.py'
"""

from __future__ import annotations

import asyncio
import io
import importlib.resources
import json
import logging
import socket
import threading
import time
import unittest
from contextlib import nullcontext
import uuid
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import app
import voice_pipeline


class StubRuntime:
    def __init__(self, deltas=None, config=None):
        self.synthesized = []
        self.histories = []
        self.generation_finished = threading.Event()
        self.deltas = deltas or ["**Build** Mjollnir ✨ and read https://example.com/guide now."]
        self.config = config or {"speech": {"minimum_chunk_chars": 18, "first_chunk_max_chars": 60,
                                             "max_chunk_chars": 80}}

    def transcribe(self, _audio, cancel_event=None):
        return "Give me a short answer."

    def llm_stream(self, _text, _cancel, history):
        self.histories.append(list(history))
        yield from self.deltas
        self.generation_finished.set()

    def synthesize(self, text, cancel_event=None):
        self.synthesized.append(text)
        return b"\x00\x00" * 480, 24000


def turn_id():
    return uuid.uuid4().hex


class SpeakableResponseTests(unittest.TestCase):
    def test_stream_boundaries_preserve_spaces_and_hard_maximum(self):
        chunks, tail = app._speakable_chunks("Hello ", max_chars=20, min_chars=8)
        next_chunks, tail = app._speakable_chunks(tail + "world", max_chars=20, min_chars=8)
        self.assertEqual(chunks + next_chunks, [])
        self.assertEqual(app._sanitize_speech(tail).strip(), "Hello world")

        long_chunks, remainder = app._speakable_chunks("word " * 25 + ".", max_chars=20, min_chars=12)
        emitted = long_chunks + ([remainder.strip()] if remainder.strip() else [])
        self.assertTrue(emitted)
        self.assertTrue(all(len(chunk) <= 20 for chunk in emitted))
        self.assertTrue("." in " ".join(emitted), "the sentence terminator must be retained")

    def test_streamed_tts_chunks_drop_markup_urls_and_emoji(self):
        original_runtime = app.runtime
        stub = StubRuntime()
        app.runtime = stub
        session = SimpleNamespace(
            stopped=False,
            session_id="review-test",
            outgoing=asyncio.Queue(maxsize=8),
            history=[],
            current_turn=None,
            turns={},
        )
        session.is_current = lambda turn: (
            not session.stopped
            and session.current_turn is turn
            and not turn.cancel_event.is_set()
        )
        turn = SimpleNamespace(
            turn_id=turn_id(),
            cancel_event=threading.Event(),
        )
        session.current_turn = turn

        async def run_turn():
            await app.process_turn(session, turn, b"audio")

        try:
            asyncio.run(run_turn())
            chunks = [
                item.event["text"]
                for item in list(session.outgoing._queue)
                if item.event and item.event.get("type") == "audio_chunk"
            ]
        finally:
            app.runtime = original_runtime

        self.assertTrue(chunks, "the response should produce at least one TTS chunk")
        spoken = " ".join(chunks)
        self.assertNotIn("**", spoken)
        self.assertNotIn("https://", spoken)
        self.assertNotIn("✨", spoken)
        self.assertIn("Mjollnir", spoken)
        tts_input = " ".join(stub.synthesized)
        self.assertNotIn("**", tts_input)
        self.assertNotIn("https://", tts_input)
        self.assertNotIn("✨", tts_input)
        self.assertIn("Mjollnir", tts_input)

    def test_fragmented_markup_urls_long_phrases_and_configured_chunks(self):
        original_runtime = app.runtime
        runtime = StubRuntime(
            deltas=[
                "**Build** Mjo", "llnir** with a very long", " unpunctuated phrase that keeps going",
                " and going until it needs a split. Read https://exa", "mple.com/guide and keep going. ✨",
            ],
            config={"speech": {"minimum_chunk_chars": 12, "first_chunk_max_chars": 20,
                               "max_chunk_chars": 45}},
        )
        app.runtime = runtime
        session = SimpleNamespace(
            stopped=False, session_id="chunking-regression", outgoing=asyncio.Queue(maxsize=32),
            history=[], current_turn=None, turns={},
        )
        session.is_current = lambda turn: not session.stopped and session.current_turn is turn and not turn.cancel_event.is_set()
        turn = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
        session.current_turn = turn
        try:
            asyncio.run(app.process_turn(session, turn, b"audio"))
            chunks = [item.event["text"] for item in list(session.outgoing._queue)
                      if item.event and item.event.get("type") == "audio_chunk"]
        finally:
            app.runtime = original_runtime
        self.assertTrue(chunks)
        self.assertLessEqual(len(chunks[0]), 20, "the first chunk should be short enough to start playback early")
        self.assertTrue(all(len(chunk) <= 45 for chunk in chunks), "configured maximum chunk length must be respected")
        spoken = " ".join(runtime.synthesized)
        self.assertNotIn("**", spoken)
        self.assertNotIn("https://", spoken)
        self.assertNotIn("example.com", spoken)
        self.assertNotIn("✨", spoken)
        self.assertIn("Mjollnir", spoken)


class StartupValidationTests(unittest.TestCase):
    def test_configuration_errors_are_exposed_by_status_endpoint(self):
        invalid = {
            "tts": {"model": "tts/v4_cyrillic.pt", "speaker": "b_lez", "sample_rate": 24000, "cpu_threads": 4},
            "stt": {"model": "stt/small", "language": "ru", "cpu_threads": 4},
            "vad": {"sample_rate": 16000, "frame_samples": 512, "threshold": 0.55, "negative_threshold": 0.35,
                    "start_frames": 3, "end_silence_ms": 640, "min_speech_ms": 240, "pre_roll_ms": 320,
                    "max_utterance_seconds": 45},
            "audio": {"max_buffered_audio_ms": 2000, "packet_duration_ms": 800},
            "speech": {"minimum_chunk_chars": 18, "first_chunk_max_chars": 60, "max_chunk_chars": 80},
        }
        cases = [
            ("tts.sample_rate", lambda cfg: cfg["tts"].update(sample_rate=12345)),
            ("stt.cpu_threads", lambda cfg: cfg["stt"].update(cpu_threads=0)),
            ("vad.threshold", lambda cfg: cfg["vad"].update(threshold=1.5)),
            ("vad.sample_rate", lambda cfg: cfg["vad"].update(sample_rate=8000)),
            ("vad hysteresis", lambda cfg: cfg["vad"].update(negative_threshold=0.7)),
            ("vad duration", lambda cfg: cfg["vad"].update(max_utterance_seconds=-1)),
            ("audio buffer limit", lambda cfg: cfg["audio"].update(max_buffered_audio_ms=0)),
            ("speech chunk order", lambda cfg: cfg["speech"].update(first_chunk_max_chars=90)),
        ]

        async def fetch_status(config):
            with patch.object(voice_pipeline, "_load_settings", return_value=config):
                runtime = voice_pipeline.VoiceRuntime()
            class ReadyLLM:
                async def wait_ready(self):
                    return {"ready": True, "mode": "local"}
            runtime.llm_client = ReadyLLM()
            with patch.object(runtime, "_load_models", return_value=None):
                previous_runtime, previous_boot = app.runtime, app.boot_task
                app.runtime, app.boot_task = runtime, None
                try:
                    async with app.app.router.lifespan_context(app.app):
                        await app.boot_task
                        messages = []

                        async def receive():
                            return {"type": "http.request", "body": b"", "more_body": False}

                        async def send(message):
                            messages.append(message)

                        scope = {
                            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                            "http_version": "1.1", "method": "GET", "scheme": "http", "path": "/api/status",
                            "raw_path": b"/api/status", "query_string": b"", "root_path": "",
                            "headers": [], "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000),
                        }
                        await app.app(scope, receive, send)
                        return json.loads(messages[-1]["body"])
                finally:
                    app.runtime, app.boot_task = previous_runtime, previous_boot

        for name, mutate in cases:
            with self.subTest(setting=name):
                config = json.loads(json.dumps(invalid))
                mutate(config)
                status = asyncio.run(fetch_status(config))
                self.assertFalse(status["ready"])
                self.assertEqual(status["stage"], "error")
                self.assertIsNotNone(status["error"])

        status = asyncio.run(fetch_status(invalid))
        self.assertTrue(status["ready"], "valid settings should allow normal initialization")

    def test_non_object_settings_root_is_reported_as_not_ready(self):
        async def status_for(root):
            with patch.object(voice_pipeline, "_load_settings", return_value=root):
                runtime = voice_pipeline.VoiceRuntime()
            with patch.object(runtime, "_load_models", return_value=None):
                await runtime.initialize()
            previous = app.runtime
            app.runtime = runtime
            try:
                response = await app.status()
                return json.loads(response.body)
            finally:
                app.runtime = previous

        for root in ([], None, "settings", 1):
            with self.subTest(root=root):
                status = asyncio.run(status_for(root))
                self.assertFalse(status["ready"])
                self.assertEqual(status["stage"], "error")
                self.assertEqual(status["error"], "ValueError")


class LogPrivacyTests(unittest.TestCase):
    def test_requested_text_and_errors_are_logged_without_audio_payload(self):
        original_runtime = app.runtime
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        app.logger.addHandler(handler)

        class Runtime(StubRuntime):
            def transcribe(self, _audio, cancel_event=None):
                return "transcript marker"

            def llm_stream(self, _text, _cancel, _history):
                yield "reply marker."

            def synthesize(self, _text, cancel_event=None):
                return b"LOCAL_AUDIO_SECRET_SENTINEL!", 24000

        class BrokenRuntime(Runtime):
            def transcribe(self, _audio, cancel_event=None):
                raise RuntimeError("synthetic inference failure")

        session = SimpleNamespace(
            stopped=False, session_id="log-test", outgoing=asyncio.Queue(maxsize=16), history=[],
            current_turn=None, turns={},
        )
        session.is_current = lambda turn: not session.stopped and session.current_turn is turn and not turn.cancel_event.is_set()

        async def run():
            app.runtime = Runtime()
            turn = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
            session.current_turn = turn
            await app.process_turn(session, turn, b"audio")
            app.runtime = BrokenRuntime()
            failed_turn = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
            session.current_turn = failed_turn
            await app.process_turn(session, failed_turn, b"audio")

        try:
            asyncio.run(run())
        finally:
            app.runtime = original_runtime
            app.logger.removeHandler(handler)
            handler.close()
        output = stream.getvalue()
        self.assertIn('"event":"user_transcript"', output)
        self.assertIn("transcript marker", output)
        self.assertIn('"event":"assistant_reply"', output)
        self.assertIn("reply marker", output)
        self.assertIn('"event":"turn_error"', output)
        self.assertIn("synthetic inference failure", output)
        self.assertNotIn("LOCAL_AUDIO_SECRET_SENTINEL", output)


class AudioPacketTests(unittest.IsolatedAsyncioTestCase):
    async def test_binary_audio_contains_turn_identity_sequence_and_rate(self):
        class Socket:
            def __init__(self, session):
                self.binary = []
                self.session = session

            async def send_text(self, _text):
                pass

            async def send_bytes(self, payload):
                self.binary.append(payload)
                if len(self.binary) == 2:
                    self.session.stopped = True

        original_runtime = app.runtime
        app.runtime = StubRuntime()
        session = app.VoiceSession("session-packet", None, None)
        socket = Socket(session)
        session.websocket = socket
        turn = voice_pipeline.Turn("00112233445566778899aabbccddeeff")
        session.current_turn = turn
        session.remember_turn(turn)
        try:
            await app.synthesize_and_queue(session, turn, "First packet sentence.")
            await app.synthesize_and_queue(session, turn, "Second packet sentence.")
            sender = asyncio.create_task(app.websocket_sender(session))
            await asyncio.wait_for(session.outgoing.join(), timeout=1)
            await asyncio.wait_for(sender, timeout=1)
            self.assertEqual(len(socket.binary), 2)
            for sequence, packet in enumerate(socket.binary):
                self.assertEqual(packet[:4], b"ELVA")
                self.assertEqual(packet[4:20], bytes.fromhex(turn.turn_id))
                self.assertEqual(int.from_bytes(packet[20:24], "big"), sequence)
                self.assertEqual(int.from_bytes(packet[24:28], "big"), 24000)
                self.assertEqual(len(packet[28:]) % 2, 0)
                self.assertEqual(packet[28:], b"\x00\x00" * 480)
        finally:
            app.runtime = original_runtime


class TurnRegressionTests(unittest.IsolatedAsyncioTestCase):
    def make_session(self, session_id="turn-regression"):
        session = SimpleNamespace(
            stopped=False,
            session_id=session_id,
            outgoing=asyncio.Queue(maxsize=32),
            history=[],
            current_turn=None,
            turns={},
        )
        session.is_current = lambda turn: (
            not session.stopped
            and session.current_turn is turn
            and not turn.cancel_event.is_set()
        )
        return session

    async def test_contextual_followup_receives_previous_exchange(self):
        original_runtime = app.runtime
        runtime = StubRuntime()
        app.runtime = runtime
        session = self.make_session()
        try:
            for index in range(2):
                turn = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
                session.current_turn = turn
                await app.process_turn(session, turn, b"audio")
                while not session.outgoing.empty():
                    session.outgoing.get_nowait()
                    session.outgoing.task_done()
            self.assertEqual(runtime.histories[0], [])
            self.assertEqual(runtime.histories[1][0], {"role": "user", "content": "Give me a short answer."})
            self.assertEqual(runtime.histories[1][1]["role"], "assistant")
        finally:
            app.runtime = original_runtime

    async def test_first_audio_is_queued_before_generation_finishes(self):
        original_runtime = app.runtime
        runtime = StubRuntime()
        generated_later = threading.Event()

        def streaming_llm(_text, _cancel, _history):
            yield "First complete sentence. "
            generated_later.wait(timeout=2)
            yield "Second complete sentence."
            runtime.generation_finished.set()

        runtime.llm_stream = streaming_llm
        app.runtime = runtime
        session = self.make_session("streaming-regression")
        turn = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
        session.current_turn = turn
        task = asyncio.create_task(app.process_turn(session, turn, b"audio"))
        try:
            async with asyncio.timeout(1):
                while not any(
                    item.event and item.event.get("type") == "audio_chunk"
                    for item in list(session.outgoing._queue)
                ):
                    await asyncio.sleep(0.005)
            self.assertFalse(runtime.generation_finished.is_set())
            self.assertTrue(runtime.synthesized)
        finally:
            generated_later.set()
            try:
                await asyncio.wait_for(task, timeout=2)
            finally:
                app.runtime = original_runtime

    async def test_llm_token_queue_backpressures_producer_and_releases_on_cancel(self):
        original_runtime = app.runtime
        produced = 0
        finished = threading.Event()

        class FastRuntime(StubRuntime):
            def llm_stream(self, _text, _cancel, _history):
                nonlocal produced
                try:
                    for _ in range(1000):
                        produced += 1
                        yield "token"
                finally:
                    finished.set()

        app.runtime = FastRuntime()
        session = self.make_session("token-backpressure")
        turn = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
        session.current_turn = turn
        stream = app._llm_items(session, turn, "question")
        try:
            self.assertEqual(await anext(stream), {"type": "delta", "text": "token"})
            await asyncio.sleep(0.05)
            self.assertLessEqual(produced, 6, "the bounded token queue should stall a fast producer")
            turn.cancel_event.set()
            await asyncio.wait_for(stream.aclose(), timeout=1)
            self.assertTrue(await asyncio.to_thread(finished.wait, 1), "cancellation should release the token producer")
        finally:
            turn.cancel_event.set()
            await stream.aclose()
            app.runtime = original_runtime

    async def test_cancelled_stt_worker_does_not_accumulate_and_replacement_runs(self):
        original_runtime = app.runtime
        started = threading.Event()
        cancelled_first = threading.Event()
        counter_lock = threading.Lock()

        class SlowRuntime(StubRuntime):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.active = 0
                self.max_active = 0

            def transcribe(self, _audio, cancel_event=None):
                with counter_lock:
                    self.calls += 1
                    call = self.calls
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    if call == 1:
                        started.set()
                        if cancel_event is not None:
                            cancel_event.wait(timeout=2)
                            cancelled_first.set()
                    return f"question {call}"
                finally:
                    with counter_lock:
                        self.active -= 1

        runtime = SlowRuntime()
        app.runtime = runtime
        session = self.make_session("cancelled-worker-regression")
        obsolete = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
        replacement = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
        session.current_turn = obsolete
        obsolete_task = asyncio.create_task(app.process_turn(session, obsolete, b"first"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1), "first STT call should start")
            obsolete.cancel_event.set()
            session.current_turn = replacement
            replacement_task = asyncio.create_task(app.process_turn(session, replacement, b"second"))
            await asyncio.wait_for(asyncio.gather(obsolete_task, replacement_task), timeout=1)
            self.assertTrue(cancelled_first.is_set(), "the old inference must observe actual turn cancellation")
            self.assertEqual(runtime.max_active, 1, "obsolete STT workers must not accumulate")
            self.assertTrue(any(item.event and item.event.get("type") == "turn_complete"
                                for item in list(session.outgoing._queue)),
                            "the replacement turn must complete after residual inference exits")
        finally:
            pending = [task for task in (obsolete_task, locals().get("replacement_task"))
                       if task and not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            app.runtime = original_runtime

    async def test_obsolete_llm_and_tts_workers_do_not_accumulate(self):
        original_runtime = app.runtime

        for stage in ("llm", "tts"):
            with self.subTest(stage=stage):
                started = threading.Event()
                first_finished = threading.Event()
                counter_lock = threading.Lock()

                class SlowRuntime(StubRuntime):
                    def __init__(self):
                        super().__init__()
                        self.calls = 0
                        self.active = 0
                        self.max_active = 0
                        self.llm_lock = threading.Lock()
                        self.tts_lock = threading.Lock()

                    def begin_slow_call(self, cancel_event):
                        with counter_lock:
                            self.calls += 1
                            call = self.calls
                            self.active += 1
                            self.max_active = max(self.max_active, self.active)
                        if call == 1:
                            started.set()
                            cancel_event.wait(timeout=2)
                        return call

                    def llm_stream(self, _text, _cancel, _history):
                        if stage == "llm":
                            with self.llm_lock:
                                call = self.begin_slow_call(_cancel)
                                try:
                                    yield "This is the first complete sentence."
                                finally:
                                    with counter_lock:
                                        self.active -= 1
                                    if call == 1:
                                        first_finished.set()
                        else:
                            yield "This is the first complete sentence."

                    def synthesize(self, text, cancel_event=None):
                        self.synthesized.append(text)
                        if stage == "tts":
                            with self.tts_lock:
                                call = self.begin_slow_call(cancel_event)
                                try:
                                    return b"\x00\x00" * 480, 24000
                                finally:
                                    with counter_lock:
                                        self.active -= 1
                                    if call == 1:
                                        first_finished.set()
                        return b"\x00\x00" * 480, 24000

                runtime = SlowRuntime()
                app.runtime = runtime
                session = self.make_session(f"cancelled-{stage}")
                old = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
                new = SimpleNamespace(turn_id=turn_id(), cancel_event=threading.Event())
                session.current_turn = old
                old_task = asyncio.create_task(app.process_turn(session, old, b"old"))
                new_task = None
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 1), f"{stage} call should start")
                    cancelled_at = time.monotonic()
                    old.cancel_event.set()
                    session.current_turn = new
                    new_task = asyncio.create_task(app.process_turn(session, new, b"new"))
                    await asyncio.wait_for(asyncio.gather(old_task, new_task), timeout=1)
                    self.assertTrue(first_finished.is_set(), f"obsolete {stage} work must terminate")
                    residual_ms = (time.monotonic() - cancelled_at) * 1000
                    self.assertLess(residual_ms, 1000, f"measured residual {stage} work must remain bounded")
                    self.assertEqual(runtime.max_active, 1)
                finally:
                    for task in (old_task, new_task):
                        if task and not task.done():
                            task.cancel()
                    await asyncio.gather(old_task, *( [new_task] if new_task else []), return_exceptions=True)
                    app.runtime = original_runtime


class RuntimeInferenceLockTests(unittest.TestCase):
    def test_uncooperative_runtime_workers_serialize_across_repeated_replacements(self):
        class EngineProbe:
            def __init__(self):
                self.calls = 0
                self.active = 0
                self.max_active = 0
                self.started = threading.Event()
                self.replacement_started = threading.Event()
                self.release = threading.Event()
                self.guard = threading.Lock()

            def enter(self):
                with self.guard:
                    self.calls += 1
                    call = self.calls
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                if call % 2:
                    self.started.set()
                    self.release.wait(timeout=2)
                else:
                    self.replacement_started.set()
                return call

            def leave(self):
                with self.guard:
                    self.active -= 1

        class Segment:
            text = "replacement transcript"

        class Whisper:
            def __init__(self, probe):
                self.probe = probe

            def transcribe(self, *_args, **_kwargs):
                def segments():
                    self.probe.enter()
                    try:
                        yield Segment()
                    finally:
                        self.probe.leave()
                return segments(), None

        class Tensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def flatten(self):
                return self

            def numel(self):
                return 2

            def clamp(self, *_args):
                return self

            def __mul__(self, _value):
                return self

            def to(self, _dtype):
                return self

            def numpy(self):
                return np.array([0.25, -0.25], dtype=np.float32)

        class TTS:
            def __init__(self, probe):
                self.probe = probe

            def apply_tts(self, **_kwargs):
                self.probe.enter()
                try:
                    return Tensor()
                finally:
                    self.probe.leave()

        class Torch:
            int16 = object()

            @staticmethod
            def inference_mode():
                return nullcontext()

            @staticmethod
            def isfinite(_audio):
                return SimpleNamespace(all=lambda: True)

        runtime = voice_pipeline.VoiceRuntime()
        runtime.config = voice_pipeline._load_settings()
        probes = {stage: EngineProbe() for stage in ("stt", "tts")}
        runtime.whisper = Whisper(probes["stt"])
        runtime.tts = TTS(probes["tts"])
        runtime.torch = Torch()

        def invoke(stage, cancelled, result):
            if stage == "stt":
                result.append(runtime.transcribe(np.zeros(16000, dtype=np.float32), cancelled))
            else:
                result.append(runtime.synthesize("spoken text", cancelled)[0])

        for stage, probe in probes.items():
            for _attempt in range(3):
                probe.started.clear()
                probe.replacement_started.clear()
                probe.release.clear()
                obsolete_cancel = threading.Event()
                replacement_cancel = threading.Event()
                obsolete_result, replacement_result = [], []
                obsolete = threading.Thread(target=invoke, args=(stage, obsolete_cancel, obsolete_result))
                replacement = threading.Thread(target=invoke, args=(stage, replacement_cancel, replacement_result))
                obsolete.start()
                self.assertTrue(probe.started.wait(1), f"{stage} obsolete engine call should start")
                cancelled_at = time.monotonic()
                obsolete_cancel.set()
                replacement.start()
                self.assertFalse(probe.replacement_started.wait(0.05),
                                 f"{stage} replacement must wait while uncancellable inference owns the runtime lock")
                self.assertEqual(probe.active, 1)
                probe.release.set()
                obsolete.join(2)
                replacement.join(2)
                self.assertFalse(obsolete.is_alive())
                self.assertFalse(replacement.is_alive())
                self.assertTrue(probe.replacement_started.is_set())
                self.assertEqual(probe.max_active, 1, f"{stage} obsolete workers must not overlap replacements")
                self.assertLess(time.monotonic() - cancelled_at, 1,
                                f"{stage} replacement should finish promptly once residual work exits")
                if stage == "stt":
                    self.assertEqual(obsolete_result, [""], f"cancelled {stage} result must be discarded")
                self.assertTrue(replacement_result, f"replacement {stage} inference must complete")


class StreamingVADStateTests(unittest.TestCase):
    def test_context_and_recurrent_state_match_reference_sequence(self):
        import onnxruntime as ort

        config = voice_pipeline._load_settings()["vad"]
        wrapped = voice_pipeline.StreamingSileroVAD(config)
        asset = importlib.resources.files("faster_whisper").joinpath("assets", "silero_vad_v6.onnx")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        reference = ort.InferenceSession(str(asset), providers=["CPUExecutionProvider"], sess_options=options)
        reference_h = np.zeros((1, 1, 128), dtype=np.float32)
        reference_c = np.zeros((1, 1, 128), dtype=np.float32)
        reference_context = np.zeros(64, dtype=np.float32)
        frames = [
            (np.sin(np.arange(voice_pipeline.VAD_FRAME_SAMPLES, dtype=np.float32) / divisor) * 0.2)
            for divisor in (7.0, 11.0, 17.0)
        ]
        wrapped_scores = []
        reference_scores = []
        for frame in frames:
            wrapped_scores.append(wrapped.probability(frame))
            model_input = np.concatenate((reference_context, frame)).astype(np.float32, copy=False)
            probability, reference_h, reference_c = reference.run(
                None,
                {"input": model_input.reshape(1, -1), "h": reference_h, "c": reference_c},
            )
            reference_scores.append(float(np.asarray(probability).reshape(-1)[0]))
            reference_context = frame[-64:].copy()
        np.testing.assert_allclose(wrapped_scores, reference_scores, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(wrapped.context, frames[-1][-64:])
        np.testing.assert_array_equal(wrapped.h, reference_h)
        np.testing.assert_array_equal(wrapped.c, reference_c)

        fresh = voice_pipeline.StreamingSileroVAD(config)
        self.assertAlmostEqual(fresh.probability(frames[0]), wrapped_scores[0], places=7)


class WebSocketCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_releases_turn_blocked_by_full_queue_and_stalled_sender(self):
        import uvicorn
        import websockets
        from starlette.websockets import WebSocket

        class VAD:
            def __init__(self):
                self.frame = 0

            def accept(self, _samples):
                self.frame += 1
                if self.frame == 1:
                    return "start", None, 0.99
                if self.frame == 2:
                    return "end", np.zeros(16000, dtype=np.float32), 0.01
                return None, None, 0.01

        class Runtime:
            def __init__(self):
                self.status = {"ready": True, "stage": "ready"}
                self.metrics = []

            async def initialize(self):
                pass

            def status_snapshot(self):
                return {"ready": True, "stage": "ready"}

            def new_vad(self):
                return VAD()

            def transcribe(self, _audio, cancel_event=None):
                return "test question"

            def llm_stream(self, _text, _cancel, _history):
                for index in range(100):
                    yield f"Spoken sentence number {index} ends here. "

            def synthesize(self, _text, cancel_event=None):
                return b"\x00\x00" * 2400, 24000

        runtime = Runtime()
        original_runtime = app.runtime
        original_boot_task = app.boot_task
        original_active_session = app.active_session
        original_send_text = WebSocket.send_text
        original_sender = app.websocket_sender
        blocked_send = asyncio.Event()
        release_send = asyncio.Event()
        sender_tasks = []

        async def stalled_send(websocket, message):
            if not blocked_send.is_set():
                blocked_send.set()
                await release_send.wait()
            await original_send_text(websocket, message)

        async def observed_sender(session):
            sender_tasks.append(asyncio.current_task())
            await original_sender(session)

        app.runtime = runtime
        app.active_session = None
        app.boot_task = None
        app.websocket_sender = observed_sender
        WebSocket.send_text = stalled_send
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app.app, log_level="critical", lifespan="on"))
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        client = None
        replacement = None
        stalled_session = None
        old_turn_tasks = ()
        try:
            async with asyncio.timeout(3):
                while not server.started:
                    await asyncio.sleep(0.01)
            client = await websockets.connect(f"ws://127.0.0.1:{port}/ws/voice", close_timeout=0.1)
            await client.send(json.dumps({"type": "start", "sample_rate": 16000}))
            await client.send(b"\x00\x00" * 512)
            await client.send(b"\x00\x00" * 512)
            async with asyncio.timeout(3):
                while app.active_session is None or not app.active_session.outgoing.full():
                    await asyncio.sleep(0.005)
            stalled_session = app.active_session
            old_session_id = stalled_session.session_id
            old_turn_id = stalled_session.current_turn.turn_id
            async with asyncio.timeout(1):
                while not stalled_session.outgoing._putters:
                    await asyncio.sleep(0.005)
            old_turn_tasks = tuple(stalled_session.tasks)
            self.assertTrue(blocked_send.is_set(), "the WebSocket sender should be stalled")
            await client.close()
            async with asyncio.timeout(1):
                while app.active_session is not None:
                    await asyncio.sleep(0.005)
            self.assertTrue(all(task.done() for task in old_turn_tasks), "old turn producers must be released")
            self.assertTrue(all(task.done() for task in sender_tasks), "old sender must be stopped")
            self.assertFalse(stalled_session.outgoing._putters, "no blocked queue producers may remain")
            release_send.set()
            replacement = await websockets.connect(f"ws://127.0.0.1:{port}/ws/voice", close_timeout=0.1)
            await replacement.send(json.dumps({"type": "start", "sample_rate": 16000}))
            session_started = json.loads(await asyncio.wait_for(replacement.recv(), timeout=1))
            self.assertEqual(session_started["type"], "session_started")
            self.assertNotEqual(session_started["session_id"], old_session_id)
            await replacement.send(b"\x00\x00" * 512)
            await replacement.send(b"\x00\x00" * 512)
            async with asyncio.timeout(1):
                while True:
                    event = json.loads(await replacement.recv())
                    if event.get("type") == "speech_started":
                        self.assertNotEqual(event["turn_id"], old_turn_id)
                        break
        finally:
            release_send.set()
            WebSocket.send_text = original_send_text
            app.websocket_sender = original_sender
            if client is not None:
                try:
                    await asyncio.wait_for(client.close(), timeout=0.5)
                except Exception:
                    pass
            if replacement is not None:
                try:
                    await asyncio.wait_for(replacement.close(), timeout=0.5)
                except Exception:
                    pass
            if app.active_session is not None:
                app.active_session.stopped = True
                for turn in app.active_session.turns.values():
                    turn.cancel_event.set()
                for task in tuple(app.active_session.tasks):
                    task.cancel()
            server.should_exit = True
            try:
                await asyncio.wait_for(server_task, timeout=3)
            except Exception:
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            app.runtime = original_runtime
            app.active_session = original_active_session
            app.boot_task = original_boot_task


class WebSocketConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_interruption_recovers_after_turn_error(self):
        import uvicorn
        import websockets

        class VAD:
            def __init__(self):
                self.index = 0

            def accept(self, _samples):
                index = self.index
                self.index += 1
                if index % 2 == 0:
                    return "start", None, 0.99
                return "end", np.zeros(16000, dtype=np.float32), 0.01

        class Runtime:
            def __init__(self, mode):
                self.mode = mode
                self.status = {"ready": True, "stage": "ready", "mode": mode}
                self.config = {"audio": {}, "speech": {}}
                self.metrics = []
                self.transcription_count = 0

            async def initialize(self):
                pass

            def status_snapshot(self):
                return {"ready": True, "stage": "ready", "mode": self.mode}

            def new_vad(self):
                return VAD()

            def transcribe(self, _audio, cancel_event=None):
                self.transcription_count += 1
                return f"question {self.transcription_count}"

            def llm_stream(self, text, _cancel, _history):
                if text == "question 1":
                    yield "The first part is already audible."
                    raise RuntimeError("synthetic failure after partial audio")
                yield "Recovered answer."

            def synthesize(self, _text, cancel_event=None):
                return b"\x01\x00" * 960, 24000

        async def receive_event(client, expected):
            async with asyncio.timeout(3):
                while True:
                    message = await client.recv()
                    if isinstance(message, bytes):
                        turn_id = message[4:20].hex()
                        sequence = int.from_bytes(message[20:24], "big")
                        await client.send(json.dumps({"type": "playback_started", "turn_id": turn_id}))
                        await client.send(json.dumps({
                            "type": "playback_queue", "turn_id": turn_id,
                            "sequence": sequence, "consumed_ms": 0, "buffered_ms": 40,
                        }))
                        continue
                    event = json.loads(message)
                    if event.get("type") == expected:
                        return event

        for mode in ("local", "remote"):
            with self.subTest(mode=mode):
                runtime = Runtime(mode)
                original_runtime, original_boot, original_active = app.runtime, app.boot_task, app.active_session
                app.runtime, app.boot_task, app.active_session = runtime, None, None
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                listener.setblocking(False)
                server = uvicorn.Server(uvicorn.Config(app.app, log_level="critical", lifespan="on"))
                server_task = asyncio.create_task(server.serve(sockets=[listener]))
                client = None
                try:
                    async with asyncio.timeout(3):
                        while not server.started:
                            await asyncio.sleep(0.01)
                    client = await websockets.connect(
                        f"ws://127.0.0.1:{listener.getsockname()[1]}/ws/voice", close_timeout=0.1
                    )
                    await client.send(json.dumps({
                        "type": "start", "sample_rate": 16000, "interrupt_enabled": False,
                    }))
                    await receive_event(client, "session_started")

                    # A failure after partial synthesis must finish queued audio and release its server slot.
                    await client.send(b"\x00\x00" * 512)
                    await client.send(b"\x00\x00" * 512)
                    await receive_event(client, "audio_chunk")
                    failed = await receive_event(client, "error")
                    self.assertEqual(failed["code"], "RuntimeError")
                    failed_done = await receive_event(client, "turn_failed")
                    self.assertEqual(app.active_session.current_turn.turn_id, failed_done["turn_id"],
                                     "the failed turn must remain active until its queued audio completes")
                    await client.send(json.dumps({
                        "type": "playback_finished", "turn_id": failed_done["turn_id"],
                    }))
                    await receive_event(client, "listening")
                    self.assertIsNone(app.active_session.current_turn)

                    # With interruption still disabled, the next onset is a new turn, not ignored noise.
                    await client.send(b"\x00\x00" * 512)
                    next_turn = await receive_event(client, "speech_started")
                    self.assertNotEqual(next_turn["turn_id"], failed_done["turn_id"])
                    await client.send(b"\x00\x00" * 512)
                    transcript = await receive_event(client, "transcript")
                    self.assertEqual(transcript["text"], "question 2")
                    await receive_event(client, "turn_complete")
                    recovered_turn = transcript["turn_id"]
                    await client.send(json.dumps({
                        "type": "playback_finished", "turn_id": recovered_turn,
                    }))
                    await receive_event(client, "listening")
                    self.assertEqual(runtime.transcription_count, 2)
                finally:
                    if client is not None:
                        try:
                            await asyncio.wait_for(client.close(), timeout=0.5)
                        except Exception:
                            pass
                    if app.active_session is not None:
                        app.active_session.stopped = True
                        for turn in app.active_session.turns.values():
                            turn.cancel_event.set()
                        for task in tuple(app.active_session.tasks):
                            task.cancel()
                    server.should_exit = True
                    try:
                        await asyncio.wait_for(server_task, timeout=3)
                    except Exception:
                        server_task.cancel()
                        await asyncio.gather(server_task, return_exceptions=True)
                    app.runtime, app.boot_task, app.active_session = original_runtime, original_boot, original_active

    async def test_initially_disabled_interruption_is_acknowledged_and_protects_active_reply(self):
        import uvicorn
        import websockets

        class VAD:
            def __init__(self):
                self.index = 0

            def accept(self, _samples):
                index = self.index
                self.index += 1
                if index % 2 == 0:
                    return "start", None, 0.99
                return "end", np.zeros(16000, dtype=np.float32), 0.01

        class Runtime:
            def __init__(self, mode):
                self.mode = mode
                self.status = {"ready": True, "stage": "ready", "mode": mode}
                self.config = {"audio": {"max_buffered_audio_ms": 2000}, "speech": {}}
                self.metrics = []
                self.transcription_count = 0

            async def initialize(self):
                pass

            def status_snapshot(self):
                return {"ready": True, "stage": "ready", "mode": self.mode}

            def new_vad(self):
                return VAD()

            def transcribe(self, _audio, cancel_event=None):
                self.transcription_count += 1
                return f"question {self.transcription_count}"

            def llm_stream(self, text, _cancel, _history):
                yield f"The answer to {text} is in progress."

            def synthesize(self, _text, cancel_event=None):
                return b"\x01\x00" * 960, 24000

        async def receive_event(client, expected):
            async with asyncio.timeout(3):
                while True:
                    message = await client.recv()
                    if isinstance(message, bytes):
                        turn_hex = message[4:20].hex()
                        sequence = int.from_bytes(message[20:24], "big")
                        await client.send(json.dumps({
                            "type": "playback_started", "turn_id": turn_hex,
                        }))
                        await client.send(json.dumps({
                            "type": "playback_queue", "turn_id": turn_hex,
                            "sequence": sequence, "consumed_ms": 0, "buffered_ms": 40,
                        }))
                        continue
                    event = json.loads(message)
                    if event.get("type") == expected or (
                        isinstance(expected, set) and event.get("type") in expected
                    ):
                        return event

        for mode in ("local", "remote"):
            with self.subTest(mode=mode):
                runtime = Runtime(mode)
                original_runtime, original_boot, original_active = app.runtime, app.boot_task, app.active_session
                app.runtime, app.boot_task, app.active_session = runtime, None, None
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                listener.setblocking(False)
                server = uvicorn.Server(uvicorn.Config(app.app, log_level="critical", lifespan="on"))
                server_task = asyncio.create_task(server.serve(sockets=[listener]))
                client = None
                try:
                    async with asyncio.timeout(3):
                        while not server.started:
                            await asyncio.sleep(0.01)
                    client = await websockets.connect(
                        f"ws://127.0.0.1:{listener.getsockname()[1]}/ws/voice", close_timeout=0.1
                    )
                    await client.send(json.dumps({
                        "type": "start", "sample_rate": 16000, "interrupt_enabled": False,
                    }))
                    started = await receive_event(client, "session_started")

                    await client.send(b"\x00\x00" * 512)
                    await client.send(b"\x00\x00" * 512)
                    first = await receive_event(client, "turn_complete")
                    session = app.active_session
                    active_turn = session.turns[first["turn_id"]]
                    async with asyncio.timeout(1):
                        while not active_turn.playback_started:
                            await asyncio.sleep(0.005)

                    await client.send(b"\x00\x00" * 512)
                    speech_result = await receive_event(client, {"speech_ignored", "interrupt"})
                    self.assertEqual(speech_result["type"], "speech_ignored",
                                     "the initial unchecked selection must protect the active reply")
                    await client.send(b"\x00\x00" * 512)
                    self.assertIs(session.current_turn, active_turn)
                    self.assertFalse(active_turn.cancel_event.is_set())
                    self.assertEqual(runtime.transcription_count, 1)
                    self.assertIs(started.get("interrupt_enabled"), False,
                                  "session acknowledgement must apply the initial unchecked selection")
                finally:
                    if client is not None:
                        try:
                            await asyncio.wait_for(client.close(), timeout=0.5)
                        except Exception:
                            pass
                    if app.active_session is not None:
                        app.active_session.stopped = True
                        for turn in app.active_session.turns.values():
                            turn.cancel_event.set()
                    server.should_exit = True
                    try:
                        await asyncio.wait_for(server_task, timeout=3)
                    except Exception:
                        server_task.cancel()
                        await asyncio.gather(server_task, return_exceptions=True)
                    app.runtime, app.boot_task, app.active_session = original_runtime, original_boot, original_active

    async def test_interrupt_toggle_preserves_active_reply_when_off_and_interrupts_when_on(self):
        import uvicorn
        import websockets

        class VAD:
            def __init__(self):
                self.index = 0

            def accept(self, _samples):
                index = self.index
                self.index += 1
                if index % 2 == 0:
                    return "start", None, 0.99
                return "end", np.zeros(16000, dtype=np.float32), 0.01

        class Runtime:
            def __init__(self, mode):
                self.mode = mode
                self.status = {"ready": True, "stage": "ready", "mode": mode}
                self.config = {"audio": {"max_buffered_audio_ms": 2000}, "speech": {}}
                self.metrics = []
                self.transcription_count = 0
                self.synthesize_count = 0
                self.first_delta_blocked = threading.Event()
                self.release_first_delta = threading.Event()

            async def initialize(self):
                pass

            def status_snapshot(self):
                return {"ready": True, "stage": "ready", "mode": self.mode}

            def new_vad(self):
                return VAD()

            def transcribe(self, _audio, cancel_event=None):
                self.transcription_count += 1
                return f"question {self.transcription_count}"

            def llm_stream(self, text, _cancel, _history):
                if text == "question 1":
                    yield "The first part is already audible."
                    self.first_delta_blocked.set()
                    self.release_first_delta.wait()
                    yield "The rest of the answer continues after the noisy speech."
                else:
                    yield f"The complete answer to {text}."

            def synthesize(self, _text, cancel_event=None):
                self.synthesize_count += 1
                return b"\x01\x00" * 960, 24000

        async def receive_event(client, expected):
            audio_packets = []

            async def acknowledge_audio(message):
                audio_packets.append(message)
                turn_hex = message[4:20].hex()
                sequence = int.from_bytes(message[20:24], "big")
                await client.send(json.dumps({
                    "type": "playback_started", "turn_id": turn_hex,
                }))
                await client.send(json.dumps({
                    "type": "playback_queue", "turn_id": turn_hex,
                    "sequence": sequence, "consumed_ms": 0, "buffered_ms": 40,
                }))

            async with asyncio.timeout(3):
                while True:
                    message = await client.recv()
                    if isinstance(message, bytes):
                        await acknowledge_audio(message)
                        continue
                    event = json.loads(message)
                    observed_events.append(event)
                    if event.get("type") == "audio_chunk" and event.get("type") == expected:
                        audio_message = await client.recv()
                        if not isinstance(audio_message, bytes):
                            raise AssertionError("audio_chunk event must be followed by an audio packet")
                        await acknowledge_audio(audio_message)
                    if event.get("type") == expected or (
                        isinstance(expected, set) and event.get("type") in expected
                    ):
                        return event, audio_packets

        for mode in ("local", "remote"):
            with self.subTest(mode=mode):
                observed_events = []
                runtime = Runtime(mode)
                original_runtime, original_boot, original_active = app.runtime, app.boot_task, app.active_session
                app.runtime, app.boot_task, app.active_session = runtime, None, None
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                listener.setblocking(False)
                server = uvicorn.Server(uvicorn.Config(app.app, log_level="critical", lifespan="on"))
                server_task = asyncio.create_task(server.serve(sockets=[listener]))
                client = None
                try:
                    async with asyncio.timeout(3):
                        while not server.started:
                            await asyncio.sleep(0.01)
                    client = await websockets.connect(
                        f"ws://127.0.0.1:{listener.getsockname()[1]}/ws/voice", close_timeout=0.1
                    )
                    await client.send(json.dumps({
                        "type": "start", "sample_rate": 16000,
                    }))
                    started, _ = await receive_event(client, "session_started")

                    # Let the first response begin playing while its LLM stream remains open.
                    await client.send(b"\x00\x00" * 512)
                    await client.send(b"\x00\x00" * 512)
                    _, initial_audio = await receive_event(client, "audio_chunk")
                    self.assertTrue(runtime.first_delta_blocked.wait(1),
                                     "the assistant generation must remain in flight after first audio")
                    session = app.active_session
                    original_turn = session.current_turn
                    async with asyncio.timeout(1):
                        while not original_turn.playback_started:
                            await asyncio.sleep(0.005)
                    self.assertTrue(original_turn.playback_started)

                    # Change enabled -> disabled while the assistant is generating and playing.
                    await client.send(json.dumps({"type": "interrupt_setting", "enabled": False}))
                    await client.send(b"\x00\x00" * 512)
                    ignored, _ = await receive_event(client, {"speech_ignored", "interrupt"})
                    self.assertEqual(ignored["type"], "speech_ignored",
                                     "disabled interruption must not stop current playback")
                    self.assertIn({"type": "interrupt_setting", "enabled": False}, observed_events,
                                  "the active WebSocket must acknowledge the live enabled-to-disabled change")
                    await client.send(b"\x00\x00" * 512)
                    self.assertEqual(ignored["turn_id"], original_turn.turn_id)
                    self.assertIs(session.current_turn, original_turn)
                    self.assertFalse(original_turn.cancel_event.is_set())
                    self.assertEqual(runtime.transcription_count, 1,
                                     "ignored speech must not create a competing transcription")

                    # Resume the same response: its tail must arrive, synthesize and complete.
                    runtime.release_first_delta.set()
                    _, remaining_audio = await receive_event(client, "turn_complete")
                    self.assertGreaterEqual(runtime.synthesize_count, 2,
                                            "speech after the noise must still reach audio synthesis")
                    self.assertGreaterEqual(len(initial_audio) + len(remaining_audio), 2,
                                            "the browser transport must receive audio before and after the noise")
                    self.assertEqual(runtime.transcription_count, 1)
                    self.assertIs(started.get("interrupt_enabled"), True,
                                  "session start defaults to enabled when no preference is supplied")

                    # Let the browser report actual end-of-playback before the next utterance.
                    await client.send(json.dumps({
                        "type": "playback_finished", "turn_id": original_turn.turn_id,
                    }))
                    await receive_event(client, "listening")

                    # A new utterance after the reply is accepted normally.
                    await client.send(b"\x00\x00" * 512)
                    await client.send(b"\x00\x00" * 512)
                    second, _ = await receive_event(client, "turn_complete")
                    self.assertNotEqual(second["turn_id"], original_turn.turn_id)
                    self.assertEqual(runtime.transcription_count, 2)

                    # Turn barge-in back on while playback is active and prove replacement.
                    await client.send(json.dumps({"type": "interrupt_setting", "enabled": True}))
                    setting, _ = await receive_event(client, "interrupt_setting")
                    self.assertIs(setting.get("enabled"), True)
                    previous_turn = app.active_session.turns[second["turn_id"]]
                    await client.send(b"\x00\x00" * 512)
                    interruption, _ = await receive_event(client, "interrupt")
                    self.assertEqual(interruption["turn_id"], previous_turn.turn_id)
                    self.assertTrue(previous_turn.cancel_event.is_set())
                    await client.send(b"\x00\x00" * 512)
                    replacement, _ = await receive_event(client, "turn_complete")
                    self.assertNotEqual(replacement["turn_id"], previous_turn.turn_id)
                    self.assertEqual(runtime.transcription_count, 3)
                finally:
                    runtime.release_first_delta.set()
                    if client is not None:
                        try:
                            await asyncio.wait_for(client.close(), timeout=0.5)
                        except Exception:
                            pass
                    if app.active_session is not None:
                        app.active_session.stopped = True
                        for turn in app.active_session.turns.values():
                            turn.cancel_event.set()
                    server.should_exit = True
                    try:
                        await asyncio.wait_for(server_task, timeout=3)
                    except Exception:
                        server_task.cancel()
                        await asyncio.gather(server_task, return_exceptions=True)
                    app.runtime, app.boot_task, app.active_session = original_runtime, original_boot, original_active

    async def test_two_websocket_turns_pass_exact_prior_exchange_to_llm(self):
        import uvicorn
        import websockets

        class VAD:
            def __init__(self):
                self.index = 0

            def accept(self, _samples):
                sequence = self.index
                self.index += 1
                if sequence in (0, 2):
                    return "start", None, 0.99
                if sequence in (1, 3):
                    return "end", np.zeros(16000, dtype=np.float32), 0.01
                return None, None, 0.01

        class Runtime:
            def __init__(self):
                self.status = {"ready": True, "stage": "ready"}
                self.config = {"audio": {"max_buffered_audio_ms": 2000}, "speech": {}}
                self.metrics = []
                self.histories = []
                self.count = 0

            async def initialize(self):
                pass

            def status_snapshot(self):
                return {"ready": True, "stage": "ready"}

            def new_vad(self):
                return VAD()

            def transcribe(self, _audio, cancel_event=None):
                self.count += 1
                return f"question {self.count}"

            def llm_stream(self, _text, _cancel, history):
                self.histories.append(list(history))
                yield "The first answer." if not history else "The contextual follow-up."

            def synthesize(self, _text, cancel_event=None):
                return b"\x01\x00" * 480, 24000

        runtime = Runtime()
        original_runtime, original_boot, original_active = app.runtime, app.boot_task, app.active_session
        app.runtime, app.boot_task, app.active_session = runtime, None, None
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app.app, log_level="critical", lifespan="on"))
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        client = None
        try:
            async with asyncio.timeout(3):
                while not server.started:
                    await asyncio.sleep(0.01)
            client = await websockets.connect(f"ws://127.0.0.1:{port}/ws/voice", close_timeout=0.1)
            await client.send(json.dumps({"type": "start", "sample_rate": 16000}))
            while json.loads(await client.recv()).get("type") != "session_started":
                pass

            for expected_turn in range(1, 3):
                await client.send(b"\x00\x00" * 512)
                await client.send(b"\x00\x00" * 512)
                async with asyncio.timeout(3):
                    while True:
                        message = await client.recv()
                        if isinstance(message, bytes):
                            self.assertEqual(message[:4], b"ELVA")
                            sequence = int.from_bytes(message[20:24], "big")
                            turn_hex = message[4:20].hex()
                            await client.send(json.dumps({
                                "type": "playback_queue", "turn_id": turn_hex,
                                "sequence": sequence, "consumed_ms": 0, "buffered_ms": 20,
                            }))
                            continue
                        event = json.loads(message)
                        if event.get("type") == "turn_complete":
                            self.assertEqual(int(event["turn_id"] is not None), 1)
                            await client.send(json.dumps({
                                "type": "playback_finished", "turn_id": event["turn_id"],
                            }))
                            break
                self.assertEqual(runtime.count, expected_turn)

            self.assertEqual(runtime.histories[0], [])
            self.assertEqual(runtime.histories[1], [
                {"role": "user", "content": "question 1"},
                {"role": "assistant", "content": "The first answer."},
            ])
        finally:
            if client is not None:
                try:
                    await asyncio.wait_for(client.close(), timeout=0.5)
                except Exception:
                    pass
            if app.active_session is not None:
                app.active_session.stopped = True
                for turn in app.active_session.turns.values():
                    turn.cancel_event.set()
            server.should_exit = True
            try:
                await asyncio.wait_for(server_task, timeout=3)
            except Exception:
                server_task.cancel()
                await asyncio.gather(server_task, return_exceptions=True)
            app.runtime, app.boot_task, app.active_session = original_runtime, original_boot, original_active


class BrowserAudioBackpressureTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_waits_for_browser_audio_capacity_acknowledgement(self):
        import uvicorn
        import websockets

        class VAD:
            def __init__(self):
                self.frame = 0

            def accept(self, _samples):
                self.frame += 1
                if self.frame == 1:
                    return "start", None, 0.99
                if self.frame == 2:
                    return "end", np.zeros(16000, dtype=np.float32), 0.01
                return None, None, 0.01

        class Runtime:
            def __init__(self):
                self.status = {"ready": True, "stage": "ready"}
                self.metrics = []
                self.config = {"audio": {"max_buffered_audio_ms": 500, "packet_duration_ms": 200}}

            async def initialize(self):
                pass

            def status_snapshot(self):
                return {"ready": True, "stage": "ready", "audio": self.config["audio"]}

            def new_vad(self):
                return VAD()

            def transcribe(self, _audio, cancel_event=None):
                return "question"

            def llm_stream(self, _text, _cancel, _history):
                yield "The long response is split into packets."

            def synthesize(self, _text, cancel_event=None):
                return b"\x00\x00" * (24000 * 3), 24000

        runtime = Runtime()
        original_runtime, original_boot, original_active = app.runtime, app.boot_task, app.active_session
        app.runtime, app.boot_task, app.active_session = runtime, None, None
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app.app, log_level="critical", lifespan="on"))
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        client = None

        async def next_audio(timeout):
            pending_audio_event = False

            async def receive():
                nonlocal pending_audio_event
                while True:
                    message = await client.recv()
                    if isinstance(message, bytes):
                        if message.startswith(b"ELVA") or pending_audio_event:
                            return message
                    else:
                        event = json.loads(message)
                        if event.get("type") == "audio_chunk":
                            pending_audio_event = True

            return await asyncio.wait_for(receive(), timeout=timeout)

        playback_tail = 0.0
        last_ack_buffered = 0.0
        scheduled_ends = []
        audio_packets = []

        async def acknowledge_schedule(packet):
            nonlocal playback_tail, last_ack_buffered
            turn_id = packet[4:20].hex()
            sequence = int.from_bytes(packet[20:24], "big")
            rate = int.from_bytes(packet[24:28], "big")
            duration_ms = (len(packet) - 28) * 500 / rate
            now = time.monotonic()
            start_at = max(now + 0.015, playback_tail)
            playback_tail = start_at + duration_ms / 1000
            scheduled_ends.append(playback_tail)
            buffered_ms = max(0.0, (playback_tail - time.monotonic()) * 1000)
            consumed_ms = max(0.0, last_ack_buffered + duration_ms - buffered_ms)
            last_ack_buffered = buffered_ms
            await client.send(json.dumps({
                "type": "playback_queue", "turn_id": turn_id, "sequence": sequence,
                "consumed_ms": consumed_ms, "buffered_ms": buffered_ms,
            }))
            audio_packets.append(packet)

        async def acknowledge_completed_source():
            nonlocal last_ack_buffered
            if not scheduled_ends:
                return
            end_at = scheduled_ends.pop(0)
            await asyncio.sleep(max(0.0, end_at - time.monotonic()))
            buffered_ms = max(0.0, (playback_tail - time.monotonic()) * 1000)
            consumed_ms = max(0.0, last_ack_buffered - buffered_ms)
            last_ack_buffered = buffered_ms
            turn_id = app.active_session.current_turn.turn_id
            sequence = int.from_bytes(audio_packets[-1][20:24], "big")
            await client.send(json.dumps({
                "type": "playback_queue", "turn_id": turn_id, "sequence": sequence,
                "consumed_ms": consumed_ms, "buffered_ms": buffered_ms,
            }))

        try:
            async with asyncio.timeout(3):
                while not server.started:
                    await asyncio.sleep(0.01)
            client = await websockets.connect(f"ws://127.0.0.1:{port}/ws/voice", close_timeout=0.1)
            await client.send(json.dumps({"type": "start", "sample_rate": 16000}))
            while json.loads(await client.recv()).get("type") != "session_started":
                pass
            await client.send(b"\x00\x00" * 512)
            await client.send(b"\x00\x00" * 512)
            for _ in range(2):
                packet = await next_audio(2)
                await acknowledge_schedule(packet)
            with self.assertRaises(asyncio.TimeoutError):
                await next_audio(0.05)
            turn_id = app.active_session.current_turn.turn_id
            await client.send(json.dumps({
                "type": "playback_queue", "turn_id": turn_id,
                "sequence": 999, "consumed_ms": 0, "buffered_ms": 0,
            }))
            with self.assertRaises(asyncio.TimeoutError):
                await next_audio(0.05)
            while True:
                try:
                    packet = await next_audio(0.04)
                    await acknowledge_schedule(packet)
                except asyncio.TimeoutError:
                    await acknowledge_completed_source()
                    packet = await next_audio(1)
                    await acknowledge_schedule(packet)
                if len(audio_packets) >= 15:
                    break
            self.assertEqual(len(audio_packets), 15, "three seconds of TTS must be split into 200 ms packets")
            self.assertEqual(sum(len(packet) - 28 for packet in audio_packets), 24000 * 2 * 3)
            while True:
                message = await asyncio.wait_for(client.recv(), timeout=1)
                if isinstance(message, str) and json.loads(message).get("type") == "turn_complete":
                    break
            while scheduled_ends:
                await acknowledge_completed_source()
        finally:
            if client is not None:
                try:
                    await asyncio.wait_for(client.close(), timeout=0.5)
                except Exception:
                    pass
            if app.active_session is not None:
                app.active_session.stopped = True
                for turn in app.active_session.turns.values():
                    turn.cancel_event.set()
                for task in tuple(app.active_session.tasks):
                    task.cancel()
            server.should_exit = True
            try:
                await asyncio.wait_for(server_task, timeout=3)
            except Exception:
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            app.runtime, app.boot_task, app.active_session = original_runtime, original_boot, original_active


class PlaybackAcknowledgementTests(unittest.IsolatedAsyncioTestCase):
    async def test_acknowledgement_keeps_reserved_unsent_packet_duration(self):
        original_runtime = app.runtime
        app.runtime = SimpleNamespace(config={"audio": {"max_buffered_audio_ms": 2000}})
        session = app.VoiceSession("credit-test", websocket=None, vad=None)
        turn = voice_pipeline.Turn(turn_id())
        session.current_turn = turn
        session.remember_turn(turn)
        session.audio_sent_sequence[turn.turn_id] = 0
        session.audio_packet_durations[turn.turn_id] = {0: 800.0, 1: 800.0}
        session.browser_buffered_ms = 1600.0
        queued_packet = (
            b"ELVA" + bytes.fromhex(turn.turn_id) + (1).to_bytes(4, "big")
            + (24000).to_bytes(4, "big") + b"\x00\x00" * (24000 * 800 // 1000)
        )
        session.outgoing.put_nowait(app.Outbound(
            event={"type": "audio_chunk", "turn_id": turn.turn_id, "sequence": 1},
            audio=queued_packet,
            turn_id=turn.turn_id,
        ))

        try:
            await app.receive_control(session, json.dumps({
                "type": "playback_queue", "turn_id": turn.turn_id,
                "sequence": 0, "consumed_ms": 785.0, "buffered_ms": 815.0,
            }))
            self.assertEqual(session.audio_packet_durations[turn.turn_id], {1: 800.0})
            self.assertEqual(session.browser_buffered_ms, 1615.0,
                             "the earlier client snapshot must preserve its reserved unsent packet")
            self.assertEqual(session.outgoing.qsize(), 1,
                             "the packet whose duration remains reserved must still be queued for the stalled sender")
        finally:
            app.runtime = original_runtime


if __name__ == "__main__":
    unittest.main()
