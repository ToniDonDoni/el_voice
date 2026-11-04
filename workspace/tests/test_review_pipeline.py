"""Focused review regressions for the local voice response path.

SDDTDD SPEC: ../../tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-REVIEW.md

Run in the voice image with:
    python -m unittest discover -s /workspace/tests -p 'test_review_*.py'
"""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import socket
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import app
import voice_pipeline


class StubRuntime:
    def __init__(self):
        self.synthesized = []
        self.histories = []
        self.generation_finished = threading.Event()

    def transcribe(self, _audio):
        return "Give me a short answer."

    def llm_stream(self, _text, _cancel, history):
        self.histories.append(list(history))
        yield "**Build** Mjollnir ✨ and read https://example.com/guide now."
        self.generation_finished.set()

    def synthesize(self, text):
        self.synthesized.append(text)
        return b"\x00\x00" * 480, 24000


class SpeakableResponseTests(unittest.TestCase):
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
            turn_id="turn-review-test",
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


class StartupValidationTests(unittest.TestCase):
    def test_configuration_errors_are_exposed_by_status_endpoint(self):
        invalid = {
            "tts": {"model": "tts/v4_cyrillic.pt", "speaker": "b_lez", "sample_rate": 24000, "cpu_threads": 4},
            "stt": {"model": "stt/small", "language": "ru", "cpu_threads": 4},
            "llm": {"model": "llm/model.gguf", "context_tokens": 3072, "cpu_threads": 4, "batch_tokens": 256, "max_response_tokens": 192},
            "vad": {"sample_rate": 16000, "frame_samples": 512, "threshold": 0.55, "negative_threshold": 0.35,
                    "start_frames": 3, "end_silence_ms": 640, "min_speech_ms": 240, "pre_roll_ms": 320,
                    "max_utterance_seconds": 45},
        }
        cases = [
            ("tts.sample_rate", lambda cfg: cfg["tts"].update(sample_rate=12345)),
            ("stt.cpu_threads", lambda cfg: cfg["stt"].update(cpu_threads=0)),
            ("vad.threshold", lambda cfg: cfg["vad"].update(threshold=1.5)),
            ("vad.sample_rate", lambda cfg: cfg["vad"].update(sample_rate=8000)),
        ]

        async def fetch_status(config):
            with patch.object(voice_pipeline, "_load_settings", return_value=config):
                runtime = voice_pipeline.VoiceRuntime()
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

        session = app.VoiceSession("session-packet", None, None)
        socket = Socket(session)
        session.websocket = socket
        turn = voice_pipeline.Turn("00112233445566778899aabbccddeeff")
        session.current_turn = turn
        session.remember_turn(turn)
        pcm = b"\x01\x02" * 32
        for sequence in range(2):
            await session.outgoing.put(app.Outbound(
                event={"type": "audio_chunk", "turn_id": turn.turn_id, "sequence": sequence,
                       "sample_rate": 24000},
                audio=pcm,
                turn_id=turn.turn_id,
            ))
        sender = asyncio.create_task(app.websocket_sender(session))
        await asyncio.wait_for(session.outgoing.join(), timeout=1)
        await asyncio.wait_for(sender, timeout=1)
        self.assertEqual(len(socket.binary), 2)
        for sequence, packet in enumerate(socket.binary):
            self.assertEqual(packet[:4], b"ELVA")
            self.assertEqual(packet[4:20], bytes.fromhex(turn.turn_id))
            self.assertEqual(int.from_bytes(packet[20:24], "big"), sequence)
            self.assertEqual(int.from_bytes(packet[24:28], "big"), 24000)
            self.assertEqual(packet[28:], pcm)


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
                turn = SimpleNamespace(turn_id=f"turn-{index}", cancel_event=threading.Event())
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
        turn = SimpleNamespace(turn_id="streaming-turn", cancel_event=threading.Event())
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
        turn = SimpleNamespace(turn_id="token-turn", cancel_event=threading.Event())
        session.current_turn = turn
        stream = app._llm_items(session, turn, "question")
        try:
            self.assertEqual(await anext(stream), "token")
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
        release_first = threading.Event()
        counter_lock = threading.Lock()

        class SlowRuntime(StubRuntime):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.active = 0
                self.max_active = 0

            def transcribe(self, _audio):
                with counter_lock:
                    self.calls += 1
                    call = self.calls
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    if call == 1:
                        started.set()
                        release_first.wait(timeout=2)
                    return f"question {call}"
                finally:
                    with counter_lock:
                        self.active -= 1

        runtime = SlowRuntime()
        app.runtime = runtime
        session = self.make_session("cancelled-worker-regression")
        obsolete = SimpleNamespace(turn_id="obsolete", cancel_event=threading.Event())
        replacement = SimpleNamespace(turn_id="replacement", cancel_event=threading.Event())
        session.current_turn = obsolete
        obsolete_task = asyncio.create_task(app.process_turn(session, obsolete, b"first"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1), "first STT call should start")
            obsolete.cancel_event.set()
            session.current_turn = replacement
            replacement_task = asyncio.create_task(app.process_turn(session, replacement, b"second"))
            await asyncio.sleep(0.05)
            self.assertEqual(runtime.calls, 1, "replacement work must wait without creating another CPU worker")
            release_first.set()
            await asyncio.wait_for(asyncio.gather(obsolete_task, replacement_task), timeout=2)
            self.assertEqual(runtime.max_active, 1, "obsolete STT workers must not accumulate")
            self.assertTrue(any(item.event and item.event.get("type") == "turn_complete"
                                for item in list(session.outgoing._queue)),
                            "the replacement turn must complete after residual inference exits")
        finally:
            release_first.set()
            for task in (obsolete_task, locals().get("replacement_task")):
                if task and not task.done():
                    task.cancel()
            app.runtime = original_runtime

    async def test_obsolete_llm_and_tts_workers_do_not_accumulate(self):
        original_runtime = app.runtime

        for stage in ("llm", "tts"):
            with self.subTest(stage=stage):
                started = threading.Event()
                release_first = threading.Event()
                counter_lock = threading.Lock()

                class SlowRuntime(StubRuntime):
                    def __init__(self):
                        super().__init__()
                        self.calls = 0
                        self.active = 0
                        self.max_active = 0

                    def begin_slow_call(self):
                        with counter_lock:
                            self.calls += 1
                            call = self.calls
                            self.active += 1
                            self.max_active = max(self.max_active, self.active)
                        if call == 1:
                            started.set()
                            release_first.wait(timeout=2)
                        return call

                    def llm_stream(self, _text, _cancel, _history):
                        call = self.begin_slow_call() if stage == "llm" else 0
                        if stage == "llm":
                            try:
                                yield "This is the first complete sentence."
                            finally:
                                with counter_lock:
                                    self.active -= 1
                        else:
                            yield "This is the first complete sentence."

                    def synthesize(self, text):
                        self.synthesized.append(text)
                        if stage == "tts":
                            call = self.begin_slow_call()
                            try:
                                return b"\x00\x00" * 480, 24000
                            finally:
                                with counter_lock:
                                    self.active -= 1
                        return b"\x00\x00" * 480, 24000

                runtime = SlowRuntime()
                app.runtime = runtime
                session = self.make_session(f"cancelled-{stage}")
                old = SimpleNamespace(turn_id=f"old-{stage}", cancel_event=threading.Event())
                new = SimpleNamespace(turn_id=f"new-{stage}", cancel_event=threading.Event())
                session.current_turn = old
                old_task = asyncio.create_task(app.process_turn(session, old, b"old"))
                new_task = None
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 1), f"{stage} call should start")
                    old.cancel_event.set()
                    session.current_turn = new
                    new_task = asyncio.create_task(app.process_turn(session, new, b"new"))
                    await asyncio.sleep(0.05)
                    self.assertEqual(runtime.calls, 1, f"replacement {stage} must await old worker without spawning another")
                    release_first.set()
                    await asyncio.wait_for(asyncio.gather(old_task, new_task), timeout=2)
                    self.assertEqual(runtime.max_active, 1)
                finally:
                    release_first.set()
                    for task in (old_task, new_task):
                        if task and not task.done():
                            task.cancel()
                    app.runtime = original_runtime


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

            def transcribe(self, _audio):
                return "test question"

            def llm_stream(self, _text, _cancel, _history):
                for index in range(100):
                    yield f"Spoken sentence number {index} ends here. "

            def synthesize(self, _text):
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


if __name__ == "__main__":
    unittest.main()
