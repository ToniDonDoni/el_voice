"""Focused review regressions for the local voice response path.

SDDTDD SPEC: ../../tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-REVIEW.md

Run in the voice image with:
    python -m unittest discover -s /workspace/tests -p 'test_review_*.py'
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import app
import voice_pipeline


class StubRuntime:
    def __init__(self):
        self.synthesized = []

    def transcribe(self, _audio):
        return "Give me a short answer."

    def llm_stream(self, _text, _cancel, _history):
        yield "**Build** Mjollnir ✨ and read https://example.com/guide now."

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
    def test_invalid_sample_rate_prevents_pipeline_readiness(self):
        invalid = {
            "tts": {"model": "tts/v4_cyrillic.pt", "speaker": "b_lez", "sample_rate": 12345, "cpu_threads": 4},
            "stt": {"model": "stt/small", "language": "ru", "cpu_threads": 4},
            "llm": {"model": "llm/model.gguf", "context_tokens": 3072, "cpu_threads": 4, "batch_tokens": 256, "max_response_tokens": 192},
            "vad": {"sample_rate": 16000, "frame_samples": 512, "threshold": 0.55, "negative_threshold": 0.35,
                    "start_frames": 3, "end_silence_ms": 640, "min_speech_ms": 240, "pre_roll_ms": 320,
                    "max_utterance_seconds": 45},
        }
        with patch.object(voice_pipeline, "_load_settings", return_value=invalid):
            runtime = voice_pipeline.VoiceRuntime()
        with patch.object(runtime, "_load_models", return_value=None):
            asyncio.run(runtime.initialize())
        self.assertFalse(runtime.status["ready"])
        self.assertEqual(runtime.status["stage"], "error")
        self.assertIsNotNone(runtime.status["error"])


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
                self.session.stopped = True

        session = app.VoiceSession("session-packet", None, None)
        socket = Socket(session)
        session.websocket = socket
        turn = voice_pipeline.Turn("00112233445566778899aabbccddeeff")
        session.current_turn = turn
        session.remember_turn(turn)
        await session.outgoing.put(app.Outbound(
            event={"type": "audio_chunk", "turn_id": turn.turn_id, "sample_rate": 24000},
            audio=b"\x00\x00" * 32,
            turn_id=turn.turn_id,
        ))
        sender = asyncio.create_task(app.websocket_sender(session))
        await asyncio.wait_for(session.outgoing.join(), timeout=1)
        await asyncio.wait_for(sender, timeout=1)
        self.assertEqual(len(socket.binary), 1)
        self.assertTrue(socket.binary[0].startswith(b"ELVA"))
        self.assertIn(bytes.fromhex(turn.turn_id), socket.binary[0])


if __name__ == "__main__":
    unittest.main()
