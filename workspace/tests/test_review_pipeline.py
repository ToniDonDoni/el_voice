"""Focused review regressions for the local voice response path.

Run in the voice image with:
    python -m unittest discover -s /workspace/tests -p 'test_review_*.py'
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace

import app


class StubRuntime:
    def transcribe(self, _audio):
        return "Give me a short answer."

    def llm_stream(self, _text, _cancel, _history):
        yield "**Build** Mjollnir ✨ and read https://example.com/guide now."

    def synthesize(self, _text):
        return b"\x00\x00" * 480, 24000


class SpeakableResponseTests(unittest.TestCase):
    def test_streamed_tts_chunks_drop_markup_urls_and_emoji(self):
        original_runtime = app.runtime
        app.runtime = StubRuntime()
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


if __name__ == "__main__":
    unittest.main()
