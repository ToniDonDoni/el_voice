"""Security checks for local speaker controls over the voice WebSocket."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from starlette.datastructures import Headers

import app
from speaker_profiles import ProfileStore


class SpeakerWebSocketOriginTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_and_unexpected_origins_are_rejected_before_accept(self):
        for origin in (None, "https://attacker.example"):
            with self.subTest(origin=origin):
                closed = {}

                async def close(*, code, reason):
                    closed.update(code=code, reason=reason)

                headers = Headers({} if origin is None else {"origin": origin})
                websocket = SimpleNamespace(headers=headers, close=close)
                with patch.dict("os.environ", {"EL_VOICE_ALLOWED_ORIGINS": "http://localhost:8765"}):
                    await app.voice_socket(websocket)

                self.assertEqual(closed.get("code"), 1008)
                self.assertEqual(closed.get("reason"), "Unexpected WebSocket origin")


class SpeakerEnrollmentFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_ten_seconds_of_detected_speech_are_saved_on_release(self):
        class EnrollmentVAD:
            threshold = 0.55

            def probability(self, _frame):
                return 0.99

        class SpeakerModel:
            def __init__(self):
                self.enrolled_samples = 0
                self.warmed_profile = None

            def enroll(self, audio):
                self.enrolled_samples = audio.size
                embedding = np.zeros(192, dtype=np.float32)
                embedding[0] = 1.0
                return embedding

            def warmup(self, profile_id, _embedding):
                self.warmed_profile = profile_id

        with tempfile.TemporaryDirectory() as directory:
            model = SpeakerModel()
            store = ProfileStore(Path(directory) / "speaker-profiles.json")
            runtime = SimpleNamespace(
                config={"speaker": {"enrollment_max_seconds": 45}},
                speaker_model=model,
                profile_store=store,
                new_vad=EnrollmentVAD,
            )
            previous_runtime = app.runtime
            app.runtime = runtime
            session = app.VoiceSession("enrollment-test", None, None)
            try:
                await app.receive_control(session, json.dumps({
                    "type": "speaker_enrollment_begin", "kind": "owner", "consent": True,
                }))
                frames = np.full(512, 1200, dtype="<i2").tobytes()
                for _ in range(320):  # 320 x 32 ms = 10.24 seconds at 16 kHz.
                    await app.receive_audio(session, frames)

                self.assertGreaterEqual(session.speaker_enrollment_speech_samples, 160_000)
                await app.receive_control(session, json.dumps({"type": "speaker_enrollment_end"}))

                events = [item.event for item in list(session.outgoing._queue)]
                self.assertEqual(model.enrolled_samples, 320 * 512)
                self.assertEqual(model.warmed_profile, "owner")
                self.assertTrue(session.speaker_filter_enabled)
                self.assertIn("owner", store.load())
                self.assertEqual(events[-1]["state"], "saved")
                self.assertTrue(events[-1]["filter_enabled"])
            finally:
                if session.speaker_worker:
                    session.speaker_worker.cancel()
                    await asyncio.gather(session.speaker_worker, return_exceptions=True)
                app.runtime = previous_runtime


if __name__ == "__main__":
    unittest.main()
