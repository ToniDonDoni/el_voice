"""Security checks for local speaker controls over the voice WebSocket."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
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
    async def test_progress_clock_reports_pressed_state_when_audio_frames_stop(self):
        previous_runtime = app.runtime
        app.runtime = SimpleNamespace(config={"speaker": {"enrollment_max_seconds": 45}})
        session = app.VoiceSession("enrollment-progress-test", None, None)
        session.speaker_enrollment_kind = "owner"
        session.speaker_enrollment_started = time.monotonic()
        task = asyncio.create_task(app.speaker_enrollment_progress(session))
        try:
            first = await asyncio.wait_for(session.outgoing.get(), timeout=0.2)
            second = await asyncio.wait_for(session.outgoing.get(), timeout=0.8)
            self.assertEqual(first.event["type"], "speaker_enrollment")
            self.assertEqual(first.event["recorded_seconds"], 0)
            self.assertIsNone(first.event["audio_frame_gap_ms"])
            self.assertEqual(second.event["state"], "capturing")
            self.assertEqual(second.event["recorded_seconds"], 0)
        finally:
            session.speaker_enrollment_kind = None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            app.runtime = previous_runtime

    async def test_ten_seconds_of_detected_speech_are_saved_on_release(self):
        class EnrollmentVAD:
            threshold = 0.55

            def probability(self, _frame):
                self.frames = getattr(self, "frames", 0) + 1
                return 0.99 if self.frames % 2 else 0.1

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
                config={"speaker": {"enrollment_min_recording_seconds": 10,
                                     "enrollment_min_detected_speech_seconds": 2,
                                     "enrollment_max_seconds": 45}},
                speaker_model=model,
                profile_store=store,
                new_vad=EnrollmentVAD,
            )
            previous_runtime = app.runtime
            app.runtime = runtime
            session = app.VoiceSession("enrollment-test", None, None)
            active_turn = app.Turn("active-turn")
            active_turn.playback_started = True
            session.current_turn = active_turn
            session.turns[active_turn.turn_id] = active_turn
            session.speaker_filter_enabled = True
            try:
                await app.receive_control(session, json.dumps({
                    "type": "speaker_enrollment_begin", "kind": "owner", "consent": True,
                }))
                self.assertIsNone(session.current_turn, "enrollment must preempt an active detected-speech turn")
                self.assertTrue(active_turn.cancel_event.is_set())
                self.assertTrue(session.speaker_filter_enabled, "an existing profile must not block re-enrollment")
                frames = np.full(512, 1200, dtype="<i2").tobytes()
                for _ in range(320):  # 320 x 32 ms = 10.24 seconds at 16 kHz.
                    await app.receive_audio(session, frames)

                self.assertEqual(session.speaker_enrollment_received_samples, 320 * 512)
                self.assertEqual(session.speaker_enrollment_speech_samples, 160 * 512)
                await app.receive_control(session, json.dumps({"type": "speaker_enrollment_end"}))

                events = [item.event for item in list(session.outgoing._queue)]
                self.assertEqual(model.enrolled_samples, 320 * 512)
                self.assertEqual(model.warmed_profile, "owner")
                self.assertTrue(session.speaker_filter_enabled)
                self.assertIn("owner", store.load())
                self.assertTrue(any(event.get("type") == "interrupt" for event in events))
                self.assertEqual(events[-1]["state"], "saved")
                self.assertTrue(events[-1]["filter_enabled"])
            finally:
                if session.speaker_worker:
                    session.speaker_worker.cancel()
                    await asyncio.gather(session.speaker_worker, return_exceptions=True)
                app.runtime = previous_runtime


if __name__ == "__main__":
    unittest.main()
