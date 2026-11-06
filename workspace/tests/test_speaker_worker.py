"""Deterministic tests for fail-closed asynchronous speaker gating."""

from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app


class FakeSpeakerModel:
    def score(self, profiles, chunk):
        self.last_profiles = tuple(profiles)
        level = float(np.mean(chunk))
        score = 0.9 if level >= 0.45 else 0.6 if level >= 0.25 else 0.4
        return score, 2.0


class SpeakerWorkerTests(unittest.TestCase):
    def test_only_two_consecutive_target_chunks_reach_vad(self):
        async def run():
            observed = []
            runtime = SimpleNamespace(
                config={"speaker": {"start_threshold": 0.75, "release_threshold": 0.5,
                                    "start_chunks": 2, "max_frame_age_ms": 1000}},
                speaker_model=FakeSpeakerModel(),
            )
            session = app.VoiceSession("test", None, None, speaker_filter_enabled=True,
                                       speaker_profiles={"owner": {"enabled": True, "embedding": [1.0]}})

            async def record(_session, frame):
                observed.append(float(np.mean(frame)))

            original_runtime = app.runtime
            app.runtime = runtime
            with patch.object(app, "_process_audio_frame", side_effect=record):
                worker = asyncio.create_task(app.speaker_audio_worker(session))
                try:
                    now = time.monotonic()
                    for _ in range(10):
                        await session.speaker_queue.put((np.full(512, 0.5, dtype=np.float32), now))
                    for _ in range(5):
                        await session.speaker_queue.put((np.zeros(512, dtype=np.float32), time.monotonic()))
                    async with asyncio.timeout(2):
                        while len(observed) < 15:
                            await asyncio.sleep(0.01)
                    self.assertEqual(observed[:10], [0.5] * 10)
                    self.assertEqual(observed[10:], [0.0] * 5)
                finally:
                    session.stopped = True
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                    app.runtime = original_runtime

        asyncio.run(run())

    def test_stale_audio_is_discarded_without_reaching_vad(self):
        async def run():
            observed = []
            runtime = SimpleNamespace(
                config={"speaker": {"start_threshold": 0.75, "release_threshold": 0.5,
                                    "start_chunks": 2, "max_frame_age_ms": 1000}},
                speaker_model=FakeSpeakerModel(),
            )
            session = app.VoiceSession("test", None, None, speaker_filter_enabled=True,
                                       speaker_profiles={"owner": {"enabled": True, "embedding": [1.0]}})

            async def record(_session, frame):
                observed.append(frame)

            original_runtime = app.runtime
            app.runtime = runtime
            with patch.object(app, "_process_audio_frame", side_effect=record):
                worker = asyncio.create_task(app.speaker_audio_worker(session))
                try:
                    stale_at = time.monotonic() - 2
                    for _ in range(5):
                        await session.speaker_queue.put((np.full(512, 0.5, dtype=np.float32), stale_at))
                    async with asyncio.timeout(1):
                        while not session.speaker_metrics:
                            await asyncio.sleep(0.01)
                    self.assertEqual(session.speaker_metrics[-1], {"stale_drops": 1})
                    self.assertEqual(observed, [])
                finally:
                    session.stopped = True
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                    app.runtime = original_runtime

        asyncio.run(run())

    def test_release_threshold_keeps_active_speaker_and_rejects_background(self):
        async def run():
            observed = []
            runtime = SimpleNamespace(
                config={"speaker": {"start_threshold": 0.75, "release_threshold": 0.5,
                                    "start_chunks": 2, "max_frame_age_ms": 1000}},
                speaker_model=FakeSpeakerModel(),
            )
            session = app.VoiceSession("test", None, None, speaker_filter_enabled=True,
                                       speaker_profiles={"owner": {"enabled": True, "embedding": [1.0]}})

            async def record(_session, frame):
                observed.append(float(np.mean(frame)))

            original_runtime = app.runtime
            app.runtime = runtime
            with patch.object(app, "_process_audio_frame", side_effect=record):
                worker = asyncio.create_task(app.speaker_audio_worker(session))
                try:
                    now = time.monotonic()
                    for value in (0.5, 0.5, 0.3, 0.1, 0.5, 0.5):
                        for _ in range(5):
                            await session.speaker_queue.put((np.full(512, value, dtype=np.float32), now))
                    async with asyncio.timeout(2):
                        while len(observed) < 30:
                            await asyncio.sleep(0.01)
                    expected = [0.5] * 10 + [0.3] * 5 + [0.0] * 5 + [0.5] * 10
                    self.assertTrue(np.allclose(observed, expected))
                    self.assertTrue(session.speaker_active)
                finally:
                    session.stopped = True
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                    app.runtime = original_runtime

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
