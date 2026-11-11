"""CPU gate regression tests with a fake VAD (no model downloads)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workspace"))
from speaker_activity_gate import SpeakerActivityGate


class FakeVAD:
    def __init__(self, events):
        self.events = iter(events)
        self.calls = 0

    def accept(self, _frame):
        self.calls += 1
        return next(self.events, None), None, 0.0


def frames(value=0.0):
    return [np.full(512, value, dtype=np.float32) for _ in range(5)]


class SpeakerActivityGateTests(unittest.TestCase):
    def test_silence_skips_speaker_inference(self):
        vad = FakeVAD([None] * 50)
        gate = SpeakerActivityGate(vad)
        assert all(not gate.observe(frames())[0] for _ in range(10))
        self.assertEqual(vad.calls, 50)
        self.assertFalse(gate.active)

    def test_onset_preserves_bounded_preroll(self):
        vad = FakeVAD([None] * 10 + ["start"])
        gate = SpeakerActivityGate(vad)
        gate.observe(frames(1))
        gate.observe(frames(2))
        should_score, started, ended, before = gate.observe(frames(3))
        self.assertTrue(should_score)
        self.assertTrue(started)
        self.assertFalse(ended)
        self.assertEqual(len(before), 2)
        self.assertEqual(len(before[0]), 5)
        self.assertEqual(float(before[0][0][0]), 1)
        self.assertEqual(float(before[1][0][0]), 2)
        self.assertTrue(gate.observe(frames(4))[0])

    def test_endpoint_disables_scoring_until_next_onset(self):
        vad = FakeVAD(["start"] + [None] * 9 + ["end"] + [None] * 9 + ["start"])
        gate = SpeakerActivityGate(vad)
        self.assertTrue(gate.observe(frames(1))[0])
        should_score, _, ended, _ = gate.observe(frames())
        self.assertTrue(should_score)
        self.assertFalse(ended)
        should_score, _, ended, _ = gate.observe(frames())
        self.assertTrue(should_score)
        self.assertTrue(ended)
        self.assertFalse(gate.active)
        self.assertFalse(gate.observe(frames())[0])
        should_score, started, _, _ = gate.observe(frames(1))
        self.assertTrue(should_score)
        self.assertTrue(started)

    def test_false_speech_resets_gate(self):
        gate = SpeakerActivityGate(FakeVAD(["start", "empty"]))
        score, started, ended, _ = gate.observe(frames())
        self.assertTrue(score)
        self.assertTrue(started)
        self.assertTrue(ended)
        self.assertFalse(gate.active)
        self.assertFalse(gate.observe(frames())[0])


if __name__ == "__main__":
    unittest.main()
