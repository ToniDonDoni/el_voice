"""Cheap Silero activity gate for optional enrolled-speaker inference.

This gate decides when Nomo-PVAD should run; it never authorizes a speaker.
The normal turn-ending VAD stays separate to preserve its state.
"""

from collections import deque


class SpeakerActivityGate:
    """Track VAD onset/end and retain a bounded pre-roll for verified speech."""

    def __init__(self, vad, pre_roll_batches: int = 2):
        if pre_roll_batches < 0:
            raise ValueError("pre_roll_batches must be nonnegative")
        self.vad = vad
        self.active = False
        self.history = deque(maxlen=max(1, pre_roll_batches))
        self.pre_roll_batches = pre_roll_batches

    def observe(self, frames):
        """Return (should_score, started, ended, preceding_batches)."""
        should_score = self.active
        started = False
        ended = False
        for frame in frames:
            event, _audio, _score = self.vad.accept(frame)
            if event == "start":
                self.active = True
                started = True
                should_score = True
            elif event in {"end", "empty"}:
                if self.active:
                    ended = True
                self.active = False
        pre_roll = list(self.history)[-self.pre_roll_batches:] if started and self.pre_roll_batches else []
        self.history.append(tuple(frame.copy() for frame in frames))
        return should_score or self.active, started, ended, pre_roll
