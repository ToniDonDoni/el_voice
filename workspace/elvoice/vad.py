"""Streaming Silero VAD (ONNX Runtime) and the endpointing segmenter.

The ONNX graph from Silero VAD v5.1.2 is a streaming model: it consumes exactly
512 samples of 16 kHz audio per call plus a recurrent state tensor, and returns a
speech probability for that frame plus the next state. State is kept for the
whole browser session so silence and speech transitions are classified in
context; only an explicit session reset clears it.

The segmenter turns frame probabilities into speech onsets and end-of-utterance
events. It keeps a configurable pre-roll buffer so the first phoneme is not
lost, and it can keep running while the assistant is generating or playing
audio so a new speech onset can trigger barge-in.
"""

from __future__ import annotations

import math
from collections import deque

FRAME_SAMPLES = 512  # 32 ms at 16 kHz, required by the Silero v5 streaming model
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = FRAME_SAMPLES * 1000 // 16000


class SileroVad:
    """Thin ONNX Runtime wrapper with explicit streaming state."""

    def __init__(self, model_path: str, sample_rate: int = 16000, threads: int = 1):
        import numpy as np

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - container check covers this
            raise RuntimeError("onnxruntime is required for streaming VAD") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._np = np
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._sample_rate = np.array(int(sample_rate), dtype=np.int64)
        self.reset_states()

    def reset_states(self) -> None:
        # Official v5.1.2 OnnxWrapper semantics: recurrent state plus the last
        # 64 samples of audio as context for the next 512-sample chunk.
        self._state = self._np.zeros((2, 1, 128), dtype=self._np.float32)
        self._context = self._np.zeros((1, 64), dtype=self._np.float32)

    def speech_probability(self, frame: bytes) -> float:
        """Return the speech probability for one 512-sample little-endian frame."""
        np = self._np
        samples = np.frombuffer(frame, dtype="<i2")
        if samples.size != FRAME_SAMPLES:
            raise ValueError(f"VAD frame must contain {FRAME_SAMPLES} samples")
        audio = samples.astype(np.float32) / 32768.0
        concatenated = np.concatenate([self._context, audio.reshape(1, -1)], axis=1)
        output, state = self._session.run(
            None,
            {
                "input": concatenated,
                "state": self._state,
                "sr": self._sample_rate,
            },
        )
        self._state = state
        self._context = concatenated[:, -64:]
        return float(output.reshape(-1)[0])


class SpeechSegmenter:
    """Turns frame probabilities into onset / end-of-utterance events."""

    def __init__(self, audio_cfg: dict, vad_cfg: dict):
        self._threshold = float(vad_cfg["threshold"])
        self._neg_threshold = float(vad_cfg["neg_threshold"])
        self._min_speech_ms = float(vad_cfg["min_speech_ms"])
        self._end_silence_ms = float(vad_cfg["end_silence_ms"])
        pre_roll_frames = max(1, math.ceil(float(audio_cfg["pre_roll_ms"]) / FRAME_MS))
        self._max_bytes = int(float(audio_cfg["max_utterance_ms"]) / FRAME_MS) * FRAME_BYTES
        self._pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        self._turn_seq = 0
        self._reset_utterance_state()

    def _reset_utterance_state(self) -> None:
        self._in_speech = False
        self._buffer = bytearray()
        self._speech_ms = 0
        self._silence_ms = 0

    def reset(self) -> None:
        self._pre_roll.clear()
        self._turn_seq = 0
        self._reset_utterance_state()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, frame: bytes, probability: float) -> list[dict]:
        """Consume one frame and return zero or more events.

        Events are ``{"type": "onset", "turn_id": int}`` when speech starts and
        ``{"type": "end", "turn_id": int, "pcm": bytes, "speech_ms": float}``
        once the utterance ends. The end event always carries the full audio of
        the utterance including the pre-roll.
        """
        events: list[dict] = []
        if not self._in_speech:
            self._pre_roll.append(frame)
            if probability >= self._threshold:
                self._turn_seq += 1
                self._in_speech = True
                self._buffer = bytearray(b"".join(self._pre_roll))
                self._pre_roll.clear()
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                events.append({"type": "onset", "turn_id": self._turn_seq})
            return events

        self._buffer.extend(frame)
        if probability >= self._neg_threshold:
            self._silence_ms = 0.0
            self._speech_ms += FRAME_MS
        else:
            self._silence_ms += FRAME_MS

        if self._silence_ms >= self._end_silence_ms or len(self._buffer) >= self._max_bytes:
            events.append({
                "type": "end",
                "turn_id": self._turn_seq,
                "pcm": bytes(self._buffer),
                "speech_ms": self._speech_ms,
            })
            self._reset_utterance_state()
        return events
