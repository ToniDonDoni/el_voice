"""SDDTDD SPEC: tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-Review.md (R4).

The streaming VAD must reproduce the official Silero VAD v5.1.2 OnnxWrapper
semantics: it prepends the last 64 samples of the previous 512-sample chunk as
context (576-sample ONNX input), keeps recurrent state, and resets both on
reset_states(). This test compares our adapter against a reference
implementation of the official wrapper on the same synthetic frames.

Run:  EL_VOICE_MODELS=/path/to/models python tests/test_vad_parity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = Path(os.environ.get("EL_VOICE_MODELS", "/models")) / "vad/silero_vad.onnx"
FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class OfficialWrapperReference:
    """Minimal copy of the v5.1.2 OnnxWrapper streaming path for 16 kHz."""

    def __init__(self, path: Path):
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._np = np
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.reset_states()

    def reset_states(self) -> None:
        self._state = self._np.zeros((2, 1, 128), dtype=self._np.float32)
        self._context = self._np.zeros((1, 0), dtype=self._np.float32)

    def __call__(self, frame: bytes) -> float:
        np = self._np
        x = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        x = x.reshape(1, -1)
        context_size = 64
        if self._context.size == 0:
            self._context = np.zeros((1, context_size), dtype=np.float32)
        concatenated = np.concatenate([self._context, x], axis=1)
        output, state = self.session.run(
            None,
            {
                "input": concatenated,
                "state": self._state,
                "sr": np.array(16000, dtype=np.int64),
            },
        )
        self._state = state
        self._context = concatenated[:, -context_size:]
        return float(output.reshape(-1)[0])


def main() -> int:
    from elvoice.vad import FRAME_BYTES, SileroVad

    our = SileroVad(str(MODEL), sample_rate=16000, threads=1)
    reference = OfficialWrapperReference(MODEL)
    rng = __import__("random").Random(11)
    frames = [
        bytes(rng.randrange(0, 256) for _ in range(FRAME_BYTES))
        for _ in range(40)
    ]

    differences = []
    for index, frame in enumerate(frames):
        ours = our.speech_probability(frame)
        expected = reference(frame)
        differences.append(abs(ours - expected))
        if index < 3:
            check(f"frame {index} probability matches the official wrapper",
                  abs(ours - expected) < 1e-5, f"ours={ours:.6f} official={expected:.6f}")

    check("all frames match the official wrapper within tolerance",
          max(differences) < 1e-5, f"max_diff={max(differences):.8f}")

    # reset_states must clear both recurrent state and audio context.
    our.reset_states()
    reference.reset_states()
    _ = our.speech_probability(frames[0])
    _ = reference(frames[0])
    first_after_reset = our.speech_probability(frames[1])
    expected_after_reset = reference(frames[1])
    check("reset_states produces the same stream restart",
          abs(first_after_reset - expected_after_reset) < 1e-5,
          f"ours={first_after_reset:.6f} official={expected_after_reset:.6f}")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
