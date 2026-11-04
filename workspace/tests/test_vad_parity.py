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
    """The upstream silero-vad==5.1.2 OnnxWrapper (not a hand-written copy)."""

    def __init__(self, path: Path):
        import torch
        try:
            from silero_vad.utils_vad import OnnxWrapper
        except OSError as exc:  # torchaudio may be unavailable in a CPU-only venv
            raise RuntimeError(f"official silero-vad import failed: {exc}") from exc
        self._torch = torch
        self.model = OnnxWrapper(str(path), force_onnx_cpu=True)
        self.model.reset_states()

    def reset_states(self) -> None:
        self.model.reset_states()

    def __call__(self, frame: bytes) -> float:
        import numpy as np
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(samples).unsqueeze(0)
        return float(self.model(tensor, 16000).item())


def main() -> int:
    from elvoice.vad import FRAME_BYTES, SileroVad

    our = SileroVad(str(MODEL), sample_rate=16000, threads=1)
    reference = OfficialWrapperReference(MODEL)
    # Prefer real recorded speech whenever it is available; fall back to a
    # deterministic synthetic stream so the parity check always runs.
    frames = []
    wav = Path(os.environ.get("EL_VOICE_MODELS", "/models")).parent / "data" / "tts-test.wav"
    if wav.is_file():
        import wave
        with wave.open(str(wav), "rb") as handle:
            raw = handle.readframes(handle.getnframes())
        source = __import__("numpy").frombuffer(raw, dtype="<i2")
        if source.size >= 16000:
            # crude 24 kHz -> 16 kHz conversion for a deterministic test input
            import numpy as np
            positions = np.linspace(0, source.size - 1, int(source.size * 16000 / 24000))
            converted = np.interp(positions, np.arange(source.size), source).astype("<i2")
            data = converted.tobytes()
            frames = [data[i:i + FRAME_BYTES] for i in range(0, len(data) - FRAME_BYTES, FRAME_BYTES)][:120]
            print(f"using {frames.__len__()} real WAV frames from {wav}")
    if not frames:
        rng = __import__("random").Random(11)
        frames = [bytes(rng.randrange(0, 256) for _ in range(FRAME_BYTES)) for _ in range(40)]

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
