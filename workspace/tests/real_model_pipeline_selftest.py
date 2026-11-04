"""Run the local STT, LLM, and TTS models end to end without saving audio or text."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_pipeline import SAMPLE_RATE, VoiceRuntime


def _run_llm(runtime: VoiceRuntime, prompt: str, history: list[dict[str, str]]) -> str:
    return "".join(runtime.llm_stream(prompt, threading.Event(), history)).strip()


def _resample_to_stt(pcm: bytes, source_rate: int) -> np.ndarray:
    source = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    length = round(source.size * SAMPLE_RATE / source_rate)
    old_positions = np.arange(source.size, dtype=np.float64)
    new_positions = np.arange(length, dtype=np.float64) * source_rate / SAMPLE_RATE
    return np.interp(new_positions, old_positions, source).astype(np.float32)


async def main() -> None:
    runtime = VoiceRuntime()
    load_started = time.monotonic()
    await runtime.initialize()
    if not runtime.status["ready"]:
        raise RuntimeError(f"Local models did not become ready: {runtime.status['error']}")
    print(f"Models initialized in {(time.monotonic() - load_started) * 1000:.0f} ms")

    source_text = "Как играть на Джаггернауте против Фантомной Ассасинки?"
    pcm, source_rate = await asyncio.to_thread(runtime.synthesize, source_text)
    speech = _resample_to_stt(pcm, source_rate)
    transcript = await asyncio.to_thread(runtime.transcribe, speech)
    if not transcript:
        raise RuntimeError("Local Whisper returned an empty transcript for the Silero voice sample")
    print(f"STT passed ({speech.size / SAMPLE_RATE:.2f}s synthesized input)")

    generation_started = time.monotonic()
    reply = await asyncio.to_thread(_run_llm, runtime, transcript, [])
    if not reply:
        raise RuntimeError("Local language model returned an empty response")
    print(f"LLM passed ({(time.monotonic() - generation_started) * 1000:.0f} ms)")

    output, output_rate = await asyncio.to_thread(runtime.synthesize, reply)
    if not output or output_rate != runtime.config["tts"]["sample_rate"]:
        raise RuntimeError("Local TTS returned empty audio or an unexpected sample rate")
    print(f"TTS passed ({len(output) / (2 * output_rate):.2f}s at {output_rate} Hz)")

    # Measure cooperative cancellation at model iterator/segment boundaries on the loaded CPU models.
    for stage in ("stt", "llm"):
        residuals = []
        for _ in range(3):
            cancel_event = threading.Event()
            started = threading.Event()

            def invoke():
                started.set()
                if stage == "stt":
                    runtime.transcribe(speech, cancel_event)
                    return
                iterator = runtime.llm_stream(transcript, cancel_event, [])
                try:
                    next(iterator, None)
                    cancel_at = time.monotonic()
                    cancel_event.set()
                    for _item in iterator:
                        pass
                    residuals.append((time.monotonic() - cancel_at) * 1000)
                finally:
                    iterator.close()

            worker = asyncio.create_task(asyncio.to_thread(invoke))
            await asyncio.to_thread(started.wait)
            if stage == "stt":
                await asyncio.sleep(0.02)
                cancel_at = time.monotonic()
                cancel_event.set()
            await worker
            if stage == "stt":
                residuals.append((time.monotonic() - cancel_at) * 1000)
        print(f"{stage.upper()} cancellation residual ms: "
              + ", ".join(f"{value:.1f}" for value in residuals))

    prompt_cancel = threading.Event()
    prompt_started = threading.Event()
    first_token = threading.Event()

    def cancel_during_first_token():
        iterator = runtime.llm_stream(transcript, prompt_cancel, [])
        try:
            prompt_started.set()
            next(iterator, None)
            first_token.set()
            for _item in iterator:
                pass
        finally:
            iterator.close()

    prompt_worker = asyncio.create_task(asyncio.to_thread(cancel_during_first_token))
    await asyncio.to_thread(prompt_started.wait)
    await asyncio.sleep(0.02)
    cancellation_during_prompt = not first_token.is_set()
    prompt_cancel_at = time.monotonic()
    prompt_cancel.set()
    await prompt_worker
    print("LLM first-token/prompt residual ms: "
          f"{(time.monotonic() - prompt_cancel_at) * 1000:.1f}; "
          f"cancelled_before_first_token={cancellation_during_prompt}")

    tts_cancel = threading.Event()
    tts_started = threading.Event()
    tts_residual = []
    tts_cancel_at = [None]

    def synthesize_for_cancellation():
        tts_started.set()
        runtime.synthesize(reply, tts_cancel)
        if tts_cancel.is_set() and tts_cancel_at[0] is not None:
            tts_residual.append((time.monotonic() - tts_cancel_at[0]) * 1000)

    tts_worker = asyncio.create_task(asyncio.to_thread(synthesize_for_cancellation))
    await asyncio.to_thread(tts_started.wait)
    await asyncio.sleep(0.02)
    tts_cancel_at[0] = time.monotonic()
    tts_cancel.set()
    await tts_worker
    print("TTS cancellation residual ms: "
          + (f"{tts_residual[0]:.1f}" if tts_residual else "completed before cancellation"))

    print("PASS: real local STT -> LLM -> b_lez TTS pipeline completed; no files were written")


if __name__ == "__main__":
    asyncio.run(main())
