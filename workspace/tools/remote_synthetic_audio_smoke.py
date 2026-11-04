"""Exercise synthetic Silero speech through Nginx, Whisper, Remote LLM, and TTS.

Run inside the voice container with:
    docker compose -f compose.yaml -f compose.remote.yaml exec voice \
      python /workspace/tools/remote_synthetic_audio_smoke.py

This verifies packet flow, not physical microphone capture or audible playback.
It stores no audio or transcript files and never prints the provider key.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_pipeline import SAMPLE_RATE, VoiceRuntime


def resample(pcm: bytes, source_rate: int) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    target_length = round(samples.size * SAMPLE_RATE / source_rate)
    old_positions = np.arange(samples.size, dtype=np.float64)
    new_positions = np.arange(target_length, dtype=np.float64) * source_rate / SAMPLE_RATE
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


async def main() -> None:
    with urllib.request.urlopen("http://proxy:8080/api/status", timeout=5) as response:
        status = json.load(response)
    llm_status = status.get("llm") or {}
    if (status.get("mode") != "remote" or not status.get("ready")
            or llm_status.get("mode") != "remote" or not llm_status.get("ready")):
        raise RuntimeError("The browser-facing service is not ready in Remote Voice mode")

    runtime = VoiceRuntime()
    await runtime.initialize()
    if not runtime.status.get("ready"):
        raise RuntimeError("Local voice models are not ready")

    question = "Как играть на Аксе в Доте?"
    pcm, sample_rate = await asyncio.to_thread(runtime.synthesize, question)
    speech = resample(pcm, sample_rate)
    pcm16 = np.clip(speech * 32767, -32768, 32767).astype("<i2").tobytes()
    frame_bytes = 512 * 2
    silence = bytes(frame_bytes * 32)
    first_audio_at = None
    speech_ended_at = None
    transcript = ""
    packet_count = 0
    audio_bytes = 0
    errors = []

    async with websockets.connect("ws://proxy:8080/ws/voice", close_timeout=2) as socket:
        await socket.send(json.dumps({"type": "start", "sample_rate": SAMPLE_RATE,
                                      "interrupt_enabled": True}))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=15))
            if event.get("type") == "session_started":
                break
            if event.get("type") in {"not_ready", "busy", "error"}:
                raise RuntimeError(f"Session could not start: {event.get('type')}")

        for offset in range(0, len(pcm16), frame_bytes):
            frame = pcm16[offset:offset + frame_bytes]
            if len(frame) == frame_bytes:
                await socket.send(frame)
        for offset in range(0, len(silence), frame_bytes):
            await socket.send(silence[offset:offset + frame_bytes])

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            message = await asyncio.wait_for(socket.recv(), timeout=20)
            if isinstance(message, bytes):
                if not message.startswith(b"ELVA") or len(message) < 28:
                    continue
                packet_count += 1
                audio_bytes += len(message) - 28
                turn_id = message[4:20].hex()
                sequence = int.from_bytes(message[20:24], "big")
                rate = int.from_bytes(message[24:28], "big")
                duration_ms = (len(message) - 28) * 500 / rate
                if first_audio_at is None:
                    first_audio_at = time.monotonic()
                    await socket.send(json.dumps({"type": "playback_started", "turn_id": turn_id}))
                await socket.send(json.dumps({
                    "type": "playback_queue", "turn_id": turn_id, "sequence": sequence,
                    "consumed_ms": duration_ms, "buffered_ms": 0,
                }))
                continue

            event = json.loads(message)
            if event.get("type") == "speech_ended":
                speech_ended_at = time.monotonic()
            elif event.get("type") == "transcript":
                transcript = str(event.get("text", ""))
            elif event.get("type") in {"error", "turn_failed"}:
                errors.append(event.get("code", event.get("message", event.get("type"))))
            elif event.get("type") == "turn_complete":
                await socket.send(json.dumps({"type": "playback_finished", "turn_id": event["turn_id"]}))
                break
        else:
            raise TimeoutError("Timed out waiting for the synthetic voice turn")

    if not transcript or not packet_count or errors:
        raise RuntimeError(f"Voice route failed: transcript={bool(transcript)} packets={packet_count} errors={errors}")
    latency = (first_audio_at - speech_ended_at) * 1000 if first_audio_at and speech_ended_at else None
    print(f"transcript_chars={len(transcript)} audio_packets={packet_count} "
          f"audio_pcm_bytes={audio_bytes} first_packet_ms={latency:.0f}" if latency is not None
          else f"transcript_chars={len(transcript)} audio_packets={packet_count} audio_pcm_bytes={audio_bytes}")
    print("PASS: synthetic audio reached local Whisper, Remote LLM, and local TTS through Nginx")


if __name__ == "__main__":
    asyncio.run(main())
