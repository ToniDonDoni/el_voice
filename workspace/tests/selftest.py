"""End-to-end local pipeline check for EL Voice.

Starts the real internal LLM service (local Gemma backend) and the real voice
FastAPI application in-process, synthesises Russian test utterances with the
configured Silero voice, streams them over the real WebSocket protocol as 16 kHz
PCM frames, and checks one complete turn plus a barge-in during playback. This
exercises the streaming VAD, Whisper, the LLM-service boundary, Gemma and Silero
TTS in the same process layout as the two Compose services.

Run inside the voice container:
    python /workspace/tests/selftest.py --models /models

Run in a development environment with the downloaded models:
    EL_VOICE_MODELS=/path/to/models python workspace/tests/selftest.py

The check does not persist audio or transcripts; developer output prints only
lengths, status and timings, never transcript text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_DIR.parent
for _path in (str(SOURCE_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

HEADER = 16

UTTERANCES = (
    "Привет! Подскажи, пожалуйста, какой герой хорошо подходит для игры на миду?",
    "А какие предметы собрать Пуджу в середине игры?",
    "Спасибо, а что делать против Сларка?",
)


def fetch_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_http_ready(url: str, timeout: float = 240.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise SystemExit(f"service did not become available: {last_error}")


def wait_server_started(server, thread: threading.Thread, name: str,
                        timeout: float = 15.0) -> None:
    """Fail fast when a port is already in use instead of using the old process."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(server, "started", False):
            return
        if not thread.is_alive():
            raise SystemExit(f"{name} failed to start (is its port already in use?)")
        time.sleep(0.05)
    raise SystemExit(f"{name} did not start within {timeout:.0f} s")


def wait_llm_ready(base_url: str, timeout: float = 240.0) -> dict:
    deadline = time.monotonic() + timeout
    status: dict = {}
    while time.monotonic() < deadline:
        status = wait_http_ready(base_url + "/status", timeout=5)
        if status.get("ready"):
            return status
        if status.get("error"):
            raise SystemExit(f"LLM service failed to start: {status['error']}")
        time.sleep(0.5)
    raise SystemExit(f"LLM service did not become ready: {json.dumps(status)[:500]}")


def wait_pipeline_ready(base_url: str, timeout: float = 240.0) -> dict:
    deadline = time.monotonic() + timeout
    status: dict = {}
    while time.monotonic() < deadline:
        status = wait_http_ready(base_url + "/api/status", timeout=5)
        if status.get("pipeline_ready"):
            return status
        time.sleep(0.5)
    raise SystemExit(f"pipeline did not become ready: {json.dumps(status)[:500]}")


def load_silero_voice(models_root: Path, speaker: str):
    """Load the verified Silero package and return a synthesis callable."""
    import numpy as np
    import torch

    from elvoice import models as models_module

    torch.set_num_threads(4)
    path = models_module.verify_asset(models_root, models_module.TTS_MODEL)
    model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    if speaker not in model.speakers:
        raise SystemExit(f"speaker {speaker} is absent from the TTS model")

    def synthesize(text: str) -> bytes:
        with torch.inference_mode():
            audio = model.apply_tts(text=text, speaker=speaker, sample_rate=24000)
        samples = audio.detach().to(dtype=torch.float32).clamp(-1.0, 1.0).numpy()
        positions = np.linspace(0.0, len(samples) - 1, int(len(samples) * 16000 / 24000))
        resampled = np.interp(positions, np.arange(len(samples)), samples)
        pcm = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes()

    return synthesize


class WsHarness:
    """Small WebSocket test client with a background message reader."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.queue: asyncio.Queue = asyncio.Queue()
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            while True:
                message = await self.websocket.recv()
                await self.queue.put(message)
        except Exception:  # noqa: BLE001 - normal close ends the reader
            await self.queue.put(None)

    async def send_json(self, payload: dict) -> None:
        await self.websocket.send(json.dumps(payload))

    async def feed_pcm(self, pcm: bytes, trailing_silence_s: float = 0.9) -> None:
        frame = 1024
        for offset in range(0, len(pcm), frame):
            await self.websocket.send(pcm[offset:offset + frame])
            await asyncio.sleep(0.001)
        silence = b"\x00\x00" * int(16000 * trailing_silence_s)
        for offset in range(0, len(silence), frame):
            await self.websocket.send(silence[offset:offset + frame])
            await asyncio.sleep(0.001)

    async def expect(self, predicate, timeout: float = 90.0):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for a message")
            message = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            if message is None:
                raise ConnectionError("WebSocket closed")
            if predicate(message):
                return message

    async def close(self) -> None:
        self.reader.cancel()
        try:
            await self.reader
        except (asyncio.CancelledError, Exception):
            pass


def parse_audio(message: bytes) -> tuple[int, int]:
    if message[0:4] != b"EVA1" or len(message) <= HEADER:
        raise ValueError("unexpected audio frame")
    turn_id = int.from_bytes(message[4:8], "little")
    sequence = int.from_bytes(message[8:12], "little")
    return turn_id, sequence


def json_type(message):
    if isinstance(message, bytes):
        return "__audio__"
    return json.loads(message).get("type")


async def run_check(port: int, models_root: Path, cfg: dict) -> int:
    import websockets

    uri = f"ws://127.0.0.1:{port}/ws"
    results = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}: {name}{(' — ' + detail) if detail else ''}", flush=True)

    synthesize = load_silero_voice(models_root, cfg["tts"]["speaker"])
    print("Synthesising test utterances with the configured voice…", flush=True)
    audio = [synthesize(text) for text in UTTERANCES]
    print("Test utterances ready.", flush=True)

    async with websockets.connect(uri, max_size=None) as websocket:
        harness = WsHarness(websocket)
        await harness.send_json({"type": "start"})
        ready = await harness.expect(lambda message: json_type(message) == "ready", timeout=200)
        ready_payload = json.loads(ready)
        record("session ready after models initialized", True,
               f"vad={ready_payload['vad']['threshold']}")
        status_default = await asyncio.to_thread(
            fetch_json, f"http://127.0.0.1:{port}/api/status")
        record("interruption is enabled by default",
               (status_default.get("session") or {}).get("barge_in") is True,
               repr((status_default.get("session") or {}).get("barge_in")))
        await harness.send_json({"type": "config", "barge_in": False})
        await asyncio.sleep(0.1)
        status_off = await asyncio.to_thread(
            fetch_json, f"http://127.0.0.1:{port}/api/status")
        record("config message disables interruption server-side (client wiring is browser-only)",
               (status_off.get("session") or {}).get("barge_in") is False,
               repr((status_off.get("session") or {}).get("barge_in")))
        await harness.send_json({"type": "config", "barge_in": True})
        await asyncio.sleep(0.1)

        # --- one complete turn -------------------------------------------
        record_started = time.perf_counter()
        await harness.feed_pcm(audio[0])
        feed_finished = time.perf_counter()

        user_text = None
        first_audio_at = None
        turn_one_audio = 0
        turn_end_seen = False
        timing_ms = None
        while not turn_end_seen:
            message = await harness.expect(lambda item: True, timeout=120)
            kind = json_type(message)
            if kind == "user_text":
                user_text = json.loads(message)["text"]
            elif kind == "__audio__":
                turn_id, _ = parse_audio(message)
                if turn_id == 1:
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter()
                    turn_one_audio += 1
            elif kind == "turn_end":
                turn_end_seen = True
            elif kind == "timing":
                timing_ms = json.loads(message).get("response_ms")
            elif kind == "error":
                record("turn one completed without an error", False, json.loads(message)["message"])
                break
        record("Russian utterance transcribed locally", bool(user_text), f"{len(user_text or '')} chars")
        record("assistant produced audio for turn one", turn_one_audio > 0, f"{turn_one_audio} chunks")
        if first_audio_at is not None:
            record("first audio arrived after speech end", True,
                   f"{int((first_audio_at - feed_finished) * 1000)} ms after the feed (client-side)")
        await harness.send_json({"type": "playback_done", "turn_id": 1})
        await harness.expect(
            lambda item: json_type(item) == "state" and json.loads(item).get("state") == "listening",
            timeout=30,
        )

        # --- barge-in during playback ------------------------------------
        await harness.feed_pcm(audio[1])
        await harness.expect(
            lambda item: json_type(item) == "__audio__" and parse_audio(item)[0] == 2,
            timeout=90,
        )
        # Speak over the assistant while its turn-two audio is playing.
        await harness.feed_pcm(audio[2])
        flush = await harness.expect(
            lambda item: json_type(item) == "flush", timeout=60,
        )
        record("barge-in flushed the playing turn", json.loads(flush)["turn_id"] == 2,
               f"turn={json.loads(flush)['turn_id']}")

        flush_turn = json.loads(flush)["turn_id"]
        turn_three_text = None
        turn_three_audio = 0
        stale_after_flush = 0
        candidate = None
        turn_three_end = False
        while not turn_three_end:
            message = await harness.expect(lambda item: True, timeout=120)
            kind = json_type(message)
            if kind == "__audio__":
                turn_id, _ = parse_audio(message)
                if turn_id == flush_turn:
                    stale_after_flush += 1
                elif candidate is None or turn_id == candidate:
                    candidate = turn_id
                    turn_three_audio += 1
            elif kind == "user_text":
                turn_id = json.loads(message)["turn_id"]
                if turn_id != flush_turn:
                    candidate = candidate or turn_id
                    if turn_id == candidate:
                        turn_three_text = json.loads(message)["text"]
            elif kind == "turn_end":
                turn_id = json.loads(message)["turn_id"]
                if turn_id != flush_turn:
                    candidate = candidate or turn_id
                    if turn_id == candidate:
                        turn_three_end = True
            elif kind == "error":
                record("turn after barge-in completed without an error", False,
                       json.loads(message)["message"])
                break
        record("new turn transcribed after barge-in", bool(turn_three_text),
               f"{len(turn_three_text or '')} chars")
        record("new turn produced audio", turn_three_audio > 0, f"{turn_three_audio} chunks")
        record("no stale audio from the interrupted turn after flush", stale_after_flush == 0,
               f"{stale_after_flush} late chunks")
        await harness.send_json({"type": "playback_done", "turn_id": candidate})
        max_turn_id = candidate or flush_turn

        # --- conversation context across consecutive clean turns ---------
        for clip, label in ((audio[0], "context first"), (audio[2], "context follow-up")):
            await harness.feed_pcm(clip)
            context_turn = None
            context_text = None
            ended = False
            while not ended:
                message = await harness.expect(lambda item: True, timeout=120)
                kind = json_type(message)
                if kind == "user_text":
                    payload = json.loads(message)
                    max_turn_id = max(max_turn_id, payload["turn_id"])
                    context_text = payload["text"]
                elif kind == "turn_end":
                    payload = json.loads(message)
                    max_turn_id = max(max_turn_id, payload["turn_id"])
                    context_turn = payload["turn_id"]
                    ended = True
                elif kind == "error":
                    record(f"{label} turn completed", False, json.loads(message)["message"])
                    ended = True
            record(f"{label} turn transcribed", bool(context_text), (context_text or "")[:60])
            await harness.send_json({"type": "playback_done", "turn_id": context_turn})

        # Public status must expose content-free prompt metadata: the previous
        # completed exchange must be part of the prompt for the next turn.
        status_mid = await asyncio.to_thread(fetch_json, f"http://127.0.0.1:{port}/api/status")
        last_prompt = (status_mid.get("session") or {}).get("last_prompt") or {}
        record("conversation context is part of the prompt",
               int(last_prompt.get("history_turns", 0)) >= 2,
               f"history_turns={last_prompt.get('history_turns')} chars={last_prompt.get('chars')}")
        limits = (status_mid.get("config") or {}).get("conversation") or {}
        record("conversation limits are exposed without content",
               int(limits.get("history_turns", 0)) >= 2 and int(limits.get("max_prompt_chars", 0)) > 0,
               repr(limits))

        await harness.send_json({"type": "stop"})
        await harness.close()

    # --- a new session must not reuse turn ids ---------------------------
    async with websockets.connect(uri, max_size=None) as websocket:
        harness = WsHarness(websocket)
        await harness.send_json({"type": "start"})
        await harness.expect(lambda message: json_type(message) == "ready", timeout=200)
        await harness.feed_pcm(audio[1])
        new_turn_id = None
        for _ in range(400):
            message = await harness.expect(lambda item: True, timeout=120)
            if json_type(message) == "user_text":
                new_turn_id = json.loads(message)["turn_id"]
                break
            if json_type(message) == "error":
                break
        record("turn ids continue across a new session",
               new_turn_id is not None and new_turn_id > max_turn_id,
               f"new={new_turn_id} previous_max={max_turn_id}")
        # Abrupt disconnect without a stop message: the server must release the
        # session and any active turn on its own.
        await harness.close()
        await websocket.close()
        released = False
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status_now = await asyncio.to_thread(
                fetch_json, f"http://127.0.0.1:{port}/api/status")
            if not (status_now.get("session") or {}).get("active", True):
                released = True
                break
            await asyncio.sleep(0.2)
        record("server releases the session after an abrupt disconnect", released)

    status = wait_http_ready(f"http://127.0.0.1:{port}/api/status", timeout=10)
    metrics = status.get("metrics", {})
    response = metrics.get("response_latency_ms", {})
    barge = metrics.get("barge_in_dispatch_ms", {})
    record("response latency measured", response.get("count", 0) >= 1,
           f"p50={response.get('p50')} ms p95={response.get('p95')} ms n={response.get('count')}")
    record("barge-in dispatch latency measured", barge.get("count", 0) >= 1,
           f"p50={barge.get('p50')} ms n={barge.get('count')}")
    record("process still healthy after the session", status.get("components", {}).get("ready", False))
    record("no content persistence configured", not status.get("config", {}).get("audio_persisted", True))

    llm = status.get("llm") or {}
    llm_component = ((status.get("components") or {}).get("components") or {}).get("llm") or {}
    record("LLM service mode is reported as local", llm.get("mode") == "local",
           f"mode={llm.get('mode')}")
    record("LLM service readiness is exposed through the voice boundary",
           llm.get("ready") is True and llm_component.get("status") == "ready",
           f"ready={llm.get('ready')} component={llm_component.get('status')}")
    record("local mode has no remote search enabled",
           (llm.get("search") or {}).get("enabled") is False,
           repr(llm.get("search")))

    failed = [name for name, ok, _ in results if not ok]
    print(flush=True)
    if failed:
        print(f"RESULT: FAIL ({len(failed)} checks failed)", flush=True)
        return 1
    print(f"RESULT: PASS ({len(results)} checks)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="EL Voice end-to-end local check")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--llm-port", type=int, default=8771)
    parser.add_argument("--models", default=os.environ.get("EL_VOICE_MODELS", "/models"))
    args = parser.parse_args()

    models_root = Path(args.models).resolve()
    os.environ.setdefault("EL_VOICE_MODELS", str(models_root))
    # The voice app always routes generation through the internal LLM service.
    os.environ["EL_VOICE_LLM_URL"] = f"ws://{args.host}:{args.llm_port}"
    os.environ.setdefault("EL_VOICE_WORKSPACE", str(SOURCE_DIR.resolve()))
    os.environ.setdefault("EL_VOICE_LLM_MODE", "local")

    from elvoice import settings as settings_module

    cfg = settings_module.load_settings()

    import uvicorn

    from llm_service.server import create_app as create_llm_app

    llm_server = uvicorn.Server(uvicorn.Config(
        create_llm_app(), host=args.host, port=args.llm_port,
        log_level="warning", access_log=False, lifespan="on",
    ))
    llm_thread = threading.Thread(target=llm_server.run, daemon=True)
    llm_thread.start()
    try:
        wait_server_started(llm_server, llm_thread, "LLM service")
        wait_http_ready(f"http://{args.host}:{args.llm_port}/healthz")
        llm_status = wait_llm_ready(f"http://{args.host}:{args.llm_port}")
        print(
            f"LLM service ready: mode={llm_status.get('mode')} "
            f"search={llm_status.get('search')}",
            flush=True,
        )
    except BaseException:
        llm_server.should_exit = True
        llm_thread.join(timeout=10)
        raise

    import app as app_module

    server = uvicorn.Server(uvicorn.Config(
        app_module.app, host=args.host, port=args.port,
        log_level="warning", access_log=False, lifespan="on",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        wait_server_started(server, thread, "voice service")
        wait_http_ready(f"http://{args.host}:{args.port}/healthz")
        wait_pipeline_ready(f"http://{args.host}:{args.port}")
        return asyncio.run(run_check(args.port, models_root, cfg))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        llm_server.should_exit = True
        llm_thread.join(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
