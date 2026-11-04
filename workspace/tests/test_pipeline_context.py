"""SDDTDD SPEC: tasks/SPEC-002-REMOTE-VOICE-LLM-SERVICE.md.

Session-level proving test for the new LLM boundary: the voice pipeline sends
only the current final utterance, the internal LLM service owns conversation
history and the system prompt, source events are forwarded to the browser, and
turn ids remain application-wide (SPEC-001 review R3).

The context itself is proved at the real service boundary in
``tests/test_llm_service.py``; this check proves the voice side of the contract
without loading any model.

Run:  EL_VOICE_MODELS=/path/to/models python tests/test_pipeline_context.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class FakeVad:
    def __init__(self):
        self.reset_calls = 0

    def reset_states(self) -> None:
        self.reset_calls += 1

    def speech_probability(self, frame: bytes) -> float:
        return 0.0


class FakeEngines:
    def __init__(self):
        self.vad = FakeVad()
        self.transcripts = ["Какой герой хорош на миду?", "А какие предметы ему собрать?"]

    async def wait_ready(self) -> None:
        return None

    async def transcribe(self, pcm: bytes) -> str:
        return self.transcripts.pop(0) if len(self.transcripts) > 1 else self.transcripts[0]

    async def synthesize(self, text: str) -> bytes:
        return b"\x00\x00" * 2400  # 0.1 s of silence at 24 kHz


class FakeLlm:
    """Stand-in for the internal LLM service client at the pipeline boundary."""

    def __init__(self):
        self.requests = []
        self.started = 0
        self.closed = 0
        self.cancelled = []

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    async def cancel(self, turn_id: int) -> None:
        self.cancelled.append(turn_id)

    async def generate(self, turn_id: int, text: str):
        self.requests.append((turn_id, text))
        yield {"type": "start", "turn_id": turn_id,
               "prompt": {"history_turns": max(0, len(self.requests) - 1), "chars": 123}}
        yield {"type": "sources", "turn_id": turn_id, "sources": [{
            "title": "Dota patch", "url": "https://www.dota2.com/patches",
            "snippet": "notes",
        }]}
        yield {"type": "delta", "turn_id": turn_id, "text": "Пуджа"}
        yield {"type": "delta", "turn_id": turn_id, "text": " хорошо играет."}
        yield {"type": "done", "turn_id": turn_id, "chars": 18}


async def drive_turn(session, sequence: int, audio_ms: int = 1200) -> None:
    await session._on_onset(sequence)
    await session._on_utterance_end({
        "type": "end", "turn_id": sequence,
        "pcm": b"\x00\x00" * (audio_ms * 16), "speech_ms": float(audio_ms),
    })
    task = session._turn.task
    await task
    turn = session._turn
    if turn is not None:
        await session.playback_done(turn.id)


async def main() -> int:
    from elvoice import settings as settings_module
    from elvoice.pipeline import Metrics, VoiceSession

    cfg = settings_module.load_settings()
    sent = []

    async def send_json(payload):
        sent.append(payload)

    async def send_audio(payload):
        return None

    engines = FakeEngines()
    llm = FakeLlm()
    session_one = VoiceSession(
        "session-1", cfg, engines, Metrics(), send_json, send_audio, llm=llm)
    await session_one.start()
    check("starting a session opens the LLM service connection", llm.started == 1)
    await drive_turn(session_one, 1)
    await drive_turn(session_one, 2)

    check("voice sends only the first final utterance",
          llm.requests[0][1] == "Какой герой хорош на миду?", repr(llm.requests))
    check("voice sends only the current utterance, not history",
          llm.requests[1][1] == "А какие предметы ему собрать?"
          and all("Пуджа" not in request[1] for request in llm.requests),
          repr(llm.requests))
    check("turn ids are forwarded to the LLM service",
          [request[0] for request in llm.requests] == sorted(request[0] for request in llm.requests),
          repr([request[0] for request in llm.requests]))
    check("prompt statistics from the service are exposed without content",
          session_one.last_prompt.get("history_turns") == 1
          and session_one.last_prompt.get("chars") == 123,
          repr(session_one.last_prompt))
    sources = [event for event in sent if event.get("type") == "assistant_sources"]
    check("sources are forwarded to the browser",
          bool(sources) and sources[0]["sources"][0]["url"].startswith("https://"),
          repr(sources[:1]))
    assistant_text = "".join(
        event.get("delta", "") for event in sent if event.get("type") == "assistant_text")
    check("answer deltas are spoken text only",
          assistant_text.count("Пуджа хорошо играет.") == 2, repr(assistant_text))
    await session_one.stop()
    check("stopping a session closes the LLM service connection", llm.closed == 1)

    first_ids = [event["turn_id"] for event in sent
                 if event.get("type") == "user_speech_start"]
    session_two = VoiceSession(
        "session-2", cfg, engines, Metrics(), send_json, send_audio, llm=llm)
    await session_two.start()
    await drive_turn(session_two, 1)
    second_ids = [event["turn_id"] for event in sent
                  if event.get("type") == "user_speech_start"][len(first_ids):]
    check("R3 a new session does not restart turn ids at 1",
          bool(second_ids) and min(second_ids) > max(first_ids),
          f"first={first_ids} second={second_ids}")
    await session_two.stop()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
