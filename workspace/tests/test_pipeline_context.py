"""SDDTDD SPEC: tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-Review.md (R1, R3).

Session-level proving test: the real VoiceSession must send previous completed
exchanges to the model prompt and must use application-wide turn ids so a new
session does not restart at 1.

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
        self.prompts = []
        self.vad = FakeVad()
        self.transcripts = ["Какой герой хорош на миду?", "А какие предметы ему собрать?"]

    async def wait_ready(self) -> None:
        return None

    async def transcribe(self, pcm: bytes) -> str:
        return self.transcripts.pop(0) if len(self.transcripts) > 1 else self.transcripts[0]

    async def synthesize(self, text: str) -> bytes:
        return b"\x00\x00" * 2400  # 0.1 s of silence at 24 kHz

    async def stream_chat(self, messages, cancel):
        self.prompts.append(messages)
        for piece in ("Пуджа", " хорошо играет."):
            yield piece


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
    cfg.setdefault("conversation", {"history_turns": 4, "max_prompt_chars": 4000})
    sent = []

    async def send_json(payload):
        sent.append(payload)

    async def send_audio(payload):
        return None

    engines = FakeEngines()
    session_one = VoiceSession("session-1", cfg, engines, Metrics(), send_json, send_audio)
    await session_one.start()
    await drive_turn(session_one, 1)
    await drive_turn(session_one, 2)

    second_prompt = engines.prompts[1]
    check("R1 previous question is part of the next model prompt",
          any(message.get("content") == "Какой герой хорош на миду?" for message in second_prompt),
          repr(second_prompt))
    check("R1 previous answer is part of the next model prompt",
          any(message.get("content") == "Пуджа хорошо играет." for message in second_prompt),
          repr(second_prompt))
    check("R1 current utterance remains the last prompt message",
          second_prompt[-1]["role"] == "user"
          and second_prompt[-1]["content"] == "А какие предметы ему собрать?",
          repr(second_prompt[-1:]))
    last_prompt = session_one.last_prompt
    check("R1 prompt statistics are exposed without content",
          last_prompt.get("history_turns") == 1 and last_prompt.get("chars") > 0,
          repr(last_prompt))
    await session_one.stop()

    first_ids = [event["turn_id"] for event in sent
                 if event.get("type") == "user_speech_start"]
    session_two = VoiceSession("session-2", cfg, engines, Metrics(), send_json, send_audio)
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
