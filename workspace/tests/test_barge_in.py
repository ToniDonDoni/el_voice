"""SDDTDD: SPEC-001 interruption toggle.

The user can turn voice interruption off (default: on). With interruption off,
speech during generation/playback is ignored and the answer finishes; with it
on, a new speech onset cancels the active turn. Enforcement is server-side.

Run:  EL_VOICE_MODELS=/path/to/models python tests/test_barge_in.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_pipeline_context import FakeEngines, FakeLlm  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class SlowLlm(FakeLlm):
    """Streams one delta, then keeps generating so interruption can be tested."""

    async def generate(self, turn_id: int, text: str):
        self.requests.append((turn_id, text))
        yield {"type": "start", "turn_id": turn_id,
               "prompt": {"history_turns": max(0, len(self.requests) - 1), "chars": 10}}
        await asyncio.sleep(0.35)
        yield {"type": "delta", "turn_id": turn_id, "text": "Ответ."}
        yield {"type": "done", "turn_id": turn_id}


async def make_session(llm):
    from elvoice import settings as settings_module
    from elvoice.pipeline import Metrics, VoiceSession

    cfg = settings_module.load_settings()
    sent: list[dict] = []

    async def send_json(payload):
        sent.append(payload)

    async def send_audio(payload):
        return None

    session = VoiceSession("session-toggle", cfg, FakeEngines(), Metrics(),
                           send_json, send_audio, llm=llm)
    await session.start()
    return session, sent


async def start_turn(session, sequence: int) -> None:
    await session._on_onset(sequence)
    await session._on_utterance_end({
        "type": "end", "turn_id": sequence, "pcm": b"\x00\x00" * 1200,
        "speech_ms": 400.0,
    })
    await asyncio.sleep(0.05)


async def main() -> int:
    # 1) interruption disabled: speech during the answer is ignored.
    llm = SlowLlm()
    session, sent = await make_session(llm)
    check("interruption is enabled by default", session.barge_in_enabled is True)
    session.set_barge_in(False)
    await start_turn(session, 1)
    first_turn = session._turn
    await session._on_onset(2)
    await session._on_utterance_end({
        "type": "end", "turn_id": 2, "pcm": b"\x00\x00" * 1200, "speech_ms": 400.0,
    })
    await asyncio.sleep(0.05)
    check("with interruption off the active answer is not cancelled",
          first_turn is not None and not first_turn.cancelled,
          repr(first_turn.state if first_turn else None))
    check("ignored speech does not start a second model request",
          len(llm.requests) == 1, repr(llm.requests))
    await first_turn.task
    await session.playback_done(first_turn.id)
    check("the answer completes while interruption is off",
          any(event.get("type") == "turn_end" for event in sent), repr(sent[-3:]))
    await session.stop()

    # 2) interruption enabled: a new onset cancels the answer.
    llm = SlowLlm()
    session, sent = await make_session(llm)
    await start_turn(session, 1)
    first_turn = session._turn
    await session._on_onset(2)
    check("with interruption on a new onset cancels the active turn",
          first_turn.cancelled is True, repr(first_turn.state))
    check("cancellation is forwarded to the LLM service",
          first_turn.id in llm.cancelled, repr(llm.cancelled))
    check("no flush is needed while the cancelled turn has not played audio",
          not any(event.get("type") == "flush" for event in sent)
          and first_turn.audio_sent is False, repr(sent[-3:]))
    await session._on_utterance_end({
        "type": "end", "turn_id": 2, "pcm": b"\x00\x00" * 1200, "speech_ms": 400.0,
    })
    await asyncio.sleep(0.6)
    check("the replacement utterance is processed as a new turn",
          len(llm.requests) == 2, repr(llm.requests))
    await session.stop()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
