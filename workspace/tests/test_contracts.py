"""SDDTDD SPEC: tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-Review.md (R1, R3).

Contract checks for the conversation context and turn identity that the local
voice application must satisfy after the review follow-ups.

Run:  python tests/test_contracts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    from elvoice.context import build_messages
    from elvoice.turnids import TurnIdSource

    system = "system prompt"
    user = "А какие предметы ему собрать?"

    # R1: prior completed turns are part of the model prompt.
    history = [
        {"role": "user", "content": "Какой герой хорош на миду?"},
        {"role": "assistant", "content": "Пуджа хорошо играет на миду."},
    ]
    messages, stats = build_messages(system, history, user, max_prompt_chars=4000)
    check("R1 prompt starts with the system message",
          messages[0] == {"role": "system", "content": system}, repr(messages[:1]))
    check("R1 prompt ends with the current user utterance",
          messages[-1] == {"role": "user", "content": user}, repr(messages[-1:]))
    check("R1 previous exchange is present in the prompt",
          any(m.get("content") == history[0]["content"] for m in messages)
          and any(m.get("content") == history[1]["content"] for m in messages),
          repr(messages))
    check("R1 stats report one previous exchange",
          stats.get("history_turns") == 1, repr(stats))

    # R1: trimming keeps whole pairs, newest first, within the character budget.
    long_history = [
        {"role": "user", "content": "a" * 50},
        {"role": "assistant", "content": "b" * 50},
        {"role": "user", "content": "c" * 50},
        {"role": "assistant", "content": "d" * 50},
    ]
    trimmed, trimmed_stats = build_messages(system, long_history, user, max_prompt_chars=220)
    check("R1 trimming never splits a question/answer pair",
          [m["role"] for m in trimmed[1:-1]] in (["user", "assistant"], []),
          repr([m["role"] for m in trimmed]))
    check("R1 trimming drops the oldest pair first",
          trimmed_stats["history_turns"] <= 1
          and all(m["content"] != long_history[0]["content"] for m in trimmed),
          repr(trimmed_stats))

    # R3: turn ids are unique and strictly increasing across sessions.
    source = TurnIdSource()
    first_session = [source.next() for _ in range(3)]
    second_session = [source.next() for _ in range(3)]
    check("R3 turn ids are unique across sessions",
          len(set(first_session + second_session)) == 6, repr(first_session + second_session))
    check("R3 turn ids never restart at 1 on a new session",
          second_session[0] > first_session[-1], repr((first_session, second_session)))

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
