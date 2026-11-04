"""The remote provider session header must be unique per conversation.

OpenCode Go expects x-opencode-session and uses it for routing/prompt caching.
A single process-wide value would merge unrelated conversations, so every LLM
service WebSocket session gets its own UUID and a matching ChatOpenAI client.

Run:  python tests/test_remote_session_header.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    import langchain_openai

    from llm_service.backends.remote import RemoteBackend
    from llm_service.config import load_config

    captured = []

    class FakeChat:
        def __init__(self, **kwargs):
            captured.append(kwargs.get("default_headers", {}))

        def bind_tools(self, tools):
            return self

    original = langchain_openai.ChatOpenAI
    langchain_openai.ChatOpenAI = FakeChat
    try:
        config = dataclasses.replace(load_config(), mode="remote")
        config = dataclasses.replace(
            config, remote=dataclasses.replace(config.remote, api_key="test-key")
        )
        backend = RemoteBackend(config)
        backend._build_model("session-a")
        backend._build_model("session-b")
        check("two conversations get different x-opencode-session values",
              captured[0]["x-opencode-session"] == "session-a"
              and captured[1]["x-opencode-session"] == "session-b",
              repr([item.get("x-opencode-session") for item in captured]))
        check("the session header is not the process-wide fallback",
              captured[0]["x-opencode-session"] != config.remote.session_id, repr(captured[0]))
        check("a user agent header is still sent",
              "User-Agent" in captured[0], repr(captured[0]))

        # The model cache is per session and released with the connection.
        backend._model_factory = lambda session_id: FakeChat()
        backend._model_for("session-a")
        backend._model_for("session-a")
        check("one bound model is cached per session",
              len(backend._models) == 1 and "session-a" in backend._models,
              repr(list(backend._models)))
        backend.release_session("session-a")
        check("the session model is released on disconnect", not backend._models,
              repr(backend._models))
    finally:
        langchain_openai.ChatOpenAI = original

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
