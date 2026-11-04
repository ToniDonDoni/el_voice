"""Deterministic tests for the remote LangChain agent (SPEC-002).

Uses an injected LangChain-style fake chat model and a fake search callable, so
no network or API key is needed. Proves tool-call assembly, sources, ToolMessage
pairing by ``tool_call_id``, final-answer-only deltas, the iteration bound,
search failure reporting and redacted provider errors.

Run:  EL_VOICE_MODELS=/path/to/models python tests/test_remote_backend.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage  # noqa: E402

from llm_service.backends.base import BackendError  # noqa: E402
from llm_service.backends.remote import RemoteBackend  # noqa: E402
from llm_service.config import load_config  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class FakeChatModel:
    """Minimal ``bind_tools``/``astream`` object accepted by RemoteBackend."""

    def __init__(self, responses: list[list] | list[Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []
        self.tools: list = []

    def bind_tools(self, tools):
        self.tools = list(tools)
        return self

    async def astream(self, messages):
        self.calls.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        for chunk in response:
            yield chunk


def tool_chunk(query: str = "Dota 2 patch 7.41", call_id: str = "call_1") -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{
            "name": "web_search",
            "args": json.dumps({"query": query}),
            "id": call_id,
            "index": 0,
        }],
    )


def final_chunks(text: str = "Патч 7.41 вышел на этой неделе.") -> list[AIMessageChunk]:
    middle = len(text) // 2
    return [AIMessageChunk(content=text[:middle]), AIMessageChunk(content=text[middle:])]


async def collect(backend: RemoteBackend, messages: list[dict]) -> list[dict]:
    cancel = threading.Event()
    collected = []
    async for event in backend.stream(messages, cancel):
        collected.append(event)
    return collected


def joined_deltas(events: list[dict]) -> str:
    return "".join(event["text"] for event in events if event["type"] == "delta")


async def first_search_scenario(config) -> None:
    model = FakeChatModel([[tool_chunk()], final_chunks()])
    searches: list[dict] = []

    async def fake_search(query: str, **kwargs):
        searches.append({"query": query, **kwargs})
        return [
            {
                "title": "Dota 2 patch notes",
                "url": "https://www.dota2.com/patches",
                "snippet": "notes " * 200,
            },
            {
                "title": "private endpoint",
                "url": "http://127.0.0.1/admin",
                "snippet": "must not be shown",
            },
        ]

    backend = RemoteBackend(config, chat_model=model, search=fake_search)
    await backend.start()
    check("bind_tools receives exactly one typed web_search tool",
          len(model.tools) == 1
          and model.tools[0].name == "web_search"
          and "query" in model.tools[0].args_schema.model_json_schema()["properties"],
          repr(model.tools))

    events = await collect(backend, [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Что нового в патче Доты?"},
    ])
    kinds = [event["type"] for event in events]
    tool_event = next(event for event in events if event["type"] == "tool")
    source_event = next(event for event in events if event["type"] == "sources")
    spoken = joined_deltas(events)

    check("first response produced a tool event with the model's query",
          tool_event["name"] == "web_search"
          and tool_event["query"] == "Dota 2 patch 7.41", repr(tool_event))
    check("search was called once with the model's query",
          len(searches) == 1 and searches[0]["query"] == "Dota 2 patch 7.41"
          and searches[0]["max_results"] == config.max_search_results,
          repr(searches))
    check("search results became bounded sources",
          len(source_event["sources"]) == 1
          and source_event["sources"][0]["url"] == "https://www.dota2.com/patches"
          and len(source_event["sources"][0]["snippet"]) <= config.search_snippet_chars,
          repr(source_event))
    check("private/loopback search URLs are not shown",
          all("127.0.0.1" not in json.dumps(source) for source in source_event["sources"]),
          repr(source_event["sources"]))
    check("final answer is streamed as deltas",
          spoken == "Патч 7.41 вышел на этой неделе.", repr(spoken))
    check("spoken text contains no tool args or control fields",
          "web_search" not in spoken and "call_1" not in spoken
          and "{" not in spoken and "query" not in spoken, repr(spoken))
    check("tool and sources precede the final delta",
          kinds.index("tool") < kinds.index("sources") < kinds.index("delta"), repr(kinds))

    second_prompt = model.calls[1]
    assistant_calls = [
        message for message in second_prompt
        if isinstance(message, AIMessage) and message.tool_calls
    ]
    tool_messages = [
        message for message in second_prompt if isinstance(message, ToolMessage)
    ]
    check("assistant tool call is preserved in the follow-up prompt",
          len(assistant_calls) == 1
          and assistant_calls[0].tool_calls[0]["id"] == "call_1",
          repr(assistant_calls))
    check("ToolMessage is paired with the tool_call_id",
          len(tool_messages) == 1 and tool_messages[0].tool_call_id == "call_1",
          repr(tool_messages))
    check("tool result passed to the model contains source data and an untrusted-data note",
          "dota2.com" in tool_messages[0].content
          and "Untrusted" in tool_messages[0].content,
          tool_messages[0].content[:160])
    check("a successful completion records that the provider was reached",
          backend.status()["provider_verified"] is True, repr(backend.status()))


def split_tool_chunks(query: str = "Dota 2 patch 7.41") -> list:
    payload = json.dumps({"query": query})
    first, second = payload[:12], payload[12:]
    return [
        AIMessageChunk(content="", tool_call_chunks=[{
            "name": "web_search", "args": first, "id": "call_split", "index": 0,
        }]),
        AIMessageChunk(content="", tool_call_chunks=[{
            "name": None, "args": second, "id": None, "index": 0,
        }]),
    ]


async def split_tool_args_scenario(config) -> None:
    model = FakeChatModel([split_tool_chunks(), final_chunks("Готово.")])
    searches: list[str] = []

    async def fake_search(query: str, **kwargs):
        searches.append(query)
        return [{"title": "ok", "url": "https://example.com/ok", "snippet": "ok"}]

    backend = RemoteBackend(config, chat_model=model, search=fake_search)
    await backend.start()
    events = await collect(backend, [{"role": "user", "content": "Что нового?"}])
    check("partial tool-call argument chunks are assembled into one call",
          searches == ["Dota 2 patch 7.41"], repr(searches))
    check("split tool-call assembly still emits sources and a final answer",
          any(event["type"] == "sources" for event in events)
          and joined_deltas(events) == "Готово.", repr([e["type"] for e in events]))


async def empty_search_scenario(config) -> None:
    model = FakeChatModel([[tool_chunk("no results query")], final_chunks("Не нашёл.")])

    async def empty_search(query: str, **kwargs):
        return []

    backend = RemoteBackend(config, chat_model=model, search=empty_search)
    await backend.start()
    events = await collect(backend, [{"role": "user", "content": "search please"}])
    source_event = next(event for event in events if event["type"] == "sources")
    check("empty search yields an empty sources event and an honest answer",
          source_event["sources"] == [] and joined_deltas(events) == "Не нашёл.",
          repr(events))


async def max_iterations_scenario(config) -> None:
    bounded = dataclasses.replace(config, max_iterations=3)
    model = FakeChatModel([[tool_chunk(call_id=f"call_{index}")]
                           for index in range(5)])
    calls: list[str] = []

    async def fake_search(query: str, **kwargs):
        calls.append(query)
        return []

    backend = RemoteBackend(bounded, chat_model=model, search=fake_search)
    await backend.start()
    try:
        await collect(backend, [{"role": "user", "content": "loop forever"}])
    except BackendError as exc:
        check("agent loop is bounded by max_iterations",
              exc.code == "max_iterations" and len(model.calls) == 3,
              f"code={exc.code} model_calls={len(model.calls)}")
        check("max-iteration error is explicit and not a fabricated answer",
              "No answer was fabricated" in exc.message, exc.message[:120])
    else:
        check("agent loop is bounded by max_iterations", False, "no BackendError raised")


async def search_failure_scenario(config) -> None:
    model = FakeChatModel([[tool_chunk()], final_chunks()])

    async def failing_search(query: str, **kwargs):
        raise RuntimeError("throttled for key sk-live-secret")

    backend = RemoteBackend(config, chat_model=model, search=failing_search)
    await backend.start()
    events = await collect(backend, [{"role": "user", "content": "search"}])
    spoken = joined_deltas(events)
    check("search failure/timeout with no answer speaks the fallback line",
          spoken == "Я так качался.", repr(spoken))
    check("search failure does not leak a credential",
          "sk-live-secret" not in json.dumps(events), "no credential in events")
    check("search failure does not fabricate an answer",
          "sk-live-secret" not in spoken and "throttled" not in spoken, spoken)


class ProviderError(Exception):
    status_code = 401


async def provider_failure_scenario(config) -> None:
    key = "sk-test-only-secret"
    remote = dataclasses.replace(config.remote, api_key=key)
    keyed = dataclasses.replace(config, remote=remote)
    model = FakeChatModel([
        ProviderError(f"401 invalid credential: {key} Authorization: Bearer {key}"),
    ])
    backend = RemoteBackend(keyed, chat_model=model, search=lambda *a, **k: [])
    await backend.start()
    try:
        await collect(backend, [{"role": "user", "content": "hi"}])
    except BackendError as exc:
        check("provider failure is actionable and mentions the key variable",
              exc.code == "provider_auth_failed"
              and "OPENCODE_GO_API_KEY" in exc.message
              and "401" in exc.message,
              f"{exc.code}: {exc.message}")
        check("provider failure redacts the API key and bearer header",
              key not in exc.message and "Bearer [redacted]" in exc.message,
              exc.message)
        check("provider failure is not silently converted into an answer",
              "fallback answer was generated" in exc.message, exc.message)
    else:
        check("provider failure is actionable and mentions the key variable", False,
              "no BackendError raised")


async def missing_key_scenario(config) -> None:
    remote = dataclasses.replace(config.remote, api_key="")
    backend = RemoteBackend(dataclasses.replace(config, remote=remote))
    try:
        await backend.start()
    except BackendError as exc:
        check("missing API key is reported without a fallback",
              exc.code == "missing_api_key"
              and "OPENCODE_GO_API_KEY" in exc.message
              and "no local fallback" in exc.message.lower(),
              f"{exc.code}: {exc.message}")
    else:
        check("missing API key is reported without a fallback", False,
              "start did not raise")
    status = backend.status()
    check("unready remote status exposes no credential",
          status["ready"] is False
          and "api_key" not in json.dumps(status)
          and status["provider_verified"] is False,
          repr(status))


async def main() -> int:
    config = dataclasses.replace(
        load_config(),
        mode="remote",
        max_iterations=3,
        max_search_results=2,
        search_snippet_chars=120,
    )
    await split_tool_args_scenario(config)
    await first_search_scenario(config)
    await empty_search_scenario(config)
    await max_iterations_scenario(config)
    await search_failure_scenario(config)
    await provider_failure_scenario(config)
    await missing_key_scenario(config)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
