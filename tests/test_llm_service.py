"""Contract tests for the internal LLM service (SPEC-002).

Starts ``create_app`` with a deterministic fake backend on a random loopback
port and drives the real HTTP/WebSocket boundary. Covers the streaming event
contract, server-owned context, bounded history, sources, cancellation, error
recovery, reset and session isolation.

Run:  EL_VOICE_MODELS=/path/to/models python tests/test_llm_service.py
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

import websockets  # noqa: E402

from _harness import WsClient, fetch_json, start_server, stop_server, wait_http, wait_status  # noqa: E402
from llm_service.backends.base import BackendError  # noqa: E402
from llm_service.config import load_config  # noqa: E402
from llm_service.server import create_app  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS" if condition else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class FakeBackend:
    mode = "fake"

    def __init__(self) -> None:
        self.prompts: list[list[dict]] = []
        self.ready = False
        self.closed = False
        self.cancel_observed = 0

    async def start(self) -> None:
        self.ready = True

    async def close(self) -> None:
        self.closed = True

    def status(self) -> dict:
        return {"ready": self.ready, "mode": "fake"}

    async def stream(self, messages, cancel):
        self.prompts.append([dict(message) for message in messages])
        text = str(messages[-1]["content"])
        if "please-fail" in text:
            raise BackendError("fake_error", "Deterministic fake backend failure.", True)
        if "please-sources" in text:
            yield {"type": "tool", "name": "web_search", "query": "dota patch"}
            yield {
                "type": "sources",
                "sources": [{
                    "title": "Dota patch notes",
                    "url": "https://www.dota2.com/patches",
                    "snippet": "Latest patch notes.",
                }],
            }
        if "slow" in text:
            try:
                for index in range(40):
                    if cancel.is_set():
                        return
                    yield {"type": "delta", "text": f"slow{index} "}
                    await asyncio.sleep(0.03)
            finally:
                if cancel.is_set():
                    self.cancel_observed += 1
            return
        answer = f"You said: {text}."
        for piece in (answer[:9], answer[9:]):
            if cancel.is_set():
                self.cancel_observed += 1
                return
            yield {"type": "delta", "text": piece}
            await asyncio.sleep(0)


async def collect_until(client: WsClient, terminal: set[str], timeout: float = 10.0):
    events = await client.until(lambda message: message.get("type") in terminal, timeout)
    return events


async def turn(client: WsClient, turn_id: int, text: str, timeout: float = 10.0):
    await client.send({"type": "generate", "turn_id": turn_id, "text": text})
    return await collect_until(client, {"done", "error", "cancelled"}, timeout)


def join_deltas(events: list[dict]) -> str:
    return "".join(
        event.get("text", "") for event in events if event.get("type") == "delta"
    )


async def run_ws_checks(port: int, backend: FakeBackend, base_config, base: str) -> None:
    uri = f"ws://127.0.0.1:{port}/generate"
    async with websockets.connect(uri, max_size=None) as socket:
        client = WsClient(socket)

        first = await turn(client, 1, "first question")
        check("stream emits start then deltas then done",
              [event["type"] for event in first][:2] == ["start", "delta"]
              and first[-1]["type"] == "done",
              repr([event["type"] for event in first]))
        check("deltas reassemble the final answer",
              join_deltas(first) == "You said: first question.",
              repr(join_deltas(first)))

        second = await turn(client, 2, "second question")
        check("second turn also completes", second[-1]["type"] == "done")
        check("server owns the conversation: previous pair is in the next prompt",
              backend.prompts[1][1] == {"role": "user", "content": "first question"}
              and backend.prompts[1][2] == {
                  "role": "assistant", "content": "You said: first question."}
              and backend.prompts[1][-1] == {"role": "user", "content": "second question"},
              repr(backend.prompts[1]))
        check("voice text is the current utterance only, history is server-side",
              len(backend.prompts[1]) == 4, f"{len(backend.prompts[1])} prompt messages")
        status_mid = fetch_json(base + "/status")
        check("status reports content-free last prompt statistics",
              status_mid["last_prompt"].get("history_turns") == 1
              and set(status_mid["last_prompt"]) <= {"history_turns", "chars", "truncated"},
              repr(status_mid["last_prompt"]))
        check("status never contains prompt or answer content",
              "first question" not in json.dumps(status_mid)
              and "You said" not in json.dumps(status_mid),
              "content-free")

        sources_events = await turn(client, 3, "please-sources")
        kinds = [event["type"] for event in sources_events]
        source_event = next(
            (event for event in sources_events if event["type"] == "sources"), None)
        tool_event = next(
            (event for event in sources_events if event["type"] == "tool"), None)
        check("tool and sources events precede done",
              "tool" in kinds and "sources" in kinds and kinds[-1] == "done", repr(kinds))
        check("sources carry title, URL and snippet",
              bool(source_event) and source_event["sources"][0]["url"].startswith("https://")
              and source_event["sources"][0]["title"]
              and "snippet" in source_event["sources"][0],
              repr(source_event))
        check("tool event names the tool and query",
              bool(tool_event) and tool_event["name"] == "web_search"
              and tool_event["query"] == "dota patch", repr(tool_event))

        # Cancellation in the middle of a stream, followed by a working turn.
        await client.send({"type": "generate", "turn_id": 4, "text": "slow turn"})
        seen = await client.until(lambda message: message.get("type") == "delta", 10)
        check("slow generation started", seen[-1]["type"] == "delta", repr(seen[-1]))
        await client.send({"type": "cancel", "turn_id": 4})
        cancelled = await client.until(
            lambda message: message.get("type") == "cancelled", 10)
        check("cancel produces a cancelled event for the turn",
              cancelled[-1].get("turn_id") == 4, repr(cancelled[-1]))
        after_cancel = await turn(client, 5, "after cancel")
        check("next turn works after cancellation",
              after_cancel[-1]["type"] == "done"
              and join_deltas(after_cancel) == "You said: after cancel.",
              repr(after_cancel))
        check("backend observed the cancellation token", backend.cancel_observed >= 1,
              f"observed={backend.cancel_observed}")

        failed = await turn(client, 6, "please-fail")
        check("backend error becomes an error event",
              failed[-1]["type"] == "error"
              and failed[-1]["code"] == "fake_error"
              and failed[-1]["retryable"] is True,
              repr(failed[-1]))
        recovered = await turn(client, 7, "after error")
        check("connection recovers after an error event",
              recovered[-1]["type"] == "done", repr(recovered[-1]))

        # Reset clears the connection-owned history.
        await turn(client, 8, "before reset")
        await client.send({"type": "reset"})
        reset = await client.until(lambda message: message.get("type") == "reset", 10)
        check("reset is acknowledged", reset[-1]["type"] == "reset")
        before = len(backend.prompts)
        after_reset = await turn(client, 9, "after reset")
        check("reset clears conversation history",
              after_reset[-1]["type"] == "done"
              and len(backend.prompts[before]) == 2
              and backend.prompts[before][-1]["content"] == "after reset",
              repr(backend.prompts[before]))
        await client.close()


async def run_isolation_checks(port: int, backend: FakeBackend) -> None:
    uri = f"ws://127.0.0.1:{port}/generate"
    async with websockets.connect(uri, max_size=None) as socket_a:
        client_a = WsClient(socket_a)
        await turn(client_a, 101, "alpha secret")
        before_b = len(backend.prompts)
        async with websockets.connect(uri, max_size=None) as socket_b:
            client_b = WsClient(socket_b)
            await turn(client_b, 201, "beta question")
            prompt_b = backend.prompts[before_b]
            check("a new connection starts with empty history",
                  len(prompt_b) == 2
                  and prompt_b[-1]["content"] == "beta question"
                  and all("alpha secret" not in str(message) for message in prompt_b),
                  repr(prompt_b))
            await client_b.close()
        await client_a.close()


async def run_bounded_history_checks(backend: FakeBackend, config) -> None:
    bounded = dataclasses.replace(config, history_turns=2)
    app = create_app(backend=backend, config=bounded)
    server, thread, port = start_server(app)
    try:
        wait_status(f"http://127.0.0.1:{port}", timeout=20)
        uri = f"ws://127.0.0.1:{port}/generate"
        async with websockets.connect(uri, max_size=None) as socket:
            client = WsClient(socket)
            for turn_id in range(1, 5):
                events = await turn(client, turn_id, f"question {turn_id}")
                assert events[-1]["type"] == "done", events
            final_prompt = backend.prompts[-1]
            check("bounded history keeps at most the configured pairs",
                  sum(1 for message in final_prompt if message["role"] == "user") == 3,
                  f"{len(final_prompt)} messages: {repr(final_prompt)}")
            check("bounded history drops the oldest pair first",
                  all("question 1" not in str(message) for message in final_prompt)
                  and any("question 2" in str(message) for message in final_prompt)
                  and any("question 3" in str(message) for message in final_prompt),
                  repr(final_prompt))
            await client.close()
    finally:
        stop_server(server, thread)


async def run_remote_unready_checks(config) -> None:
    """Remote mode with no key must stay unready and fail loudly, not fall back."""
    remote = dataclasses.replace(
        config, mode="remote",
        remote=dataclasses.replace(config.remote, api_key=""),
    )
    app = create_app(config=remote)
    server, thread, port = start_server(app)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_http(base + "/healthz", timeout=10)
        status = {}
        deadline = __import__("time").monotonic() + 15
        while __import__("time").monotonic() < deadline:
            status = fetch_json(base + "/status")
            if status.get("error"):
                break
            await asyncio.sleep(0.05)
        check("remote mode without a key reports an actionable readiness error",
              status.get("ready") is False
              and "OPENCODE_GO_API_KEY" in str(status.get("error")),
              repr(status.get("error")))
        check("remote status exposes provider config but no credential",
              status.get("mode") == "remote"
              and status.get("remote") == {
                  "base_url": config.remote.base_url,
                  "model": config.remote.model,
              }
              and "sk-" not in json.dumps(status),
              repr(status.get("remote")))
        async with websockets.connect(f"ws://127.0.0.1:{port}/generate",
                                      max_size=None) as socket:
            client = WsClient(socket)
            await client.send({"type": "generate", "turn_id": 1, "text": "hi"})
            events = await client.until(
                lambda message: message.get("type") in ("error", "done"), 10)
            check("generate in unready remote mode returns an actionable error",
                  events[-1]["type"] == "error"
                  and events[-1]["code"] == "not_ready"
                  and "OPENCODE_GO_API_KEY" in events[-1]["message"],
                  repr(events[-1]))
            await client.close()
    finally:
        stop_server(server, thread)


async def main() -> int:
    config = load_config()
    backend = FakeBackend()
    app = create_app(backend=backend, config=config)
    server, thread, port = start_server(app)
    base = f"http://127.0.0.1:{port}"
    try:
        status = wait_status(base, timeout=20)
        check("healthz reports process liveness",
              fetch_json(base + "/healthz")["status"] == "ok")
        check("status reports mode and readiness",
              status["mode"] == config.mode and status["ready"] is True, repr(status))
        check("status exposes context limits without content",
              status["context"] == {
                  "history_turns": config.history_turns,
                  "max_prompt_chars": config.max_prompt_chars}
              and status["limits"]["max_iterations"] == config.max_iterations,
              repr(status["context"]))
        check("status starts with empty prompt statistics",
              status["last_prompt"] == {}, repr(status["last_prompt"]))

        await run_ws_checks(port, backend, config, base)
        await run_isolation_checks(port, backend)

        status_after = fetch_json(base + "/status")
        check("status remains content-free after sessions close",
              set(status_after["last_prompt"]) <= {"history_turns", "chars", "truncated"}
              and "first question" not in json.dumps(status_after),
              repr(status_after["last_prompt"]))
        check("backend was not closed while serving", backend.closed is False)
    finally:
        stop_server(server, thread)

    await run_bounded_history_checks(backend, config)
    await run_remote_unready_checks(config)

    check("backend close is called on service shutdown", backend.closed is True)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
