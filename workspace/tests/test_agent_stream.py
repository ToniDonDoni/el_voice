"""Deterministic tests for the one-agent streaming event contract (SPEC-003)."""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from llm_service.agent_stream import stream_agent
from llm_service.search import SearchFailure


def model(content="", calls=None, *, kind="ai"):
    return SimpleNamespace(
        type=kind, content=content, tool_calls=calls or [],
        tool_call_chunks=[], additional_kwargs={},
    )


def tool_result(call_id, results=None, *, status="success"):
    return SimpleNamespace(
        type="tool", name="web_search", tool_call_id=call_id, status=status,
        content=json.dumps({"results": results or []}),
    )


def text_delta(value):
    return ("messages", (model(value), {"langgraph_node": "model"}))


def updated(node, message):
    return ("updates", {node: {"messages": [message]}})


class FakeAgent:
    def __init__(self, sequence):
        self.sequence = sequence
        self.config = None

    async def astream(self, state, *, stream_mode, config):
        self.config = config
        assert stream_mode == ["messages", "updates"]
        assert len(state["messages"]) > 0
        for value in self.sequence:
            await asyncio.sleep(0)
            yield value


class AgentStreamContractTests(unittest.IsolatedAsyncioTestCase):
    async def run_stream(self, sequence, *, limit=4, cancel=None):
        agent = FakeAgent(sequence)
        events = []
        observed = []

        def observe(name, **fields):
            observed.append((name, fields))

        async for event in stream_agent(
            agent, [model("question", kind="human")],
            cancel or asyncio.Event(), max_tool_calls=limit,
            max_snippet_chars=100, observe=observe,
        ):
            events.append(event)
        return events, observed, agent

    async def test_direct_response_without_search(self):
        events, observed, _ = await self.run_stream([
            text_delta("Hello"), text_delta(" world"),
            updated("model", model("Hello world")),
        ])
        self.assertEqual(events, [
            {"type": "delta", "text": "Hello"},
            {"type": "delta", "text": " world"},
        ])
        self.assertEqual(observed, [])

    async def test_search_then_followup_search_then_final_answer(self):
        call1 = {"name": "web_search", "id": "call-1", "args": {"query": "first"}}
        call2 = {"name": "web_search", "id": "call-2", "args": {"query": "second"}}
        source1 = {"title": "First", "url": "https://example.com/first",
                   "snippet": "Result one"}
        source2 = {"title": "Second", "url": "https://example.com/second",
                   "snippet": "Result two"}
        events, observed, agent = await self.run_stream([
            text_delta("Do not say this"),
            updated("model", model("hidden", [call1])),
            updated("tools", tool_result("call-1", [source1])),
            updated("model", model("", [call2])),
            updated("tools", tool_result("call-2", [source2])),
            text_delta("Final"), text_delta(" answer"),
            updated("model", model("Final answer")),
        ])
        self.assertEqual([event["type"] for event in events], [
            "tool_call", "sources", "tool_call", "sources", "delta", "delta",
        ])
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "delta"),
                         "Final answer")
        source_events = [e for e in events if e["type"] == "sources"]
        self.assertEqual([[item["title"] for item in e["sources"]] for e in source_events],
                         [["First"], ["First", "Second"]])
        self.assertEqual([e["tool_call_id"] for e in source_events],
                         ["call-1", "call-2"])
        self.assertEqual([e["tool_call_id"] for e in events if e["type"] == "tool_call"],
                         ["call-1", "call-2"])
        self.assertEqual([v[1]["tool_call_id"] for v in observed if v[0] == "agent_tool_call"],
                         ["call-1", "call-2"])
        self.assertEqual(agent.config["recursion_limit"], 10)

    async def test_late_tool_call_never_leaks_intermediate_text(self):
        call = {"name": "web_search", "id": "late", "args": {"query": "query"}}
        events, _, _ = await self.run_stream([
            text_delta("Speculative model preamble"),
            updated("model", model("Speculative model preamble", [call])),
            updated("tools", tool_result("late", [])),
            updated("model", model("Safe final")),
        ])
        self.assertEqual([event["type"] for event in events],
                         ["tool_call", "sources", "delta"])
        self.assertEqual(events[-1]["text"], "Safe final")

    async def test_reject_tool_failure_without_fabricating_sources(self):
        failed = SimpleNamespace(type="tool", name="web_search",
                                 tool_call_id="call", status="error",
                                 content="Error: search_timeout")
        with self.assertRaises(SearchFailure) as result:
            await self.run_stream([updated("tools", failed)])
        self.assertEqual(result.exception.code, "search_timeout")

    async def test_iteration_limit_is_hard(self):
        calls = [
            {"name": "web_search", "id": "first", "args": {}},
            {"name": "web_search", "id": "second", "args": {}},
        ]
        with self.assertRaisesRegex(RuntimeError, "provider_tool_loop_failed"):
            await self.run_stream([updated("model", model("", calls))], limit=1)

    async def test_empty_generation_is_error(self):
        with self.assertRaisesRegex(RuntimeError, "empty_generation"):
            await self.run_stream([updated("model", model(""))])

    async def test_cancelled_turn_has_no_events(self):
        cancelled = asyncio.Event()
        cancelled.set()
        events, _, _ = await self.run_stream([updated("model", model("answer"))],
                                              cancel=cancelled)
        self.assertEqual(events, [])

    async def test_reasoning_block_never_becomes_spoken_text(self):
        reasoning = model([
            {"type": "reasoning", "text": "private thought"},
            {"type": "text", "text": "Spoken"},
        ])
        events, _, _ = await self.run_stream([
            ("messages", (reasoning, {"langgraph_node": "model"})),
            updated("model", model("Spoken")),
        ])
        self.assertEqual(events, [{"type": "delta", "text": "Spoken"}])


if __name__ == "__main__":
    unittest.main()
