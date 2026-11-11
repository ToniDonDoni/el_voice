"""Contract tests against the pinned LangChain graph when its packages are installed."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
    from langchain_core.outputs import ChatGenerationChunk
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from fastapi.testclient import TestClient
from llm_service.context import paired_tool_messages
from llm_service.app import LLMService, create_app
from llm_service.remote_provider import OpenCodeLangChainProvider
from llm_service.search import SearchFailure, WebSearch, make_langchain_search_tool


@unittest.skipUnless(LANGCHAIN_AVAILABLE, "Install the pinned LangChain dependencies to run graph tests")
class LangChainGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_clipboard_image_stays_with_question_through_final_model_call(self):
        from image_fixtures import image_url

        class Model(FakeMessagesListChatModel):
            def bind_tools(self, _tools, **_kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                seen.append(messages)
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        seen = []
        provider = object.__new__(OpenCodeLangChainProvider)
        provider.model = Model(responses=[
            AIMessage(content="", tool_calls=[{
                "id": "finish", "name": "begin_final_answer", "args": {}}]),
            AIMessage(content="Red square and blue circle."),
        ])
        provider.tool_iterations = 3
        image = image_url()
        events = [event async for event in provider.stream(
            "Describe", [], asyncio.Event(), image=image)]
        self.assertEqual(events[-1]["text"], "Red square and blue circle.")
        self.assertEqual(len(seen), 2)
        for messages in seen:
            question = next(message for message in messages if message.type == "human")
            self.assertEqual(question.content, [{"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": image}}])

    async def test_search_tool_is_registered_only_for_turns_that_enable_it(self):
        captured = []

        async def empty_stream(*_args, **_kwargs):
            if False:
                yield None

        async def consume(search_enabled):
            provider = object.__new__(OpenCodeLangChainProvider)
            provider.model = object()
            provider.search_tool = make_langchain_search_tool(WebSearch())
            provider.tool_iterations = 3
            with patch("langchain.agents.create_agent", side_effect=lambda **kwargs: (
                captured.append(kwargs["tools"]) or object()
            )), patch("llm_service.remote_provider.stream_agent", side_effect=empty_stream):
                events = [event async for event in provider.stream(
                    "Answer briefly", [], asyncio.Event(), session_id="a" * 32,
                    turn_id="b" * 32, search_enabled=search_enabled,
                )]
            self.assertEqual(events, [])

        await consume(False)
        await consume(True)
        self.assertEqual([tool.name for tool in captured[0]], ["begin_final_answer"])
        self.assertEqual([tool.name for tool in captured[1]], ["web_search", "begin_final_answer"])

    async def test_real_agent_runs_one_search_and_plans_from_bounded_history(self):
        class Model(FakeMessagesListChatModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                object.__setattr__(self, "seen_messages", [])

            def bind_tools(self, _tools, **_kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.seen_messages.append(messages)
                return super()._generate(messages, stop, run_manager, **kwargs)

        class Search(WebSearch):
            async def tool_payload(self, query):
                return json.dumps({"query": query, "results": [{
                    "title": "Axe guide", "url": "https://example.com/axe", "snippet": "Item counters"
                }]})

        model = Model(responses=[
            AIMessage(content="", tool_calls=[{
                "id": "call-search", "name": "web_search", "args": {"query": "current Axe counter guide"}
            }]),
            AIMessage(content="", tool_calls=[{
                "id": "call-finish", "name": "begin_final_answer", "args": {}
            }]),
            AIMessage(content="The guide recommends a Silver Edge for Axe."),
        ])
        provider = object.__new__(OpenCodeLangChainProvider)
        provider.model = model
        provider.search_tool = make_langchain_search_tool(Search())
        provider.tool_iterations = 3
        history = [
            {"role": "user", "content": "What should I buy against Axe?"},
            {"role": "assistant", "content": "Silver Edge can help against Berserker's Call."},
        ]

        events = [event async for event in provider.stream(
            "Find a current guide for that item", history, asyncio.Event(),
            session_id="a" * 32, search_enabled=True,
        )]

        self.assertEqual([event["type"] for event in events], ["tool_call", "sources", "delta"])
        self.assertEqual(events[1]["sources"][0]["title"], "Axe guide")
        self.assertIn("Silver Edge", events[2]["text"])
        seen = model.seen_messages[0]
        self.assertIn("What should I buy against Axe?", [message.content for message in seen])
        self.assertIn("Silver Edge can help against Berserker's Call.", [message.content for message in seen])
        self.assertIn("Find a current guide for that item", [message.content for message in seen])

    async def test_real_langchain_streams_first_speech_delta_before_model_finishes(self):
        class PausingModel(FakeMessagesListChatModel):
            def __init__(self):
                super().__init__(responses=[AIMessage(content="unused")])
                object.__setattr__(self, "release", asyncio.Event())
                object.__setattr__(self, "final_started", asyncio.Event())
                object.__setattr__(self, "bound_tool_counts", [])

            def bind_tools(self, tools, **kwargs):
                self.bound_tool_counts.append((len(tools), kwargs.get("tool_choice")))
                return self

            async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
                final = any(
                    isinstance(message, ToolMessage)
                    and message.name == "begin_final_answer"
                    for message in messages
                )
                if not final:
                    yield ChatGenerationChunk(message=AIMessageChunk(
                        content="", tool_call_chunks=[{
                            "name": "begin_final_answer", "args": "{}",
                            "id": "commit-1", "index": 0,
                        }]
                    ))
                    return
                self.final_started.set()
                yield ChatGenerationChunk(message=AIMessageChunk(content="Early"))
                await self.release.wait()
                yield ChatGenerationChunk(message=AIMessageChunk(content=" reply"))

        model = PausingModel()
        provider = object.__new__(OpenCodeLangChainProvider)
        provider.model = model
        provider.search_tool = make_langchain_search_tool(WebSearch())
        provider.tool_iterations = 3
        generator = provider.stream(
            "Explain one thing briefly", [], asyncio.Event(),
            session_id="a" * 32, turn_id="b" * 32, search_enabled=True,
        )
        try:
            first = await asyncio.wait_for(anext(generator), timeout=3.0)
            self.assertEqual(first, {"type": "delta", "text": "Early"})
            self.assertTrue(model.final_started.is_set())
            self.assertFalse(model.release.is_set(),
                             "Speech must arrive before final model completion")
            model.release.set()
            remaining = [event async for event in generator]
            self.assertEqual(remaining, [{"type": "delta", "text": " reply"}])
            self.assertEqual(model.bound_tool_counts, [(2, "required")],
                             "Final request must invoke the unbound model without tools")
        finally:
            model.release.set()
            await generator.aclose()

    async def test_cancel_closes_graph_before_turn_context_reset(self):
        closed = []

        class FakeGraph:
            async def astream(self, _state, **_kwargs):
                try:
                    yield ("updates", {"model": {"messages": [
                        AIMessage(content="", tool_calls=[{
                            "id": "call-finish", "name": "begin_final_answer", "args": {}
                        }])
                    ]}})
                    yield ("updates", {"tools": {"messages": [
                        ToolMessage(content="Ready", tool_call_id="call-finish",
                                    name="begin_final_answer")
                    ]}})
                    yield ("messages", (
                        AIMessageChunk(content="Hello"),
                        {"langgraph_node": "model"},
                    ))
                    await asyncio.Event().wait()
                finally:
                    closed.append(True)

        provider = object.__new__(OpenCodeLangChainProvider)
        provider.model = object()
        provider.search_tool = object()
        provider.tool_iterations = 3
        with patch("langchain.agents.create_agent", return_value=FakeGraph()):
            response = provider.stream(
                "hello", [], asyncio.Event(), session_id="a" * 32,
                turn_id="b" * 32,
            )
            first = await anext(response)
            self.assertEqual(first, {"type": "delta", "text": "Hello"})
            await response.aclose()
        self.assertEqual(closed, [True])
        self.assertEqual(provider._trace_id.get(), "")

    def test_mixed_tool_call_message_is_rebuilt_with_only_matched_ids(self):
        call = AIMessage(content="", tool_calls=[
            {"id": "matched", "name": "web_search", "args": {"query": "patch"}},
            {"id": "orphan", "name": "web_search", "args": {"query": "orphan"}},
        ])
        result = ToolMessage(content="results", tool_call_id="matched")

        paired = paired_tool_messages([call, result])

        self.assertEqual([item["id"] for item in paired[0].tool_calls], ["matched"])
        self.assertEqual(paired[1].tool_call_id, "matched")

    def test_search_rate_limit_becomes_a_tool_result_and_allows_final_disclosure(self):
        class Model(FakeMessagesListChatModel):
            def bind_tools(self, _tools, **_kwargs):
                return self

        class Search(WebSearch):
            async def tool_payload(self, _query):
                raise SearchFailure("search_rate_limited")

        provider = object.__new__(OpenCodeLangChainProvider)
        provider.model = Model(responses=[
            AIMessage(content="", tool_calls=[{
                "id": "call-search", "name": "web_search", "args": {"query": "latest patch"}
            }]),
            AIMessage(content="", tool_calls=[{
                "id": "call-final", "name": "begin_final_answer", "args": {}
            }]),
            AIMessage(content="I cannot verify the latest patch."),
        ])
        provider.search_tool = make_langchain_search_tool(Search())
        provider.tool_iterations = 3
        class HTTPClient:
            async def aclose(self):
                pass
        provider.http_async_client = HTTPClient()
        app = create_app(LLMService(provider, mode="remote"))
        with TestClient(app) as client:
            response = client.post(
                f"/v1/sessions/{'a' * 32}/turns/{'b' * 32}/stream",
                json={"text": "latest Dota patch", "search_enabled": True},
            )
        events = [json.loads(line) for line in response.iter_lines() if line]
        self.assertEqual([event["type"] for event in events],
                         ["tool_call", "sources", "delta", "done"])
        self.assertEqual(events[1]["status"], "error")
        self.assertEqual(events[1]["error_code"], "search_rate_limited")
        self.assertEqual(events[1]["sources"], [])
        self.assertEqual(events[1]["tool_call_id"], "call-search")
        self.assertIn("cannot verify", events[2]["text"])
        self.assertNotIn("OPENCODE_GO_API_KEY", response.text)

    def test_parallel_real_agent_keeps_success_when_other_search_fails(self):
        class Model(FakeMessagesListChatModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                object.__setattr__(self, "seen", [])

            def bind_tools(self, _tools, **_kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.seen.append(messages)
                return super()._generate(messages, stop, run_manager, **kwargs)

        class Search(WebSearch):
            async def tool_payload(self, query):
                if query == "good":
                    return json.dumps({"query": query, "results": [{
                        "title": "Evidence", "url": "https://example.com/evidence",
                        "snippet": "Verified fact",
                    }]})
                raise SearchFailure("search_unavailable")

        model = Model(responses=[
            AIMessage(content="", tool_calls=[
                {"id": "good", "name": "web_search", "args": {"query": "good"}},
                {"id": "bad", "name": "web_search", "args": {"query": "bad"}},
            ]),
            AIMessage(content="", tool_calls=[{
                "id": "final", "name": "begin_final_answer", "args": {},
            }]),
            AIMessage(content="The available source confirms the result."),
        ])
        provider = object.__new__(OpenCodeLangChainProvider)
        provider.model = model
        provider.search_tool = make_langchain_search_tool(Search())
        provider.tool_iterations = 4
        class HTTPClient:
            async def aclose(self):
                pass
        provider.http_async_client = HTTPClient()
        app = create_app(LLMService(provider, mode="remote"))
        with TestClient(app) as client:
            response = client.post(
                f"/v1/sessions/{'a' * 32}/turns/{'b' * 32}/stream",
                json={"text": "Check two pieces of information", "search_enabled": True},
            )
        events = [json.loads(line) for line in response.iter_lines() if line]
        self.assertEqual(events[-1]["type"], "done")
        self.assertNotIn("replace", [item["type"] for item in events])
        self.assertEqual([item["tool_call_id"] for item in events
                          if item["type"] == "tool_call"], ["good", "bad"])
        result_events = [item for item in events if item["type"] == "sources"]
        self.assertEqual(len(result_events), 2)
        by_id = {item["tool_call_id"]: item for item in result_events}
        self.assertEqual(by_id["good"]["status"], "success")
        self.assertEqual(by_id["bad"]["status"], "error")
        self.assertEqual(by_id["bad"]["error_code"], "search_unavailable")
        self.assertTrue(all(item["sources"] and
                            item["sources"][0]["title"] == "Evidence"
                            for item in result_events[-1:]))
        self.assertIn("confirms", "".join(item.get("text", "") for item in events
                                          if item["type"] == "delta"))
        messages = [message for turn in model.seen for message in turn
                    if isinstance(message, ToolMessage)]
        paired = {message.tool_call_id: json.loads(message.content)
                  for message in messages if message.tool_call_id in {"good", "bad"}}
        self.assertEqual(paired["good"]["results"][0]["title"], "Evidence")
        self.assertEqual(paired["bad"]["error"], "search_unavailable")
