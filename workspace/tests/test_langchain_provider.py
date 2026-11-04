"""Contract tests against the pinned LangChain graph when its packages are installed."""

from __future__ import annotations

import asyncio
import json
import unittest

try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, ToolMessage
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
            AIMessage(content="Search completed"),
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
            session_id="a" * 32,
        )]

        self.assertEqual([event["type"] for event in events], ["tool_call", "sources", "delta"])
        self.assertEqual(events[1]["sources"][0]["title"], "Axe guide")
        self.assertIn("Silver Edge", events[2]["text"])
        seen = model.seen_messages[0]
        self.assertIn("What should I buy against Axe?", [message.content for message in seen])
        self.assertIn("Silver Edge can help against Berserker's Call.", [message.content for message in seen])
        self.assertIn("Find a current guide for that item", [message.content for message in seen])

    def test_mixed_tool_call_message_is_rebuilt_with_only_matched_ids(self):
        call = AIMessage(content="", tool_calls=[
            {"id": "matched", "name": "web_search", "args": {"query": "patch"}},
            {"id": "orphan", "name": "web_search", "args": {"query": "orphan"}},
        ])
        result = ToolMessage(content="results", tool_call_id="matched")

        paired = paired_tool_messages([call, result])

        self.assertEqual([item["id"] for item in paired[0].tool_calls], ["matched"])
        self.assertEqual(paired[1].tool_call_id, "matched")

    def test_search_rate_limit_survives_real_agent_and_service_error_mapping(self):
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
            AIMessage(content="Search failed"),
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
                json={"text": "latest Dota patch"},
            )
        events = [json.loads(line) for line in response.iter_lines() if line]
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "search_rate_limited")
        self.assertNotIn("OPENCODE_GO_API_KEY", response.text)
