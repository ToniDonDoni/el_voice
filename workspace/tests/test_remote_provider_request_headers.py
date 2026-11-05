"""HTTP-boundary regression for the OpenCode Go request identity contract.

SDDTDD SPEC: tasks/SPEC-002-REMOTE-VOICE-LLM-SERVICE.md
Every request made by the Remote LLM during a routed turn must carry a stable
OpenCode session id and the EL Voice client user-agent. This includes regular
generation and agent tool continuations in a single execution.
"""

from __future__ import annotations

import json
import asyncio
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from llm_service.app import LLMService, create_app
from llm_service.remote_provider import OpenCodeLangChainProvider, _ObservedProviderStream


try:
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@unittest.skipUnless(LANGCHAIN_AVAILABLE, "Install the pinned LangChain dependencies")
class RemoteProviderRequestHeaderTests(unittest.TestCase):
    def test_raw_sse_telemetry_counts_text_reasoning_finish_and_done_without_content(self):
        class ChunkedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}\n\n'
                yield b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\n'
                yield b'data: [DONE]\n\n'

            async def aclose(self):
                return None

        async def consume(stream):
            return [chunk async for chunk in stream]

        context = {"trace_id": "a" * 32, "turn_id": "a" * 32,
                   "session_id": "b" * 32, "request_index": 1,
                   "provider_host": "opencode.ai", "path": "/chat/completions",
                   "model": "deepseek-v4.1-flash", "status_code": 200,
                   "content_type": "text/event-stream"}
        stream = _ObservedProviderStream(ChunkedStream(), context)
        with self.assertLogs("el_voice.remote_provider", level="INFO") as captured:
            chunks = asyncio.run(consume(stream))

        self.assertEqual(len(chunks), 3)
        event = json.loads(captured.records[-1].getMessage())
        self.assertEqual(event["event"], "provider_response_body_completed")
        self.assertEqual(event["trace_id"], "a" * 32)
        self.assertEqual(event["transport_chunk_count"], 3)
        self.assertEqual(event["sse_json_event_count"], 2)
        self.assertEqual(event["content_chars"], len("answer"))
        self.assertEqual(event["reasoning_chars"], len("think"))
        self.assertEqual(event["finish_reason"], "stop")
        self.assertTrue(event["done_marker_seen"])
        self.assertTrue(event["transport_eof"])

    def test_streaming_api_route_sends_stable_session_and_app_user_agent(self):
        captured: list[httpx.Request] = []
        captured_payloads: list[dict] = []

        def sse(payload: dict) -> httpx.Response:
            chunk = {"id": "chatcmpl-test", "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": payload, "finish_reason": None}]}
            final = {"id": "chatcmpl-test", "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            body = "data: " + json.dumps(chunk) + "\n\n"
            body += "data: " + json.dumps(final) + "\n\n"
            body += "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

        async def handle(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            payload = json.loads(request.content)
            captured_payloads.append(payload)
            history = payload.get("messages", [])
            has_search_prompt = any(
                message.get("role") == "user"
                and "latest Dota patch" in str(message.get("content", ""))
                for message in history
            )
            if payload.get("tools"):
                has_search_result = any(
                    message.get("role") == "tool"
                    and message.get("tool_call_id") == "call-search"
                    for message in history
                )
                if has_search_prompt and not has_search_result:
                    return sse({"tool_calls": [{
                        "index": 0,
                        "id": "call-search",
                        "type": "function",
                        "function": {"name": "web_search",
                                     "arguments": json.dumps({"query": "latest Dota patch"})},
                    }]})
                return sse({"tool_calls": [{
                    "index": 0,
                    "id": "call-finish",
                    "type": "function",
                    "function": {"name": "begin_final_answer", "arguments": "{}"},
                }]})
            return sse({"content": "Axe обычно играет на позиции оффлейнера."})

        class Search:
            async def tool_payload(self, query: str) -> str:
                return json.dumps({"query": query, "results": [{
                    "title": "Dota patch notes", "url": "https://example.com/patch",
                    "snippet": "Patch summary",
                }]})

        async_client_class = httpx.AsyncClient
        original_init = async_client_class.__init__
        transport = httpx.MockTransport(handle)
        def init_with_transport(client, *args, **kwargs):
            original_init(client, *args, transport=transport, **kwargs)

        with patch.object(async_client_class, "__init__", init_with_transport):
            provider = OpenCodeLangChainProvider(api_key="test-key", search=Search())
        service = LLMService(provider, mode="remote")
        app = create_app(service)

        def post_turn(client: TestClient, session_id: str, turn_id: str, text: str) -> list[dict]:
            response = client.post(
                f"/v1/sessions/{session_id}/turns/{turn_id}/stream", json={"text": text}
            )
            self.assertEqual(response.status_code, 200, response.text)
            return [json.loads(line) for line in response.iter_lines() if line]

        session_a, session_b, session_c = "a" * 32, "b" * 32, "c" * 32
        with self.assertLogs("el_voice.remote_provider", level="INFO") as provider_log:
            with TestClient(app) as client:
                first = post_turn(client, session_a, "1" * 32, "What role does Axe play?")
                second = post_turn(client, session_a, "2" * 32, "And what does that role do?")
                third = post_turn(client, session_b, "3" * 32, "Describe Axe briefly.")
                searched = post_turn(client, session_c, "4" * 32, "Find the latest Dota patch notes.")

        self.assertEqual(first[-1]["type"], "done")
        self.assertEqual(second[-1]["type"], "done")
        self.assertEqual(third[-1]["type"], "done")
        self.assertEqual([event["type"] for event in searched], ["tool_call", "sources", "delta", "done"])
        self.assertEqual(len(captured), 9, "planning, optional search, and speech share one agent")
        expected = [session_a, session_a, session_a, session_a, session_b, session_b, session_c, session_c, session_c]
        actual = [request.headers.get("x-opencode-session", "") for request in captured]
        self.assertEqual(actual, ["ses_" + session_id for session_id in expected])
        self.assertTrue(all(request.headers.get("user-agent") == "EL-Voice/0.1" for request in captured))
        self.assertTrue(all(payload.get("thinking") == {"type": "enabled"}
                            and payload.get("reasoning_effort") == "high"
                            for payload in captured_payloads))
        self.assertTrue(all(payload.get("max_tokens", payload.get("max_completion_tokens")) == 256
                            for payload in captured_payloads),
                        [sorted(payload) for payload in captured_payloads])
        log_events = [json.loads(record.getMessage()) for record in provider_log.records]
        responses = [event for event in log_events if event["event"] == "remote_provider_http_response"]
        self.assertEqual(len(responses), 9)
        self.assertTrue(all(event["provider_host"] == "opencode.ai" for event in responses))
        self.assertTrue(all(event["model"] == "deepseek-v4.1-flash" for event in responses))
        self.assertTrue(all(event["status_code"] == 200 for event in responses))
        self.assertTrue(all(event["session_header_present"] for event in responses))
        tool_events = [event for event in log_events if event["event"] == "agent_tool_result"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["tool_call_id"], "call-search")
        invocations = [event for event in log_events if event["event"] == "tool_call_started"]
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0]["tool_call_id"], "call-search")
        self.assertEqual(invocations[0]["trace_id"], "4" * 32)
        self.assertEqual(tool_events[0]["result_count"], 1)
        self.assertEqual(tool_events[0]["trace_id"], "4" * 32)
        self.assertEqual(tool_events[0]["turn_id"], "4" * 32)
        self.assertIn("timestamp", tool_events[0])
        per_turn = [event for event in responses if event["turn_id"] == "4" * 32]
        self.assertEqual([event["request_index"] for event in per_turn], [1, 2, 3])
        self.assertNotIn("test-key", " ".join(record.getMessage() for record in provider_log.records))
        self.assertNotIn("Find the latest Dota patch notes", " ".join(
            record.getMessage() for record in provider_log.records
        ))


if __name__ == "__main__":
    unittest.main()
