"""Voice HTTP client integration against the internal FastAPI service boundary."""

from __future__ import annotations

import asyncio
import json
import threading
import unittest

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from llm_client import LLMServiceClient, LLMServiceError
from llm_service.app import LLMService, create_app


@unittest.skipUnless(HTTPX_AVAILABLE, "Install the pinned HTTPX dependency to run client integration tests")
class LLMClientIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_streams_through_service_api_and_clears_context(self):
        class Provider:
            async def stream(self, text, _history, _cancelled, session_id="", turn_id=""):
                yield {"type": "delta", "text": f"answer: {text}"}

        service = LLMService(Provider(), mode="remote")
        transport = httpx.ASGITransport(app=create_app(service))
        client = LLMServiceClient("http://llm", transport=transport)
        session_id = "a" * 32
        events = [event async for event in client.stream(
            session_id, "b" * 32, "test question", threading.Event()
        )]

        self.assertEqual([event["type"] for event in events], ["delta"])
        self.assertEqual(events[0]["text"], "answer: test question")
        self.assertEqual(len(service.contexts.sessions[session_id].history), 2)
        await client.clear_session(session_id)
        self.assertNotIn(session_id, service.contexts.sessions)

    async def test_client_rejects_eof_without_done_as_a_truncated_stream(self):
        transport = httpx.MockTransport(lambda _request: httpx.Response(
            200, text='{"type":"delta","text":"partial"}\n'
        ))
        client = LLMServiceClient("http://llm", transport=transport)
        with self.assertRaisesRegex(LLMServiceError, "truncated_llm_service_stream"):
            _ = [event async for event in client.stream(
                "a" * 32, "b" * 32, "test question", threading.Event()
            )]

    async def test_client_receives_empty_generation_fallback_with_trace_id(self):
        class EmptyProvider:
            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                if False:
                    yield None

        session_id, turn_id = "c" * 32, "d" * 32
        service = LLMService(EmptyProvider(), mode="remote")
        transport = httpx.ASGITransport(app=create_app(service))
        client = LLMServiceClient("http://llm", transport=transport)
        with self.assertLogs("el_voice.llm_client", level="INFO") as captured:
            received = [event async for event in client.stream(
                session_id, turn_id, "question", threading.Event()
            )]
        self.assertEqual([event["type"] for event in received], ["replace"])
        self.assertEqual(received[0]["text"], "Я так качался")

        events = [json.loads(record.getMessage()) for record in captured.records]
        fallback = next(event for event in events
                        if event["event"] == "llm_client_event_yielded"
                        and event["event_type"] == "replace")
        self.assertEqual(fallback["trace_id"], turn_id)

    async def test_consumer_cancellation_closes_owned_read_and_notifies_service(self):
        class BlockingStream(httpx.AsyncByteStream):
            def __init__(self):
                self.read_started = asyncio.Event()
                self.read_cancelled = asyncio.Event()
                self.response_closed = asyncio.Event()
                self.read_task = None

            async def __aiter__(self):
                self.read_task = asyncio.current_task()
                self.read_started.set()
                try:
                    await asyncio.Event().wait()
                    yield b""
                except asyncio.CancelledError:
                    self.read_cancelled.set()
                    raise

            async def aclose(self):
                self.response_closed.set()

        stream = BlockingStream()
        cancel_event = threading.Event()
        cancel_requested = asyncio.Event()

        def handler(request):
            if request.url.path.endswith("/cancel"):
                cancel_requested.set()
                return httpx.Response(202, json={"cancelled": True})
            return httpx.Response(200, stream=stream)

        client = LLMServiceClient("http://llm", transport=httpx.MockTransport(handler))

        async def consume():
            async for _event in client.stream("a" * 32, "b" * 32, "hello", cancel_event):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(stream.read_started.wait(), timeout=1)
        cancel_event.set()
        consumer.cancel()
        await asyncio.wait_for(asyncio.gather(consumer, return_exceptions=True), timeout=1)

        self.assertTrue(stream.read_cancelled.is_set())
        self.assertTrue(stream.response_closed.is_set())
        self.assertTrue(cancel_requested.is_set())
        self.assertTrue(stream.read_task.done())
