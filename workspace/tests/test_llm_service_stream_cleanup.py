"""Regression for closing provider streams in their owning asyncio context."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import gc
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from llm_service.app import LLMService, TurnRequest, create_app
from llm_service.remote_provider import OpenCodeLangChainProvider


try:
    from langchain_core.messages import AIMessageChunk
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@unittest.skipUnless(LANGCHAIN_AVAILABLE, "Install the pinned LangChain dependencies")
class LLMProviderStreamCleanupTests(IsolatedAsyncioTestCase):
    async def test_cancellation_closes_provider_iterator_before_leaving_producer_task(self):
        class FastModel:
            async def astream(self, _messages):
                for _ in range(100):
                    yield AIMessageChunk(content="short answer chunk")

        provider = OpenCodeLangChainProvider.__new__(OpenCodeLangChainProvider)
        provider._session_header = ContextVar("test_opencode_session", default="")
        provider.model = FastModel()
        provider.close = AsyncMock()
        service = LLMService(provider, mode="remote")
        app = create_app(service)
        session_id, turn_id = "d" * 32, "e" * 32
        errors = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))

        try:
            async with app.router.lifespan_context(app):
                route = next(route for route in app.routes
                             if getattr(route, "path", None) == "/v1/sessions/{session_id}/turns/{turn_id}/stream")
                response = await route.endpoint(
                    session_id, turn_id, TurnRequest(text="Give a short answer")
                )
                body = response.body_iterator
                first_line = await anext(body)
                self.assertIn(b'"type":"delta"', first_line)

                # Leave the consumer idle until the bounded queue fills and the
                # producer is suspended while it still owns provider.stream().
                await asyncio.sleep(0.02)
                await service.clear_session(session_id)
                await body.aclose()
                del body, response
                gc.collect()
                await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        cleanup_errors = [context.get("exception") for context in errors]
        self.assertFalse(
            any(isinstance(error, ValueError) and "different Context" in str(error)
                for error in cleanup_errors),
            cleanup_errors,
        )


if __name__ == "__main__":
    unittest.main()
