"""Deterministic tests for the shared LLM service and Remote Voice contract."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from llm_service.app import LLMService, TurnRequest, create_app, safe_error_code
from llm_service.context import ConversationStore, paired_tool_messages
from llm_service.local_provider import LocalGemmaProvider, MODEL_RELATIVE_PATH
from llm_service.remote_provider import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ProviderConfigurationError,
    validate_provider_config,
)
from llm_service.local_provider import CancelOnToken
from llm_service.search import WebSearch, public_https_url


class ToolCallMessage:
    def __init__(self, calls):
        self.tool_calls = calls


class ToolResultMessage:
    def __init__(self, call_id):
        self.tool_call_id = call_id


class ContextTests(unittest.TestCase):
    def test_context_budget_keeps_recent_complete_user_assistant_pairs(self):
        store = ConversationStore(token_budget=20)
        state = store.get("a" * 32)
        store.remember(state, "old " * 20, "old answer " * 20)
        store.remember(state, "middle " * 20, "middle answer " * 20)
        store.remember(state, "new", "new answer")
        self.assertEqual(state.history, [
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "new answer"},
        ])

    def test_tool_call_trimming_keeps_only_id_matched_pairs(self):
        call = ToolCallMessage([
            {"id": "call-good", "name": "web_search"},
            {"id": "call-orphan", "name": "web_search"},
        ])
        good = ToolResultMessage("call-good")
        wrong = ToolResultMessage("unmatched")
        paired = paired_tool_messages([call, good, wrong])
        self.assertEqual([item["id"] for item in paired[0].tool_calls], ["call-good"])
        self.assertEqual([item.tool_call_id for item in paired[1:]], ["call-good"])

    def test_session_clear_cancels_pending_inference(self):
        async def run():
            service = LLMService(provider=object(), mode="local")
            state = service.contexts.get("a" * 32)
            finished = asyncio.Event()
            started = asyncio.Event()

            async def pending():
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()

            state.active_task = asyncio.create_task(pending())
            state.cancel_event = asyncio.Event()
            await started.wait()
            await service.clear_session("a" * 32)
            self.assertTrue(finished.is_set())
            self.assertTrue(state.cancel_event.is_set())
            self.assertNotIn("a" * 32, service.contexts.sessions)

        asyncio.run(run())

    def test_token_cancellation_forces_llama_cpp_eos(self):
        cancelled = threading.Event()
        processor = CancelOnToken(cancelled, eos_token=2)
        scores = np.asarray([0.0, 1.0, 2.0, 3.0])
        processor(None, scores)
        self.assertEqual(scores.tolist(), [0.0, 1.0, 2.0, 3.0])
        cancelled.set()
        processor(None, scores)
        self.assertTrue(np.isneginf(scores[[0, 1, 3]]).all())
        self.assertEqual(scores[2], 0.0)

    def test_session_limit_does_not_evict_an_active_context(self):
        store = ConversationStore(max_sessions=1)
        active = store.get("a" * 32)

        class RunningTask:
            def done(self):
                return False

            def cancel(self):
                pass

        active.active_task = RunningTask()
        with self.assertRaises(OverflowError):
            store.get("b" * 32)
        self.assertIn("a" * 32, store.sessions)

    def test_local_gemma_verifies_the_manifest_asset_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / MODEL_RELATIVE_PATH
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"test gguf placeholder")
            llama = Mock()
            with patch.dict(os.environ, {"EL_VOICE_MODELS": directory}):
                with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=llama)}):
                    with patch("llm_service.local_provider.subprocess.run", return_value=SimpleNamespace(returncode=0)) as verify:
                        LocalGemmaProvider(model_path=model_path)
            self.assertEqual(verify.call_args.args[0][-3:], ["--verify", "--only", MODEL_RELATIVE_PATH])
            llama.assert_called_once()

    def test_local_gemma_refuses_to_load_an_unverified_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / MODEL_RELATIVE_PATH
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"test gguf placeholder")
            llama = Mock()
            with patch.dict(os.environ, {"EL_VOICE_MODELS": directory}):
                with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=llama)}):
                    with patch("llm_service.local_provider.subprocess.run", return_value=SimpleNamespace(returncode=1)):
                        with self.assertRaisesRegex(RuntimeError, "verification failed"):
                            LocalGemmaProvider(model_path=model_path)
            llama.assert_not_called()


class RemoteConfigurationTests(unittest.TestCase):
    def test_provider_configuration_is_explicit_and_redacted(self):
        validate_provider_config("test-key", DEFAULT_BASE_URL, DEFAULT_MODEL)
        validate_provider_config(
            "test-key", "https://opencode.ai/inference/go/openai/v1", DEFAULT_MODEL
        )
        with self.assertRaises(ProviderConfigurationError):
            validate_provider_config(None, DEFAULT_BASE_URL, DEFAULT_MODEL)
        with self.assertRaises(ProviderConfigurationError):
            validate_provider_config("test-key", "http://opencode.ai/zen/go/v1", DEFAULT_MODEL)
        with self.assertRaises(ProviderConfigurationError):
            validate_provider_config("test-key", DEFAULT_BASE_URL, "other-model")
        self.assertEqual(safe_error_code(RuntimeError("test-key")), "generation_failed")

    def test_missing_remote_key_keeps_service_unready_with_actionable_redacted_error(self):
        with patch.dict(os.environ, {"EL_VOICE_MODE": "remote", "OPENCODE_GO_API_KEY": ""}):
            with TestClient(create_app()) as client:
                response = client.get("/healthz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "provider_configuration_error")
        self.assertNotIn("OPENCODE_GO_API_KEY=", response.text)



class SearchToolTests(unittest.IsolatedAsyncioTestCase):
    def test_search_limits_accept_the_requested_timeout_and_snippet_size(self):
        with patch.dict(os.environ, {
            "WEB_SEARCH_TIMEOUT_SECONDS": "21",
            "WEB_SEARCH_MAX_SNIPPET_CHARS": "9999",
        }):
            search = WebSearch(backend="duckduckgo")
        self.assertEqual(search.timeout_seconds, 21)
        self.assertEqual(search.max_snippet_chars, 9999)

    def test_result_url_filter_blocks_private_targets(self):
        self.assertTrue(public_https_url("https://example.com/article"))
        self.assertFalse(public_https_url("http://example.com/article"))
        self.assertFalse(public_https_url("https://127.0.0.1/admin"))
        self.assertFalse(public_https_url("https://localhost/admin"))
        self.assertFalse(public_https_url("https://router.local/admin"))

    async def test_tool_result_count_snippets_and_total_size_are_bounded(self):
        search = WebSearch(backend="duckduckgo", max_results=50, max_snippet_chars=500)
        async def results(_query):
            return [
            {"title": "x" * 300, "url": f"https://example.com/{index}", "snippet": "s" * 1000}
            for index in range(20)
            ]
        search._run_worker = results
        payload = await search.tool_payload("latest Dota patch")
        parsed = json.loads(payload)
        self.assertLessEqual(len(parsed["results"]), 5)
        self.assertLessEqual(len(payload), 6000)
        self.assertTrue(all(len(result["snippet"]) <= 500 for result in parsed["results"]))

    async def test_large_configured_snippet_fits_the_bounded_tool_payload(self):
        search = WebSearch(backend="duckduckgo", max_snippet_chars=9999)
        async def results(_query):
            return [{"title": "guide", "url": "https://example.com/guide", "snippet": "s" * 9999}]
        search._run_worker = results
        parsed = json.loads(await search.tool_payload("find the guide"))
        self.assertEqual(len(parsed["results"][0]["snippet"]), 9999)

    async def test_search_timeouts_and_backend_configuration_are_explicit(self):
        search = WebSearch(backend="duckduckgo")
        search.timeout_seconds = 0.01

        async def delayed(_query):
            await asyncio.sleep(0.05)
            return []

        search._run_worker = delayed
        with self.assertRaisesRegex(Exception, "search_timeout"):
            await search.search("a current fact")
        with self.assertRaises(ValueError):
            WebSearch(backend="http://127.0.0.1")

    async def test_cancel_search_terminates_the_owned_worker_process(self):
        class Process:
            def __init__(self):
                self.returncode = None
                self.terminated = False

            async def communicate(self, input=None):
                await asyncio.Event().wait()

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            async def wait(self):
                return self.returncode

            def kill(self):
                self.returncode = -9

        search = WebSearch()
        process = Process()
        spawned = asyncio.Event()

        async def spawn():
            spawned.set()
            return process

        search._spawn_worker = spawn
        task = asyncio.create_task(search.search("current game patch"))
        await spawned.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(process.terminated)


class StreamingContractTests(unittest.TestCase):
    def test_streaming_context_isolated_sources_and_history_lifecycle(self):
        class FakeProvider:
            def __init__(self):
                self.histories = []

            async def stream(self, text, history, cancelled, session_id="", turn_id=""):
                self.histories.append(list(history))
                if "search" in text.lower():
                    yield {"type": "tool_call", "name": "web_search", "result_count": 1}
                    yield {"type": "sources", "sources": [{
                        "title": "Patch notes", "url": "https://example.com/patch", "snippet": "Current patch"
                    }]}
                yield {"type": "delta", "text": f"answer to {text}"}

            async def close(self):
                pass

        provider = FakeProvider()
        service = LLMService(provider, mode="remote", token_budget=2400)
        app = create_app(service)
        session_id = uuid.uuid4().hex
        other_session = uuid.uuid4().hex
        turn_id = uuid.uuid4().hex
        with TestClient(app) as client:
            health = client.get("/healthz")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["ready"])
            first = client.post(
                f"/v1/sessions/{session_id}/turns/{turn_id}/stream", json={"text": "search latest"}
            )
            events = [json.loads(line) for line in first.iter_lines() if line]
            self.assertEqual([item["type"] for item in events], ["tool_call", "sources", "delta", "done"])
            self.assertEqual(events[1]["sources"][0]["title"], "Patch notes")

            second = client.post(
                f"/v1/sessions/{session_id}/turns/{uuid.uuid4().hex}/stream", json={"text": "follow up"}
            )
            self.assertEqual(second.status_code, 200)
            self.assertEqual(provider.histories[1], [
                {"role": "user", "content": "search latest"},
                {"role": "assistant", "content": "answer to search latest"},
            ])

            isolated = client.post(
                f"/v1/sessions/{other_session}/turns/{uuid.uuid4().hex}/stream", json={"text": "separate"}
            )
            self.assertEqual(isolated.status_code, 200)
            self.assertEqual(provider.histories[2], [])
            self.assertEqual(client.delete(f"/v1/sessions/{session_id}").status_code, 204)
            self.assertNotIn(session_id, service.contexts.sessions)

    def test_provider_failure_replaces_partial_text_with_fallback(self):
        class FailingProvider:
            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                yield {"type": "delta", "text": "partial"}
                raise RuntimeError("provider_rate_limited")

            async def close(self):
                pass

        service = LLMService(FailingProvider(), mode="remote")
        with TestClient(create_app(service)) as client:
            session_id = uuid.uuid4().hex
            response = client.post(
                f"/v1/sessions/{session_id}/turns/{uuid.uuid4().hex}/stream", json={"text": "question"}
            )
            events = [json.loads(line) for line in response.iter_lines() if line]
            self.assertEqual([event["type"] for event in events], ["delta", "replace", "done"])
            self.assertEqual(events[1]["text"], "Я так качался")
            self.assertEqual(events[1]["fallback_reason"], "provider_rate_limited")
            self.assertNotIn("test-key", response.text)
            self.assertEqual(service.contexts.sessions[session_id].history[-1]["content"], "Я так качался")

    def test_remote_provider_timeout_without_text_emits_configured_fallback(self):
        class TimedOutProvider:
            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                raise RuntimeError("provider_timeout")
                yield

            async def close(self):
                pass

        service = LLMService(TimedOutProvider(), mode="remote")
        session_id, turn_id = uuid.uuid4().hex, uuid.uuid4().hex
        with TestClient(create_app(service)) as client:
            response = client.post(
                f"/v1/sessions/{session_id}/turns/{turn_id}/stream",
                json={"text": "question"},
            )
        events = [json.loads(line) for line in response.iter_lines() if line]
        self.assertEqual([event["type"] for event in events], ["replace", "done"])
        self.assertEqual(events[0]["text"], "Я так качался")
        self.assertEqual(events[0]["trace_id"], turn_id)

    def test_remote_provider_timeout_after_partial_text_replaces_partial_answer(self):
        class TimedOutProvider:
            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                yield {"type": "delta", "text": "partial"}
                raise RuntimeError("provider_timeout")

            async def close(self):
                pass

        service = LLMService(TimedOutProvider(), mode="remote")
        session_id, turn_id = uuid.uuid4().hex, uuid.uuid4().hex
        with TestClient(create_app(service)) as client:
            response = client.post(
                f"/v1/sessions/{session_id}/turns/{turn_id}/stream",
                json={"text": "question"},
            )
        events = [json.loads(line) for line in response.iter_lines() if line]
        self.assertEqual(events[0]["text"], "partial")
        self.assertEqual([event["type"] for event in events], ["delta", "replace", "done"])
        self.assertEqual(events[1]["text"], "Я так качался")

    def test_empty_provider_output_emits_fallback_answer(self):
        class EmptyProvider:
            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                if False:
                    yield None

            async def close(self):
                pass

        service = LLMService(EmptyProvider(), mode="remote")
        session_id, turn_id = uuid.uuid4().hex, uuid.uuid4().hex
        with TestClient(create_app(service)) as client:
            response = client.post(
                f"/v1/sessions/{session_id}/turns/{turn_id}/stream",
                json={"text": "question"},
            )
        events = [json.loads(line) for line in response.iter_lines() if line]
        self.assertEqual([event["type"] for event in events], ["replace", "done"])
        self.assertEqual(events[0]["text"], "Я так качался")
        self.assertEqual(events[0]["trace_id"], turn_id)
        self.assertIn("timestamp", events[0])

    def test_cancel_http_route_stops_work_and_discards_partial_history(self):
        class BlockingProvider:
            def __init__(self):
                self.started = threading.Event()
                self.cancelled = threading.Event()

            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
                yield {"type": "delta", "text": "unreachable"}

            async def close(self):
                pass

        provider = BlockingProvider()
        service = LLMService(provider, mode="remote")
        session_id, turn_id = uuid.uuid4().hex, uuid.uuid4().hex
        with TestClient(create_app(service)) as client:
            result = []

            def stream_request():
                result.append(client.post(
                    f"/v1/sessions/{session_id}/turns/{turn_id}/stream", json={"text": "cancel me"}
                ))

            request_thread = threading.Thread(target=stream_request)
            request_thread.start()
            self.assertTrue(provider.started.wait(2))
            cancelled = client.post(f"/v1/sessions/{session_id}/turns/{turn_id}/cancel")
            request_thread.join(2)
            self.assertFalse(request_thread.is_alive())
            self.assertEqual(cancelled.status_code, 202)
            self.assertTrue(provider.cancelled.wait(1))
            self.assertEqual(service.contexts.sessions[session_id].history, [])


    def test_active_session_limit_is_an_http_error(self):
        class Provider:
            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                yield {"type": "delta", "text": "ok"}

            async def close(self):
                pass

        service = LLMService(Provider(), mode="remote", max_sessions=1)
        active = service.contexts.get("a" * 32)

        class RunningTask:
            def done(self):
                return False

            def cancel(self):
                pass

        active.active_task = RunningTask()
        with TestClient(create_app(service)) as client:
            response = client.post(
                f"/v1/sessions/{'b' * 32}/turns/{uuid.uuid4().hex}/stream", json={"text": "hello"}
            )
            self.assertEqual(response.status_code, 429)
            self.assertIn("a" * 32, service.contexts.sessions)
            active.active_task = None

    def test_cold_burst_stream_keeps_terminal_done_after_slow_context_save(self):
        class BurstProvider:
            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                for _ in range(14):
                    yield {"type": "delta", "text": "chunk"}

            async def close(self):
                pass

        service = LLMService(BurstProvider(), mode="remote")
        original_remember = service.contexts.remember

        def slow_remember(*args, **kwargs):
            import time
            time.sleep(0.15)
            return original_remember(*args, **kwargs)

        service.contexts.remember = slow_remember
        application = create_app(service)
        endpoint = next(route.endpoint for route in application.routes
                        if getattr(route, "path", "").endswith("/stream"))

        async def consume():
            response = await endpoint("c" * 32, "d" * 32, TurnRequest(text="cold"))
            body = response.body_iterator
            try:
                return [json.loads(line) async for line in body]
            finally:
                await body.aclose()

        events = asyncio.run(consume())
        self.assertEqual(sum(item["type"] == "delta" for item in events), 14)
        self.assertEqual([item["type"] for item in events].count("done"), 1)
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(all(item["turn_id"] == "d" * 32 for item in events))

    def test_stream_reaches_eof_after_backpressured_burst(self):
        class BurstProvider:
            async def stream(self, _text, _history, _cancelled, session_id="", turn_id=""):
                for index in range(20):
                    yield {"type": "delta", "text": str(index)}

            async def close(self):
                pass

        service = LLMService(BurstProvider(), mode="remote")
        application = create_app(service)
        endpoint = next(route.endpoint for route in application.routes
                        if getattr(route, "path", "").endswith("/stream"))

        async def consume():
            response = await endpoint("a" * 32, "b" * 32, TurnRequest(text="burst"))
            body = response.body_iterator
            items = [await anext(body)]
            await asyncio.sleep(0.05)
            try:
                while True:
                    items.append(await asyncio.wait_for(anext(body), timeout=0.3))
                    await asyncio.sleep(0.005)
            except StopAsyncIteration:
                return [json.loads(item) for item in items]
            finally:
                await body.aclose()

        events = asyncio.run(consume())
        self.assertEqual(sum(event["type"] == "delta" for event in events), 20)
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("timestamp", events[-1])
        self.assertIn("trace_id", events[-1])


if __name__ == "__main__":
    unittest.main()
