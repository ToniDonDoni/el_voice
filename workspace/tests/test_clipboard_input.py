"""Clipboard/text contracts through the voice, internal API, and provider request."""

import asyncio
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import app
from image_attachment import MAX_IMAGE_DATA_CHARS, image_metadata
from image_fixtures import image_url
from llm_client import LLMServiceClient
from llm_service.app import LLMService, create_app
from llm_service.remote_provider import OpenCodeLangChainProvider
from voice_pipeline import Turn


class ClipboardInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_and_image_cross_internal_api_without_image_history_or_payload_logs(self):
        received = []

        class Provider:
            async def stream(self, text, history, cancelled, **options):
                received.append((text, history, options))
                yield {"type": "delta", "text": "A red square and blue circle."}

        service = LLMService(Provider(), "remote")
        client = LLMServiceClient("http://llm", transport=httpx.ASGITransport(app=create_app(service)))
        image = image_url()
        with self.assertLogs("el_voice.llm", "INFO") as captured:
            events = [event async for event in client.stream(
                "a" * 32, "b" * 32, "Describe the screenshot", threading.Event(), image=image)]
        self.assertEqual(events[0]["type"], "delta")
        self.assertEqual(received[0][2]["image"], image)
        self.assertNotIn(image, str(service.contexts.sessions["a" * 32].history))
        logs = " ".join(captured.output)
        self.assertNotIn("base64", logs)
        self.assertIn(image_metadata(image)["sha256"], logs)
        self.assertIn("Describe the screenshot", logs)

    async def test_voice_consumes_image_once_but_empty_recognition_keeps_it(self):
        received = []

        class Runtime:
            llm_client = object()
            transcribe = lambda *_args: ""

            async def llm_stream(self, text, _cancel, _history, **options):
                received.append((text, options.get("image")))
                yield {"type": "delta", "text": "Description."}

        session = app.VoiceSession("a" * 32, None, None, outgoing=asyncio.Queue(64))
        session.pending_image, session.pending_image_id = image_url(), "d" * 32
        runtime = Runtime()
        with patch.object(app, "runtime", runtime), patch.object(app, "synthesize_and_queue", AsyncMock()):
            turn = Turn("b" * 32)
            session.current_turn = turn
            await app.process_turn(session, turn, np.zeros(512))
            self.assertIsNotNone(session.pending_image)
            runtime.transcribe = lambda *_args: "What is here?"
            turn = Turn("c" * 32)
            session.current_turn = turn
            await app.process_turn(session, turn, np.zeros(512))
            self.assertIsNone(session.pending_image)
            self.assertIsNone(turn.image)
            turn = Turn("e" * 32)
            session.current_turn = turn
            await app.process_turn(session, turn, np.zeros(512))
        self.assertEqual(received, [("What is here?", image_url()), ("What is here?", None)])

    async def test_typed_send_interrupts_current_reply_and_uses_pending_image(self):
        received = []
        runtime = SimpleNamespace(mode="remote", new_vad=lambda: object(), llm_client=object())

        async def stream(text, _cancel, _history, **options):
            received.append((text, options.get("image")))
            yield {"type": "delta", "text": "Description."}

        runtime.llm_stream = stream
        session = app.VoiceSession("a" * 32, None, None, outgoing=asyncio.Queue(64))
        previous = Turn("b" * 32)
        session.current_turn = previous
        session.interrupt_enabled = False
        with patch.object(app, "runtime", runtime), patch.object(app, "synthesize_and_queue", AsyncMock()):
            await app.receive_control(session, json.dumps({"type": "image_attachment",
                "attachment_id": "d" * 32, "image": image_url()}))
            await app.receive_control(session, json.dumps({"type": "text_query", "text": ""}))
            await asyncio.gather(*tuple(session.tasks))
        self.assertTrue(previous.cancel_event.is_set())
        self.assertEqual(received, [("Опиши, что на этой картинке.", image_url())])
        self.assertIsNone(session.pending_image)

    async def test_local_and_invalid_inputs_do_not_reach_provider_or_echo_image(self):
        image = image_url()
        session = app.VoiceSession("a" * 32, None, None)
        with patch.object(app, "runtime", SimpleNamespace(mode="local")):
            await app.receive_control(session, json.dumps({"type": "image_attachment",
                "attachment_id": "d" * 32, "image": image}))
        self.assertEqual(session.outgoing.get_nowait().event["type"], "image_error")
        self.assertIsNone(session.pending_image)
        for mode, value, expected in [("local", image, 400), ("remote", image[:-8], 422),
                                      ("remote", "x" * (MAX_IMAGE_DATA_CHARS + 1), 422)]:
            provider = SimpleNamespace(stream=AsyncMock())
            transport = httpx.ASGITransport(app=create_app(LLMService(provider, mode)))
            async with httpx.AsyncClient(transport=transport, base_url="http://llm") as client:
                response = await client.post(f'/v1/sessions/{"a" * 32}/turns/{"b" * 32}/stream',
                                             json={"text": "test", "image": value})
            self.assertEqual(response.status_code, expected)
            self.assertNotIn(value, response.text)
            provider.stream.assert_not_called()

    async def test_local_typed_question_skips_speech_recognition(self):
        received = []

        async def stream(text, _cancel, _history, **options):
            received.append((text, options))
            yield {"type": "delta", "text": "Local answer."}

        runtime = SimpleNamespace(mode="local", new_vad=lambda: object(),
                                  llm_client=object(), llm_stream=stream)
        session = app.VoiceSession("a" * 32, None, None, outgoing=asyncio.Queue(64))
        with patch.object(app, "runtime", runtime), patch.object(app, "synthesize_and_queue", AsyncMock()):
            await app.receive_control(session, json.dumps({"type": "text_query", "text": "Local question"}))
            await asyncio.gather(*tuple(session.tasks))
        self.assertEqual(received[0][0], "Local question")
        self.assertNotIn("image", received[0][1])

    async def test_provider_transport_logs_actual_attachment_metadata_only(self):
        from contextvars import ContextVar
        provider = object.__new__(OpenCodeLangChainProvider)
        for name in ("_session_header", "_trace_id", "_session_id"):
            setattr(provider, name, ContextVar(name, default="a" * 32))
        provider._request_sequence = ContextVar("counter", default={"value": 0})
        provider.model_name = "test"
        request = httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions", json={
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": image_url()}}]}]})
        with self.assertLogs("el_voice.remote_provider", "INFO") as captured:
            await provider._attach_session_header(request)
        event = json.loads(captured.records[0].getMessage())
        self.assertEqual(event["image_count"], 1)
        self.assertEqual(event["images"], [image_metadata(image_url())])
        self.assertNotIn("base64", str(event))


class ImageValidationTests(unittest.TestCase):
    def test_rejects_truncated_corrupt_external_and_oversize_data(self):
        valid = image_url()
        self.assertEqual(image_metadata(valid)["width"], 320)
        for value in [valid[:-8], valid.replace("iVBOR", "zVBOR"), "https://example.com/image.png",
                      "data:image/png;base64,broken", "x" * (MAX_IMAGE_DATA_CHARS + 1)]:
            with self.subTest(value=value[:30]), self.assertRaises(ValueError):
                image_metadata(value)

    def test_rejects_png_with_valid_chunk_crc_but_invalid_pixel_stream(self):
        import base64
        import struct
        import zlib

        def chunk(kind, data):
            return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack("!IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
               + chunk(b"IDAT", b"not a zlib stream")
               + chunk(b"IEND", b""))
        with self.assertRaisesRegex(ValueError, "pixel data"):
            image_metadata("data:image/png;base64," + base64.b64encode(png).decode())
