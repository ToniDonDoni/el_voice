"""Security checks for local speaker controls over the voice WebSocket."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from starlette.datastructures import Headers

import app


class SpeakerWebSocketOriginTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_and_unexpected_origins_are_rejected_before_accept(self):
        for origin in (None, "https://attacker.example"):
            with self.subTest(origin=origin):
                closed = {}

                async def close(*, code, reason):
                    closed.update(code=code, reason=reason)

                headers = Headers({} if origin is None else {"origin": origin})
                websocket = SimpleNamespace(headers=headers, close=close)
                with patch.dict("os.environ", {"EL_VOICE_ALLOWED_ORIGINS": "http://localhost:8765"}):
                    await app.voice_socket(websocket)

                self.assertEqual(closed.get("code"), 1008)
                self.assertEqual(closed.get("reason"), "Unexpected WebSocket origin")


if __name__ == "__main__":
    unittest.main()
