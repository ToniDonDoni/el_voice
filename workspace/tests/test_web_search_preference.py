"""Voice-session preference boundary for the opt-in remote web-search feature.

SDDTDD SPEC: tasks/SPEC-004-REMOTE-WEB-SEARCH-TOGGLE.md
The browser preference defaults off, can be changed during a session, and is
snapshotted when a voice turn starts so only later turns use the new value.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app
from voice_pipeline import Turn


class WebSearchPreferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_search_setting_defaults_off_and_control_change_is_acknowledged(self):
        session = app.VoiceSession("session", None, None)
        self.assertFalse(session.search_enabled)

        await app.receive_control(session, json.dumps({
            "type": "search_setting", "enabled": True,
        }))

        self.assertTrue(session.search_enabled)
        acknowledgement = session.outgoing.get_nowait().event
        self.assertEqual(acknowledgement, {"type": "search_setting", "enabled": True})

    async def test_turn_uses_its_search_preference_snapshot(self):
        received = []

        class Runtime:
            llm_client = object()

            async def llm_stream(self, _text, _cancelled, _history, *, session_id, turn_id,
                                 search_enabled):
                received.append((session_id, turn_id, search_enabled))
                yield {"type": "delta", "text": "reply"}

        session = app.VoiceSession("session", None, None)
        turn = Turn("b" * 32, search_enabled=True)
        session.current_turn = turn
        session.search_enabled = False  # A later toggle must not alter this turn.
        session.is_current = lambda candidate: candidate is turn

        with patch.object(app, "runtime", Runtime()):
            events = [event async for event in app._llm_items(session, turn, "question")]

        self.assertEqual(events, [{"type": "delta", "text": "reply"}])
        self.assertEqual(received, [("session", "b" * 32, True)])
