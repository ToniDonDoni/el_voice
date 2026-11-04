"""Bounded, in-memory conversation context for the LLM service."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import copy
import time
from typing import Any


def token_count(text: str) -> int:
    """Count context tokens with a local tokenizer and a deterministic fallback."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


@dataclass
class SessionContext:
    history: list[dict[str, str]] = field(default_factory=list)
    lock: Any = None
    active_turn_id: str | None = None
    cancel_event: Any = None
    active_task: Any = None
    touched_at: float = field(default_factory=time.monotonic)
    reserved: bool = False


class ConversationStore:
    """Own short-lived session histories; never persist tool traces or user data."""

    def __init__(self, token_budget: int = 2400, max_sessions: int = 64):
        self.token_budget = token_budget
        self.max_sessions = max_sessions
        self.sessions: dict[str, SessionContext] = {}

    def get(self, session_id: str) -> SessionContext:
        import asyncio

        state = self.sessions.get(session_id)
        if state is None:
            if len(self.sessions) >= self.max_sessions:
                idle = [
                    key for key, candidate in self.sessions.items()
                    if not candidate.reserved
                    and (candidate.active_task is None or candidate.active_task.done())
                ]
                if not idle:
                    raise OverflowError("The LLM service has reached its active session limit")
                oldest_id = min(idle, key=lambda key: self.sessions[key].touched_at)
                self.sessions.pop(oldest_id)
            state = SessionContext(lock=asyncio.Lock())
            self.sessions[session_id] = state
        state.touched_at = time.monotonic()
        return state

    def trimmed_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Keep complete user/assistant pairs within the configured token budget."""
        pairs = []
        for index in range(0, len(history) - 1, 2):
            user, assistant = history[index:index + 2]
            if user.get("role") == "user" and assistant.get("role") == "assistant":
                pairs.append((user, assistant))
        kept: list[dict[str, str]] = []
        total = 0
        for user, assistant in reversed(pairs):
            cost = token_count(user["content"]) + token_count(assistant["content"])
            if total + cost > self.token_budget:
                break
            kept[0:0] = [user, assistant]
            total += cost
        return kept

    def remember(self, state: SessionContext, user_text: str, assistant_text: str) -> None:
        state.history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        state.history = self.trimmed_history(state.history)
        state.touched_at = time.monotonic()

    def clear(self, session_id: str) -> SessionContext | None:
        return self.sessions.pop(session_id, None)


def paired_tool_messages(messages: list[Any]) -> list[Any]:
    """Keep complete tool calls/results and remove orphan IDs from mixed messages."""
    tool_calls: dict[str, Any] = {}
    tool_results: dict[str, Any] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            if call_id:
                tool_calls[call_id] = message
        call_id = getattr(message, "tool_call_id", None)
        if call_id:
            tool_results[call_id] = message
    complete_ids = tool_calls.keys() & tool_results.keys()
    if not complete_ids:
        return []

    paired = []
    for message in messages:
        calls = list(getattr(message, "tool_calls", []) or [])
        if calls:
            valid_calls = [
                call for call in calls
                if (call.get("id") if isinstance(call, dict) else getattr(call, "id", None)) in complete_ids
            ]
            if not valid_calls:
                continue
            if len(valid_calls) == len(calls):
                paired.append(message)
                continue
            updates: dict[str, Any] = {"tool_calls": valid_calls}
            chunks = getattr(message, "tool_call_chunks", None)
            if chunks:
                updates["tool_call_chunks"] = [
                    chunk for chunk in chunks if chunk.get("id") in complete_ids
                ]
            additional = getattr(message, "additional_kwargs", None)
            if isinstance(additional, dict) and isinstance(additional.get("tool_calls"), list):
                updates["additional_kwargs"] = {
                    **additional,
                    "tool_calls": [item for item in additional["tool_calls"]
                                   if item.get("id") in complete_ids],
                }
            clone = message.model_copy(update=updates) if hasattr(message, "model_copy") else copy(message)
            if clone is message:
                raise ValueError("Could not remove unmatched tool calls")
            for name, value in updates.items():
                if not hasattr(message, "model_copy"):
                    setattr(clone, name, value)
            paired.append(clone)
            continue

        call_id = getattr(message, "tool_call_id", None)
        if call_id in complete_ids:
            paired.append(message)
    return paired
