"""Bounded conversation context shared by the voice pipeline and LLM service.

Historically each turn was sent to the model in isolation. The review follow-up
requires every message to be processed in one shared conversation context: the
system prompt, a bounded number of previous question/answer pairs, and the
current user utterance.

Only whole pairs are kept; the oldest pair is dropped first when the character
budget is exceeded. Nothing here persists: the history lives in memory for the
duration of a session and is cleared on stop.
"""

from __future__ import annotations

DEFAULT_HISTORY_TURNS = 4
DEFAULT_MAX_PROMPT_CHARS = 4000


def _pair(history: list[dict]) -> list[list[dict]]:
    pairs: list[list[dict]] = []
    index = 0
    while index + 1 < len(history):
        first, second = history[index], history[index + 1]
        if first.get("role") == "user" and second.get("role") == "assistant":
            pairs.append([first, second])
            index += 2
        else:  # tolerate malformed history: skip one item, never split a pair
            index += 1
    return pairs


def build_messages(system_prompt: str, history: list[dict], user_text: str,
                   max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> tuple[list[dict], dict]:
    """Build the model prompt and content-free statistics for one turn."""
    budget = max(0, int(max_prompt_chars) - len(system_prompt) - len(user_text))
    selected: list[list[dict]] = []
    for pair in reversed(_pair(history)):
        cost = sum(len(item.get("content", "")) for item in pair)
        if cost > budget:
            break
        selected.insert(0, pair)
        budget -= cost

    messages = [{"role": "system", "content": system_prompt}]
    for pair in selected:
        messages.extend({"role": item["role"], "content": item["content"]} for item in pair)
    messages.append({"role": "user", "content": user_text})
    stats = {
        "history_turns": len(selected),
        "chars": sum(len(item["content"]) for item in messages),
    }
    return messages, stats


class ConversationHistory:
    """In-memory question/answer history with a hard pair limit."""

    def __init__(self, turns_limit: int = DEFAULT_HISTORY_TURNS):
        self._limit = max(0, int(turns_limit))
        self._items: list[dict] = []

    def reset(self) -> None:
        self._items.clear()

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        if self._limit == 0 or not assistant_text.strip():
            return
        self._items.append({"role": "user", "content": user_text})
        self._items.append({"role": "assistant", "content": assistant_text})
        while len(self._items) > self._limit * 2:
            del self._items[:2]

    def as_list(self) -> list[dict]:
        return [dict(item) for item in self._items]

    def __len__(self) -> int:
        return len(self._items) // 2
