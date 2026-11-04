"""Prepare streamed language-model text for incremental speech synthesis.

Language models emit prose that is not directly speakable: markdown emphasis,
emoji, URLs and very long sentences. The chunker removes unspeakable
characters and cuts the stream into short phrases that can be synthesised and
played while the rest of the answer is still being generated. Keeping chunks
short also bounds the amount of CPU synthesis that can be in flight when a
barge-in cancels the turn.
"""

from __future__ import annotations

import re

_SENTENCE_END = ".!?…"
_CLAUSE_END = ",;:—–-"
_KEEP_PUNCTUATION = set(".,!?…;:—–-()«»\"'")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def sanitize_for_speech(text: str) -> str:
    """Drop markdown/emoji/URL noise but keep Russian letters and punctuation."""
    text = _URL.sub(" ", text)
    kept = []
    for char in text:
        if char.isalnum() or char.isspace() or char in _KEEP_PUNCTUATION:
            kept.append(char)
    return _SPACES.sub(" ", "".join(kept)).strip()


def _speakable(text: str) -> bool:
    return any(char.isalnum() for char in text)


class SpeakableChunker:
    """Accumulates streamed text and emits short speakable phrases."""

    def __init__(self, min_chars: int, soft_chars: int, max_chars: int):
        self._min = int(min_chars)
        self._soft = int(soft_chars)
        self._max = int(max_chars)
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        return self._drain(final=False)

    def flush(self) -> list[str]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[str]:
        chunks: list[str] = []
        while True:
            buffer = self._buffer
            if not buffer:
                break
            sentence_end = self._sentence_end(buffer)
            if sentence_end is not None and sentence_end + 1 >= self._min:
                self._buffer = buffer[sentence_end + 1:]
                self._push(chunks, buffer[: sentence_end + 1])
                continue
            if len(buffer) >= self._max:
                cut = self._cut(buffer, self._max)
                self._buffer = buffer[cut:]
                self._push(chunks, buffer[:cut])
                continue
            if len(buffer) >= self._soft:
                cut = self._clause_cut(buffer)
                if cut:
                    self._buffer = buffer[cut:]
                    self._push(chunks, buffer[:cut])
                    continue
            break
        if final and self._buffer:
            self._push(chunks, self._buffer)
            self._buffer = ""
        return chunks

    def _push(self, chunks: list[str], raw: str) -> None:
        text = sanitize_for_speech(raw)
        self._buffer = self._buffer.lstrip()
        if text and _speakable(text):
            chunks.append(text)

    @staticmethod
    def _sentence_end(buffer: str) -> int | None:
        for index, char in enumerate(buffer):
            if char in _SENTENCE_END:
                following = buffer[index + 1] if index + 1 < len(buffer) else " "
                if following.isspace():
                    return index
        return None

    @staticmethod
    def _cut(buffer: str, limit: int) -> int:
        space = buffer.rfind(" ", 0, limit)
        if space <= 0:
            return limit
        return space

    def _clause_cut(self, buffer: str) -> int:
        start = self._soft // 2
        best = 0
        for index in range(len(buffer) - 1):
            if index < start:
                continue
            if buffer[index] in _CLAUSE_END and buffer[index + 1].isspace():
                best = index + 1
        return best
