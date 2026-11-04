"""Turn-scoped event adapter for LangChain's streaming agent graph.

A model may stream prose before later choosing a tool. Buffer each model step
until the graph confirms it is a final answer: speculative prose must never
reach the voice synthesizer.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from .search import SearchFailure, public_https_url


SEARCH_FAILURE_CODES = (
    "search_rate_limited", "search_timeout", "search_unavailable",
    "invalid_search_query",
)


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type", "text") == "text"
        and isinstance(part.get("text"), str)
    )


def _part(value: Any) -> tuple[str, Any]:
    # LangGraph v1 returns (mode, data); v2 returns typed StreamPart dicts.
    if isinstance(value, dict) and "type" in value and "data" in value:
        return value["type"], value["data"]
    if isinstance(value, tuple) and len(value) == 2:
        return value
    raise RuntimeError("provider_tool_loop_failed")


def _messages(update: Any) -> list[Any]:
    if not isinstance(update, dict):
        return []
    messages = update.get("messages", [])
    if isinstance(messages, list):
        return messages
    return [messages] if messages is not None else []


def _error_code(text: str) -> str:
    for code in SEARCH_FAILURE_CODES:
        if code in text:
            return code
    return "search_unavailable"


def _sources(tool_message: Any, max_snippet_chars: int) -> list[dict[str, str]]:
    content = _text(tool_message)
    if getattr(tool_message, "status", None) == "error":
        raise SearchFailure(_error_code(content))
    try:
        payload = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise SearchFailure(_error_code(content)) from exc
    if not isinstance(payload, dict):
        raise SearchFailure("search_unavailable")
    if payload.get("error"):
        raise SearchFailure(_error_code(str(payload["error"])))
    results = payload.get("results")
    if not isinstance(results, list):
        raise SearchFailure("search_unavailable")
    sources = []
    for item in results[:5]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", ""))[:2048]
        if not public_https_url(url):
            continue
        sources.append({
            "title": str(item.get("title", "Search result"))[:200],
            "url": url,
            "snippet": str(item.get("snippet", ""))[:max_snippet_chars],
        })
    return sources


async def stream_agent(
    agent: Any, messages: list[Any], cancelled: Any, *,
    max_tool_calls: int, max_snippet_chars: int, observe: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Yield only committed final-answer text and validated tool metadata."""
    from contextlib import aclosing

    pending_text: list[str] = []
    tool_count = 0
    answer_emitted = False
    request = agent.astream(
        {"messages": messages},
        config={"recursion_limit": 2 * max_tool_calls + 2},
        stream_mode=["messages", "updates"],
    )
    async with aclosing(request):
        async for part in request:
            if cancelled.is_set():
                return
            mode, data = _part(part)
            if mode == "messages":
                if not isinstance(data, tuple) or len(data) != 2:
                    continue
                chunk, metadata = data
                if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "model":
                    continue
                if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
                    continue
                text = _text(chunk)
                if text:
                    pending_text.append(text)
                continue
            if mode != "updates" or not isinstance(data, dict):
                continue
            if "model" in data:
                outputs = _messages(data["model"])
                if not outputs:
                    continue
                response = outputs[-1]
                calls = list(getattr(response, "tool_calls", None) or [])
                if calls:
                    # Discard model preambles preceding a tool call.
                    pending_text.clear()
                    tool_count += len(calls)
                    if tool_count > max_tool_calls:
                        raise RuntimeError("provider_tool_loop_failed")
                    for call in calls:
                        call_id = str(call.get("id") or "")
                        name = str(call.get("name") or "")
                        observe("agent_tool_call", tool_name=name, tool_call_id=call_id)
                        yield {"type": "tool_call", "name": name, "result_count": 0}
                    continue
                # An update without a tool request confirms a final model answer.
                chunks = pending_text or [_text(response)]
                pending_text = []
                for chunk in chunks:
                    if cancelled.is_set():
                        return
                    if chunk:
                        answer_emitted = True
                        yield {"type": "delta", "text": chunk}
                continue
            if "tools" in data:
                for result in _messages(data["tools"]):
                    if getattr(result, "type", None) != "tool" and type(result).__name__ != "ToolMessage":
                        continue
                    call_id = str(getattr(result, "tool_call_id", ""))
                    name = str(getattr(result, "name", None) or "web_search")
                    if name != "web_search":
                        raise RuntimeError("provider_tool_loop_failed")
                    sources = _sources(result, max_snippet_chars)
                    observe("agent_tool_result", tool_name=name, tool_call_id=call_id,
                            result_count=len(sources))
                    yield {"type": "sources", "sources": sources}
    if not cancelled.is_set() and not answer_emitted:
        raise RuntimeError("empty_generation")
