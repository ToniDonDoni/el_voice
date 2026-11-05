"""Turn-scoped event adapter for LangChain's streaming agent graph.

Only speak after the agent commits to its final-answer phase. Tool failures
must stay paired with their calls and must not erase previous search results.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from .search import public_https_url


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


def _sources(tool_message: Any, max_snippet_chars: int) -> tuple[list[dict[str, str]], str | None]:
    content = _text(tool_message)
    if getattr(tool_message, "status", None) == "error":
        return [], _error_code(content)
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return [], _error_code(content)
    if not isinstance(payload, dict):
        return [], "search_unavailable"
    if payload.get("error"):
        return [], _error_code(str(payload["error"]))
    results = payload.get("results")
    if not isinstance(results, list):
        return [], "search_unavailable"
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
    return sources, None


async def stream_agent(
    agent: Any, messages: list[Any], cancelled: Any, *,
    max_tool_calls: int, max_snippet_chars: int, observe: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Stream only explicitly committed final speech from one LangChain agent."""
    from contextlib import aclosing

    tool_count = 0
    answer_committed = False
    answer_emitted = False
    final_model_completed = False
    all_sources: list[dict[str, str]] = []
    known_urls: set[str] = set()
    call_names: dict[str, str] = {}

    request = agent.astream(
        {"messages": messages},
        config={"recursion_limit": 2 * max_tool_calls + 4},
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
                if not answer_committed:
                    # Planning messages are never user-facing speech.
                    continue
                if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
                    raise RuntimeError("provider_tool_loop_failed")
                text = _text(chunk)
                if text:
                    answer_emitted = True
                    yield {"type": "delta", "text": text}
                continue
            if mode != "updates" or not isinstance(data, dict):
                continue
            if "model" in data:
                outputs = _messages(data["model"])
                if not outputs:
                    continue
                response = outputs[-1]
                calls = list(getattr(response, "tool_calls", None) or [])
                if answer_committed:
                    if calls:
                        raise RuntimeError("provider_tool_loop_failed")
                    final_model_completed = True
                    if not answer_emitted and _text(response):
                        answer_emitted = True
                        yield {"type": "delta", "text": _text(response)}
                    continue
                if not calls:
                    # A planning step must explicitly enter final-answer mode.
                    # Never speak a possible tool-call preamble.
                    raise RuntimeError("provider_tool_loop_failed")
                if any(call.get("name") == "begin_final_answer" for call in calls) and len(calls) != 1:
                    raise RuntimeError("provider_tool_loop_failed")
                for call in calls:
                    call_id = str(call.get("id") or "")
                    name = str(call.get("name") or "")
                    call_names[call_id] = name
                    if name == "begin_final_answer":
                        observe("agent_answer_committed", tool_call_id=call_id)
                        continue
                    if name != "web_search":
                        raise RuntimeError("provider_tool_loop_failed")
                    tool_count += 1
                    if tool_count > max_tool_calls:
                        raise RuntimeError("provider_tool_loop_failed")
                    query = str((call.get("args") or {}).get("query", ""))[:240]
                    observe("agent_tool_call", tool_name=name, tool_call_id=call_id,
                            query=query)
                    yield {"type": "tool_call", "name": name,
                           "tool_call_id": call_id, "result_count": 0}
                continue
            if "tools" in data:
                for result in _messages(data["tools"]):
                    if getattr(result, "type", None) != "tool" and type(result).__name__ != "ToolMessage":
                        continue
                    call_id = str(getattr(result, "tool_call_id", ""))
                    name = str(getattr(result, "name", None) or call_names.get(call_id, ""))
                    if name == "begin_final_answer":
                        if getattr(result, "status", None) == "error":
                            raise RuntimeError("provider_tool_loop_failed")
                        answer_committed = True
                        continue
                    if name != "web_search":
                        raise RuntimeError("provider_tool_loop_failed")
                    sources, error_code = _sources(result, max_snippet_chars)
                    status = "error" if error_code else "success"
                    observe("agent_tool_result", tool_name=name, tool_call_id=call_id,
                            status=status, error_code=error_code,
                            result_count=len(sources))
                    for source in sources:
                        if source["url"] not in known_urls and len(all_sources) < 5:
                            known_urls.add(source["url"])
                            all_sources.append(source)
                    # Every result keeps its own status and ID. On failure,
                    # preserve links from prior successful tool calls.
                    yield {"type": "sources", "sources": list(all_sources),
                           "tool_call_id": call_id, "status": status,
                           "error_code": error_code}
    if not cancelled.is_set() and not final_model_completed:
        raise RuntimeError("provider_tool_loop_failed")
    if not cancelled.is_set() and (not answer_committed or not answer_emitted):
        raise RuntimeError("empty_generation")
