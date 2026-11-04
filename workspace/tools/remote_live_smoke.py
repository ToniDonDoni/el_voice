"""Run credential-safe live text, context, and search checks against the LLM service.

Run inside the Remote LLM container with:
    docker compose -f compose.yaml -f compose.remote.yaml exec llm \
      python /workspace/tools/remote_live_smoke.py

The API key is read only by the service process and is never printed by this tool.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid


BASE_URL = os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")


def turn(session_id: str, text: str) -> tuple[list[dict], float]:
    turn_id = uuid.uuid4().hex
    body = json.dumps({"text": text}, ensure_ascii=False).encode()
    request = urllib.request.Request(
        f"{BASE_URL}/v1/sessions/{session_id}/turns/{turn_id}/stream",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=90) as response:
        events = [json.loads(line) for line in response if line.strip()]
    return events, (time.monotonic() - started) * 1000


def main() -> None:
    session_id = uuid.uuid4().hex
    checks = (
        ("normal", "Ответь одним коротким предложением: зачем в Доте нужны варды?"),
        ("context", "А как это помогает против ганков?"),
        ("search", "Найди в интернете официальные заметки к последнему обновлению Dota 2 и назови номер патча."),
    )
    try:
        for label, prompt in checks:
            events, elapsed_ms = turn(session_id, prompt)
            kinds = [event.get("type", "unknown") for event in events]
            answer = "".join(event.get("text", "") for event in events if event.get("type") == "delta")
            sources = [source for event in events if event.get("type") == "sources"
                       for source in event.get("sources", [])]
            errors = [event.get("code", "unknown") for event in events if event.get("type") == "error"]
            print(f"{label}: events={','.join(kinds)} elapsed_ms={elapsed_ms:.0f} "
                  f"answer_chars={len(answer)} sources={len(sources)} errors={errors}")
            if not answer or errors or "done" not in kinds:
                raise RuntimeError(f"{label} turn did not complete successfully")
            if label == "search" and ("tool_call" not in kinds or not sources):
                raise RuntimeError("Live search did not return a tool call and sources")
    finally:
        request = urllib.request.Request(f"{BASE_URL}/v1/sessions/{session_id}", method="DELETE")
        try:
            urllib.request.urlopen(request, timeout=5).close()
        except (OSError, urllib.error.URLError):
            pass
    print("PASS: live provider, contextual follow-up, and web search completed")


if __name__ == "__main__":
    main()
