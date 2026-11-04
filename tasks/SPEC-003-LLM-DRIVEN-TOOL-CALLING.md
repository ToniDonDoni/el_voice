# SPEC-003: LLM-Driven Tool Calling

## 1. Problem and Current State
- `needs_search()` uses regex to decide whether searching is allowed.
- A search-only `create_agent().ainvoke()` runs before a separate final `model.astream()` call.
- Search results are inserted into `SystemMessage`.

## 2. Required State
- In Remote Voice, one LangChain agent execution owns the entire turn: decide, search if needed, and answer.
- The LLM may answer without tools or invoke `web_search` one or multiple times, including follow-up searches.
- Search data stays in paired `ToolMessage` results, never in `SystemMessage`.
- Stream only final user-facing answer text to TTS; never speak reasoning, tool arguments, or intermediate agent messages.
- All model requests, reasoning metrics, tool calls/results/errors, source events, and answer chunks of one turn share the existing `turn_id` as `trace_id`. Retain `session_id`, per-request index, and individual `tool_call_id` / `tool_invocation_id` values.
- Preserve existing context, source UI, cancellation, bounded execution, provider session headers, and Local Voice behavior.
- Recover from individual search failures without discarding earlier successful results. Send a bounded error code as paired tool data so the LLM may retry or answer from available evidence; if no search succeeded, it must disclose the inability to verify current facts.

## 3. Acceptance Criteria
- [ ] No regex gate or hardcoded search router; the model decides when to search.
- [ ] One agent execution per turn handles zero, one, or multiple tool calls; no separate final-answer workflow.
- [ ] LangChain pairs tool calls with `ToolMessage` results; untrusted search text never becomes a system instruction.
- [ ] Final-answer tokens reach TTS incrementally; reasoning, tool data, and intermediate messages never do.
- [ ] Every model/tool/stream/error event is correlated by the same `turn_id`/`trace_id`; distinct requests and tools keep distinct IDs, and turns never mix.
- [ ] Cancellation, search timeouts, failure codes, history, source links, and iteration limits remain functional.
- [ ] Tests cover no-search, single-search, repeated-search, streaming, tracing, errors, and interruption.
- [ ] Mixed successful/failed parallel searches preserve the successful sources and can finish with `delta` + `done` without `replace`; all-failed searches retain per-call error codes and can produce an honest unverified answer.

## 4. Analysis
LangChain `create_agent` already supports repeated tool calls and streaming. Multiple LLM requests inside one agent execution are expected; the separate search-agent and final-answer workflows are not needed. Keep reasoning telemetry, not reasoning content, in logs.

## 5. Proposed Solution
Reuse Codex's model, search tool, transport, and observability. Replace `needs_search()`, `_search_context()`, and the standalone final `model.astream()` with one streaming `create_agent` execution. Extract sources from tool results, forward only final answer deltas, and propagate turn correlation throughout the agent run.

## 6. ToDo
- [ ] Refactor the remote provider into one agent-driven stream.
- [ ] Preserve safe answer streaming, source events, cancellation, and limits.
- [ ] Propagate `trace_id` to every provider request, tool event, and reasoning metric.
- [ ] Add deterministic contract tests and run regression/integration checks.
- [ ] Test a parallel search batch with both successful and failed results, plus an all-failed batch.
