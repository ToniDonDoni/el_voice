# SPEC-003: LLM-Driven Tool Calling

## 1. Problem and Current State
- Regex decides whether web search is needed.
- Search and final response use separate LLM workflows.
- Search results are injected into `SystemMessage`.

## 2. Required State
- LLM autonomously decides when to call `web_search`.
- One LangChain agent manages the entire conversation turn.
- LLM can call tools repeatedly or answer directly.
- Tool results are passed through `ToolMessage`, never `SystemMessage`.
- Every model request, tool call/result, reasoning telemetry, and final answer within a turn shares its existing `turn_id` (also used as `trace_id`) for correlation.

## 3. Acceptance Criteria
- [ ] No regex-based search routing.
- [ ] One `create_agent` workflow per turn.
- [ ] Zero, one, or multiple tool calls supported.
- [ ] Final answers stream to TTS.
- [ ] Search results remain untrusted tool data.
- [ ] All model/tool/stream/error events for a turn carry the same `turn_id`/`trace_id`; individual tool calls retain distinct `tool_call_id` values.
- [ ] Cancellation, history, source events and iteration limits work.

## 4. Analysis
Existing LangChain `create_agent` supports autonomous tool calling. No custom agent loop is necessary. Codex already correlates requests by `turn_id`/`trace_id`; preserve this across the full agent loop without logging reasoning content.

## 5. Proposed Solution
Reuse Codex infrastructure and `create_agent`. Remove `needs_search()` and the separate final LLM call. Let the agent manage tool execution and return the final response. Propagate the turn ID across each agent step and keep per-tool call IDs.

## 6. ToDo
- [ ] Refactor remote provider.
- [ ] Preserve streaming and cancellation.
- [ ] Preserve turn-wide correlation for model calls, reasoning metrics, tools and errors.
- [ ] Add tests for direct answers, repeated searches, trace IDs and failures.
- [ ] Run regression and integration tests.
