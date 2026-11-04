# SPEC-001: LLM-Driven Tool Calling

## 1. Problem and Current State
- Regex decides whether web search is needed.
- Search and final response use separate LLM workflows.
- Search results are injected into `SystemMessage`.

## 2. Required State
- LLM autonomously decides when to call `web_search`.
- One LangChain agent manages the entire conversation turn.
- LLM can call tools repeatedly or answer directly.
- Tool results are passed through `ToolMessage`, never `SystemMessage`.

## 3. Acceptance Criteria
- [ ] No regex-based search routing.
- [ ] One `create_agent` workflow per turn.
- [ ] Zero, one, or multiple tool calls supported.
- [ ] Final answers stream to TTS.
- [ ] Search results remain untrusted tool data.
- [ ] Cancellation, history, source events and iteration limits work.

## 4. Analysis
Existing LangChain `create_agent` supports autonomous tool calling. No custom agent loop is necessary.

## 5. Proposed Solution
Reuse Codex infrastructure and `create_agent`. Remove `needs_search()` and the separate final LLM call. Let the agent manage tool execution and return the final response.

## 6. ToDo
- [ ] Refactor remote provider.
- [ ] Preserve streaming and cancellation.
- [ ] Add tests for direct answers, repeated searches and failures.
- [ ] Run regression and integration tests.
