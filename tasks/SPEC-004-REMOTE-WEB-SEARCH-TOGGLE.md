# SPEC-004: Remote Voice Web Search Toggle

## 1. Problem and Current State

In Remote Voice, the LangChain agent always receives the `web_search` tool. The
browser interface has no control to disable web search, so every Remote Voice
turn may use it. The voice-session controls already include a user preference for
interrupting the assistant.

## 2. Required State

- Add an **Enable web search** checkbox beside the interruption preference in
  Remote Voice. It is unchecked by default, including at the start of each new
  session.
- Pass the setting from the browser through the voice API to the LLM service.
- When disabled, omit `web_search` from the LangChain agent's tools while keeping
  `begin_final_answer`. When enabled, include both tools.
- Apply a changed setting to subsequent user turns. An in-progress turn retains
  the setting and tool list with which it began.
- Keep web search unavailable in Local Voice. Local inference and its existing
  network isolation remain unchanged.
- Keep the preference session-scoped; do not persist it across page loads or
  sessions.

## 3. Acceptance Criteria

### P0 — Required behavior

- [ ] The checkbox is visible beside the interruption preference in Remote Voice
  and has an accessible label.
- [ ] The checkbox is unchecked on page load and at the start of a new session.
- [ ] The initial setting and subsequent changes are communicated to the voice
  service; acknowledgements keep the browser state synchronized.
- [ ] Each turn uses a stable snapshot of the session's web-search preference.
- [ ] With search disabled, the remote agent receives `begin_final_answer` and
  does not receive `web_search`.
- [ ] With search enabled, the remote agent receives `web_search` and
  `begin_final_answer`; successful search results and source events continue to
  work.
- [ ] Local Voice does not expose or enable web search, and its behavior remains
  unchanged.

### P1 — Regression coverage

- [ ] Browser tests verify the control, default-off state, session start payload,
  and change propagation.
- [ ] Voice WebSocket tests verify defaults, setting updates, acknowledgement,
  and applying the new value to the next turn.
- [ ] LLM service / LangChain tests verify the actual tool list for both states
  and prove that a disabled agent cannot invoke `web_search`.
- [ ] Existing interruption, Local Voice, Remote Voice, and source-display tests
  pass.
- [ ] Compose configuration remains valid; no new provider credentials or
  external network access are introduced.

## 4. Analysis

The user preference must cross three boundaries: browser state, the voice
WebSocket session, and the per-turn LLM request. The setting should be captured
when a turn starts so a concurrent checkbox change cannot alter an already
running agent. The remote provider can then assemble the agent's tool list from
that immutable per-turn value. Local Voice does not use the remote search tool.

## 5. Proposed Solution

Add a remote-only browser control with a default-off session value. Extend the
voice WebSocket start/control contract to accept and acknowledge the preference,
store it on the session, and snapshot it when processing each recognized turn.
Propagate that boolean with the LLM request. Conditionally add the search tool
when constructing the remote LangChain agent; retain the final-answer control
tool in either state. Add deterministic browser, WebSocket, and LLM-agent tests
for both values and for a setting change between turns.

## 6. Test Boundary

The proving tests must fail before implementation because the UI lacks the
preference and the remote agent always receives the search tool. After
implementation, tests must demonstrate the observable browser-to-agent behavior:
disabled means no search tool is registered for the turn; enabled means it is
registered and usable. Use deterministic model/search fakes; no live provider key
or real web search is required to prove toggle behavior.

## 7. ToDo

- [ ] Add browser control and state handling.
- [ ] Propagate the setting through the voice session and per-turn LLM request.
- [ ] Conditionally register the remote `web_search` tool.
- [ ] Add proving and regression tests for enabled, disabled, default, and
  between-turn changes.
- [ ] Validate Compose and run the relevant browser and Python suites.
