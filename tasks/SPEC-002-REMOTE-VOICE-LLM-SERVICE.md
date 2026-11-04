# Specification 002: EL Voice — Remote Voice and LLM Provider Service

## 1. Problem Statement

The current browser assistant runs VAD, Whisper, Gemma, and Silero in one Python
application. Conversation context is assembled manually; there is no agent
framework, remote LLM provider, or internet-search tool. Users need two explicit
Compose launch modes: Local Voice with offline inference, and Remote Voice with
DeepSeek Flash through OpenCode Go and internet search.

This task specifies the next implementation milestone. It does not claim that the
remaining [SPEC-001 review](SPEC-001-LOCAL-VOICE-CONVERSATION-Review.md) findings or
browser acceptance checks have been completed.

## 2. Design Statement

- Provide named Local Voice and Remote Voice launch configurations in Docker
  Compose, using profiles or override files. A documented command for each mode
  starts its complete service graph. Switching modes must stop the previous graph
  without leaving conflicting host ports or obsolete provider processes.
- Refactor LLM access behind a separate internal Compose service with one documented
  streaming and cancellation contract. The voice application calls this service
  in both modes. Local mode runs the existing Gemma model in that service; remote
  mode runs a LangChain agent that calls OpenCode Go. No GPU or additional local
  remote-model weights are needed for Remote Voice.
- Keep browser capture/playback, streaming VAD, Whisper STT, and Silero `b_lez` TTS
  local. Remote mode sends text, bounded conversation context, and necessary tool
  results to external services. Raw microphone audio stays local.
- Give the remote LLM service outbound HTTPS access for inference and search. Keep
  the voice application on an internal network and publish only the existing
  loopback-bound Nginx browser endpoint. The LLM service has no host-published port.
- Use LangChain for remote model integration, agent/tool execution, bounded session
  context, and streaming. Register a real `web_search` tool. Use a maintained,
  pinned DuckDuckGo search integration as the initial backend; make the backend
  configurable and document any replacement or additional credential requirement.
- Search returns titles, URLs, and bounded snippets. The agent incorporates results
  and the UI shows clickable sources. Spoken replies remain concise and do not
  read raw URLs. Search is invoked when requested or needed for current facts;
  ordinary conversational turns need not search.
- Keep provider keys exclusively in the remote service at runtime. Preserve local
  mode without credentials, remote calls, or search. Update project guidance during
  implementation to describe this explicitly authorized remote-mode extension to
  the existing local-only policy.

### Provider configuration

| Setting | Required value |
| --- | --- |
| Secret environment variable | `OPENCODE_GO_API_KEY` (user supplied; never committed) |
| Base URL environment variable | `OPENCODE_GO_BASE_URL` |
| Requested default base URL | `https://opencode.ai/zen/go/v1` |
| Model environment variable | `OPENCODE_GO_MODEL` |
| Requested default model | `deepseek-v4.1-flash` |

Use an OpenAI-compatible chat integration with an explicit base URL and API key.
The user-supplied URL remains the requested default. At specification time,
[OpenCode Go documentation](https://opencode.ai/docs/go/) lists this model but
advertises `https://opencode.ai/inference/go/openai/v1/chat/completions` as its
chat endpoint. Verify the requested URL with the supplied account before live
acceptance; document any required configuration correction. Do not silently select
another endpoint, model, or provider. Do not assume that the inference key also
provides a search API or that hosted search is built into this model.

## 3. Acceptance Criteria

### P0 — Required behavior

- [ ] **Two launch modes:** README contains exact preparation, start, stop, and
  switch commands for both modes. Each starts Nginx, voice, and its selected LLM
  service with working health/readiness checks. Local mode starts without remote
  keys; Remote Voice does not load or require Gemma weights. Both modes work on
  Linux ARM64 under Docker Desktop on Apple Silicon.
- [ ] **Separate LLM boundary:** Both modes route generation through the internal
  LLM service. Voice code does not directly instantiate Gemma or contact OpenCode.
  The shared contract covers session/turn identity, context ownership, text deltas,
  tool/source events, errors, completion, and cancellation.
- [ ] **Live remote provider:** With the user's runtime key, the selected service
  actually calls the configured OpenCode Go endpoint with `deepseek-v4.1-flash`.
  Verify streaming and tool calling against that account. Missing/invalid keys,
  unsupported model/tool behavior, and endpoint incompatibility produce actionable
  redacted errors; no silent fallback or fabricated successful answer occurs.
- [ ] **LangChain agent and context:** LangChain drives the remote model/tool loop.
  Multiple turns retain useful context, including a contextual follow-up after
  search. Sessions remain isolated and context stays within configured token and
  tool-result budgets. Tool-call IDs and results remain correctly paired after
  trimming; clearing/stopping a session removes its in-memory context.
- [ ] **Real web search:** A spoken request for current information produces an
  observable tool call, live search results, a grounded spoken answer, and matching
  clickable sources. Tool input has a schema; result count, content size, execution
  time, and agent iterations are bounded. Empty results, unavailable search, and
  throttling are reported honestly. Any separate search key is documented and is
  never confused with `OPENCODE_GO_API_KEY`.
- [ ] **Voice streaming:** In Remote Voice, a real browser conversation runs through
  Nginx, local Whisper, remote generation, and local `b_lez` synthesis. First audio
  can play before the complete answer is generated. Reasoning fragments, partial
  tool arguments, and tool-control messages are never spoken as answer text.
- [ ] **Interruption and recovery:** Speech interrupts remote generation, tool
  execution, and playback. Close/cancel in-flight local HTTP streams and pending
  work, flush buffers, reject stale turn results, and accept the next utterance.
  The user-controlled interruption toggle from SPEC-001 applies in both modes: when it is
  disabled, speech during generation must not cancel the answer and the answer must finish.
  Distinguish local cancellation from any unverified provider-side cancellation.
  Disconnect, stop/restart, timeout, rate-limit, and provider failure recover without
  stuck sessions, duplicate answers, or unbounded retries and queues.
- [ ] **Local regression:** Local Voice still works with external access unavailable,
  preserves conversation context, VAD, and `b_lez`, and has no remote/search fallback.
  Continue recording any inherited SPEC-001 failures against that specification.
- [ ] **Network and credentials:** Only the remote provider service receives the key
  and intentional outbound connectivity. The browser and voice container never
  receive it. Runtime secrets are excluded from Git and image build contexts;
  logs, errors, status, and screenshots redact credentials and authorization headers.
  Requests follow the configured HTTPS provider without leaking keys on redirects.
- [ ] **Data and tool boundaries:** UI clearly identifies Local or Remote mode and
  explains remote text/search disclosure. Keep audio local and avoid persistent
  agent memory or external tracing. Preserve explicitly requested local diagnostic
  logging while reporting the existing SPEC-001 privacy conflict. Search results
  are untrusted data; tools cannot execute shell commands or fetch private,
  loopback, or metadata endpoints through model-generated URLs.

### P1 — Verification and reproducibility

- [ ] **Reproducible deployment:** Pin new LangChain/provider/search dependencies
  and image versions, update lock files and ignore rules as needed, document rebuilds
  and required environment variables with blank secret placeholders. Routine source
  edits retain the mounted-source workflow.
- [ ] **Focused checks:** Verify the service contract, context trimming, tool-call
  assembly, stream errors, cancellation, session isolation, and both Compose
  configurations with reproducible checks. Use deterministic provider/search fakes
  for failure paths; identify them separately from live integrations.
- [ ] **Live browser acceptance:** On the target Mac, test at least 10 representative
  remote turns including search and follow-ups; measure response latency separately
  for search/no-search turns and interruption latency. Check multiple turns,
  generation/search/playback interruptions, headphones without self-interruption,
  reconnect, and switching back to Local Voice. Record p50/p95 and limitations.
- [ ] **Criterion-by-criterion report:** Create
  `SPEC-002-REMOTE-VOICE-LLM-SERVICE-IMPLEMENTATION-REPORT.md` beside this task with
  commands, dependency versions, configured provider/model, evidence, measurements,
  and pass/partial/fail/not-verified status for every criterion. Without a live key
  or working search, mark affected criteria unverified; mock checks do not complete
  live acceptance. Do not weaken this specification to match the implementation.

## 4. Analysis

The existing container/proxy and local speech models can be retained. The required
refactor separates model access and agent orchestration from audio/session handling,
while retaining one browser interface and predictable turn cancellation. A shared
provider contract prevents separate local and remote voice applications from
accumulating divergent playback and session logic.

LangChain supplies model/tool abstractions, but endpoint compatibility, streaming
tool-call assembly, search connectivity, and bounded context still require explicit
implementation and verification. See [LangChain models](https://docs.langchain.com/oss/python/langchain/models),
[tools](https://docs.langchain.com/oss/python/langchain/tools), and
[ChatOpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/openai).
The [DuckDuckGo tool reference](https://reference.langchain.com/python/langchain-community/tools/ddg_search/tool/DuckDuckGoSearchResults)
is an initial integration candidate; choose and lock mutually compatible packages
and verify live behavior before declaring search ready.

Remote mode requires a deliberate change to network topology and data disclosure.
Only its LLM service needs inference/search access. The original local task remains
an offline deliverable; this task adds an opt-in online mode. Game control, arbitrary
tools, cloud STT/TTS, persistent memory, and unrelated UI redesign are outside scope.

## 5. Proposed Solution

Define one internal LLM service contract, extract existing Gemma access behind it,
and add a remote implementation using LangChain, the configured OpenCode Go model,
and a typed search tool. Keep session context under one documented owner and stream
only final-answer text into the existing bounded TTS/playback path. Propagate turn
identity and cancellation across voice, provider, agent, and browser boundaries.

Add complete Compose launch configurations for each mode, explicit runtime secret
injection, remote-service egress, and clear health/readiness states. Document the
commands and mode differences. Verify local regressions and deterministic failure
cases first, then the real endpoint, search, and Mac browser scenarios. Record
outcomes without modifying the original task or its acceptance criteria.

## 6. ToDo

- [ ] Confirm provider URL, model access, streaming/tool compatibility, and a working
  search backend; record required packages and runtime configuration.
- [ ] Define the internal streaming/cancellation contract and context ownership.
- [ ] Extract local Gemma access into the dedicated LLM service.
- [ ] Implement the remote LangChain agent, bounded context, and real search tool.
- [ ] Add Local Voice / Remote Voice Compose configurations and runtime secrets.
- [ ] Integrate voice streaming, cancellation, source display, and mode indication.
- [ ] Update README, scoped project guidance, locks, and reproducible checks.
- [ ] Complete local and remote acceptance and publish the criterion-by-criterion
  implementation report with remaining limitations.
