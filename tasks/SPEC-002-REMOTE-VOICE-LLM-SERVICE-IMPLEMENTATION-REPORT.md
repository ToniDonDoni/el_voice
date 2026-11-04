# SPEC-002 Implementation Report

## Summary

Implemented the two-mode Compose design and separate internal LLM service:
Local Voice routes to Gemma; Remote Voice uses LangChain with OpenCode Go and a
typed DuckDuckGo search tool. Browser audio capture, Whisper, and Silero `b_lez`
remain in the voice path. The requested Remote defaults are
`https://opencode.ai/zen/go/v1` and `deepseek-v4.1-flash`; an explicitly validated
same-provider documentation endpoint override is supported.

GPT-6.1 Sol High independently reviewed the SPEC-002 diff and approved cycle 3
with no blocking findings. This is a **code-review approval, not full acceptance**.
The new ARM64 image was not built or started, no user API key was supplied or
used, no live OpenCode request or internet search was made, and no browser/mic
conversation was run. Existing acceptance criteria remain intact; the requested
configurable-interruption criterion is additive.

## Verification performed

- Focused tests in a temporary Python 3.12 environment with pinned project dependencies:
  - `PYTHONPATH=workspace python -m unittest discover -s workspace/tests -p 'test_remote_voice.py'` — 20 passed.
  - `PYTHONPATH=workspace python -m unittest discover -s workspace/tests -p 'test_langchain_provider.py'` — 3 passed using the actual pinned LangChain graph (`langchain==1.4.8`).
  - `PYTHONPATH=workspace python -m unittest discover -s workspace/tests -p 'test_llm_client.py'` — 3 passed using HTTPX 0.28.1 and the actual client/service API boundary.
- Existing container regression suite: `docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_*.py'` — 44 passed, 6 expected skips because the already-running image predates the new LangChain/HTTPX tests. The container was not rebuilt, restarted, or otherwise changed.
- `node --test workspace/tests/audio_capture.test.cjs workspace/tests/browser_playback.test.cjs` — both suites passed.
- Compose validation passed for both modes without a key:
  - `docker compose --env-file /dev/null -f compose.yaml -f compose.local.yaml config --quiet`
  - `docker compose --env-file /dev/null -f compose.yaml -f compose.remote.yaml config --quiet`
- Static Compose inspection confirmed only Remote LLM receives `OPENCODE_GO_API_KEY` and external egress; voice/download do not receive the key; LLM has no host-published port; only Nginx publishes loopback port 8765. Local LLM receives the Gemma mount; Remote LLM does not. Remote model preparation is key-independent.
- Real LangChain graph tests cover context-aware search planning, tool-call/result pairing, and search failure codes through service error mapping. A synthetic client test covers cancellation and truncated EOF. Search/provider fakes were used; they do not prove live external integrations.
- Lock generation was repeated byte-for-byte; pinned dependency compatibility check passed. `git diff --check` passed. `.env` remains ignored and was not read or staged.
- Independent GPT-6.1 Sol High cycle 3 review: **PASS / APPROVED**, with no findings. Review artifact: [`SPEC-002-REMOTE-VOICE-LLM-SERVICE-REVIEW.md`](SPEC-002-REMOTE-VOICE-LLM-SERVICE-REVIEW.md).

## Acceptance criteria

| Original criterion | Status | Evidence and remaining work |
| --- | --- | --- |
| Two launch modes | PARTIAL | Both keyless Compose configurations validate; service graph, mounts, readiness definitions, network layout, and start/stop/switch instructions were inspected. Neither new graph was built, started, or switched. Linux ARM64 runtime remains unverified. |
| Separate LLM boundary | PARTIAL | Voice code routes generation through the internal HTTP client; both providers implement the documented session/turn streaming API. Client-to-service tests cover events, history, cancellation, and clearing. Network streaming timing and a complete browser-to-service turn remain unverified. |
| Live remote provider | NOT VERIFIED | Requested model/base URL, explicit endpoint override, credential location, redacted error handling, and redirect prohibition are covered by code/config checks. No account key or live inference request was used. |
| LangChain agent and context | PARTIAL | Actual pinned `create_agent` graph tests cover contextual search planning, complete paired tool messages, bounded history, and session isolation/clear behavior with deterministic fakes. Real provider follow-up quality and provider tool compatibility remain unverified. |
| Real web search | PARTIAL | Typed search tool, limits, public HTTPS filtering, error handling, process cancellation, and bounded admission are tested with fakes. No live DDG results or grounded spoken answer were observed; any backend policy/credential requirement remains to be confirmed. |
| Voice streaming | PARTIAL | Existing incremental audio/browser regressions pass; only final answer deltas are routed to TTS, and source/tool events have a UI path. No live Remote browser conversation or proof of first audio before full DeepSeek generation was performed. |
| Interruption and recovery | PARTIAL | Deterministic tests cover cancellation across the client/service boundary, closing a pending read, search worker termination, stale turns, and error/EOF handling. Live generation/search/playback interruption, provider-side cancellation, reconnect, and stop/restart remain unverified. |
| Configurable interruption | PASS (deterministic scope) | The rendered browser checkbox was verified checked by default and toggled off/on. WebSocket tests in Local and Remote flows verify that disabled interruption lets the active answer finish, enabled interruption accepts a new turn, and a later turn is accepted after a failure. The full connected browser/microphone flow remains unverified. |
| Local regression | PARTIAL | Existing regressions pass; local Gemma loading verifies the model receipt, tokenizer assets are packaged, and Local mode has no remote fallback/egress. The new Local service was not started for a real offline conversation. Inherited SPEC-001 acceptance remains incomplete. |
| Network and credentials | PARTIAL | Static Compose inspection confirms the key and intentional egress are limited to Remote LLM, with no LLM host port and redirects disabled. Runtime network/log inspection against the new containers was not performed. |
| Data and tool boundaries | PARTIAL | UI disclosure, in-memory context, local audio path, constrained search schema/URL filtering, no shell tool, and disabled external tracing are implemented and inspected. Rendered Remote UI and runtime storage were not exercised. Transcript/reply logs retain the inherited SPEC-001 privacy conflict. |
| Reproducible deployment | PARTIAL | Dependencies and image base are pinned, lock is reproducible, preparation/start/switch commands are documented, and source mounts remain. New ARM64 Docker image build was not run. |
| Focused checks | PASS (deterministic scope) | 20 service tests, 3 actual LangChain graph tests, 3 HTTPX client tests, 44 existing-container tests (6 dependency skips), 2 Node suites, Compose configs, lock reproducibility, and diff check passed. These checks do not stand in for live integration. |
| Live browser acceptance | NOT VERIFIED | No ten-turn remote conversation, latency p50/p95, live search/no-search comparison, interruption timing, headphones, reconnect, or switch-back run was performed. |
| Criterion-by-criterion report | PASS | This report records every original SPEC-002 acceptance item and distinguishes deterministic evidence from pending deployment/live acceptance. |

## Remaining acceptance work

Interruption-control regression checks passed in the existing container
(21 Python pipeline test methods) and both Node browser/audio suites. These deterministic
checks do not replace the connected browser/microphone acceptance run.

The user must build and start one mode at a time using the commands in the
repository README. Remote acceptance additionally needs the user-provided key in
the ignored local `.env`; do not paste it into chat or a shell command. Then verify
the configured OpenCode endpoint/model and real streaming/tool calls, live search,
ten representative browser turns and contextual follow-ups, first-audio timing,
interruptions during model/search/playback, recovery, and switch back to Local.
Record search/no-search latency and interruption latency separately. Inspect
logs/storage and confirm the UI disclosure. Until those observations exist, the
corresponding criteria above remain partial or unverified.
