# SPEC-002 Implementation Report

## Summary

Implemented the two-mode Compose design and separate internal LLM service:
Local Voice routes to Gemma; Remote Voice uses LangChain with OpenCode Go and a
typed DuckDuckGo search tool. Browser audio capture, Whisper, and Silero `b_lez`
remain in the voice path. The requested Remote defaults are
`https://opencode.ai/zen/go/v1` and `deepseek-v4.1-flash`; an explicitly validated
same-provider documentation endpoint override is supported.

The acceptance criteria below are copied by name and status from SPEC-002; they
have not been weakened. This report separates real provider and synthetic audio
evidence from the still-unverified physical browser microphone run. The requested
configurable-interruption criterion remains additive.

## Verification performed

- Focused tests in a temporary Python 3.12 environment with pinned project dependencies:
  - `PYTHONPATH=workspace python -m unittest discover -s workspace/tests -p 'test_remote_voice.py'` — 20 passed.
  - `PYTHONPATH=workspace python -m unittest discover -s workspace/tests -p 'test_langchain_provider.py'` — 3 passed using the actual pinned LangChain graph (`langchain==1.4.8`).
  - `PYTHONPATH=workspace python -m unittest discover -s workspace/tests -p 'test_llm_client.py'` — 3 passed using HTTPX 0.28.1 and the actual client/service API boundary.
- Active branch: `codex/display-build-revision`. Remote Compose is currently running on ARM64: voice and LLM are healthy; Nginx publishes `127.0.0.1:8765->8080/tcp`.
- Host `curl` to `/healthz` returned ready. `/api/status` reported `ready=true`, `mode=remote`, VAD/STT/LLM/TTS ready, and revision `1fe259374fa0f31bd4738f270e31034b1ac385be-dirty`.
- The supplied public SSH-tunnel `/api/status` endpoint returned HTTP 200 and `ready=true`, `mode=remote` when TLS verification was disabled for diagnosis. Default certificate verification failed with `CERTIFICATE_VERIFY_FAILED`; public HTTPS trust is therefore not verified.
- Read the existing user-provided API environment file and copied its `OPENCODE_GO_API_KEY` into the repository's ignored `.env`; values match, `.env` mode is `0600`, and Git ignores it. Docker inspection confirmed the nonempty key is present in `el-voice-llm-1` and absent from voice and Nginx. The key was not printed, logged, or staged.
- A real streaming request through the running Remote LLM service returned a Russian Dota answer from the configured OpenCode Go provider; it emitted text deltas followed by `done` and no error.
- A live search request through the same service emitted `tool_call`, five source results, answer deltas, and `done` with no error. This verifies the live search/provider path, but not the full browser display or source-click behavior.
- A synthetic spoken Russian question was generated with local Silero `b_lez`, sent as audio through the Nginx `/ws/voice` endpoint, transcribed by local Whisper, answered by the live Remote provider, and synthesized locally. The saved smoke script returned 23 `ELVA` audio packets (831,600 PCM bytes at 24 kHz) and completed playback acknowledgements without a pipeline error. The first packet arrived 3,803 ms after the server marked speech ended. This was synthetic audio, not speech captured from the physical microphone or proof of audible playback on the Mac.
- Runtime status currently records five synthetic response samples: speech end to first audio p50 3,807 ms and p95 5,310 ms; no interruption-latency samples exist. These acknowledgements do not represent physical browser playback and do not meet SPEC-002's ten-turn measurement requirement.
- The browser page rendered Remote mode and the interruption checkbox checked by default. Starting a session ended back at `MIC OFF`; no connected browser microphone session or audible output was established. Physical browser audio acceptance remains unverified.
- Reproducible live checks are saved in `workspace/tools/remote_live_smoke.py` and `workspace/tools/remote_synthetic_audio_smoke.py`; their exact commands are listed below. They contain no key, persist no audio/transcript, and report only event types, lengths, counts, and timings.
- Full Python regression suite in the running LLM container: 50 passed, including the new cancellation regression. Node audio-capture and browser-playback suites: 2 passed. Both Local and Remote Compose configurations validate.
- Static/runtime Compose checks confirm only Remote LLM receives the provider key and only Remote LLM is attached to the dedicated `llm_egress` network. Nginx also joins the ordinary `ingress` bridge so the browser port is reachable; that bridge gives Nginx general outbound access, which conflicts with a strict reading of the unchanged Network and credentials criterion.
- `node --test workspace/tests/audio_capture.test.cjs workspace/tests/browser_playback.test.cjs` — both suites passed.
- Compose validation passed for both modes:
  - `docker compose -f compose.yaml -f compose.local.yaml config -q`
  - `docker compose -f compose.yaml -f compose.remote.yaml config -q`
- Live Remote provider, contextual follow-up, and web-search smoke:
  - `docker compose -f compose.yaml -f compose.remote.yaml exec llm python /workspace/tools/remote_live_smoke.py`
- Synthetic voice request through Nginx, local Whisper, live Remote LLM, and local Silero:
  - `docker compose -f compose.yaml -f compose.remote.yaml exec voice python /workspace/tools/remote_synthetic_audio_smoke.py`
- Static Compose inspection confirmed only Remote LLM receives `OPENCODE_GO_API_KEY` and is attached to the dedicated `llm_egress` network; voice/download do not receive the key; LLM has no host-published port; only Nginx publishes loopback port 8765. Nginx's separate ordinary `ingress` network allows general outbound access. Local LLM receives the Gemma mount; Remote LLM does not. Remote model preparation is key-independent.
- Real LangChain graph tests cover context-aware search planning, tool-call/result pairing, and search failure codes through service error mapping. A synthetic client test covers cancellation and truncated EOF. Search/provider fakes were used; they do not prove live external integrations.
- Lock generation was repeated byte-for-byte; pinned dependency compatibility check passed. `.env` is ignored and was not staged; its key was read in memory only for the local copy check.
- Historical GPT-6.1 Sol High cycle 3 review approved a narrowed code diff, but missed cancellation cleanup at the provider async-generator boundary. The issue is fixed and the final re-review confirmed the header, cancellation, reproducibility, and Remote-mode guard changes. Final verdict remains **REQUEST_CHANGES** because strict Nginx egress isolation and live browser/ten-turn acceptance are incomplete; the SPEC criteria were not changed.

## Acceptance criteria

| Original criterion | Status | Evidence and remaining work |
| --- | --- | --- |
| Two launch modes | PARTIAL | Remote ARM64 Compose is healthy and serves through the published loopback port. Local configuration validates and the prior local real-model self-test passed, but a clean Local-to-Remote-to-Local switch was not run in this verification pass. |
| Separate LLM boundary | PARTIAL | The live voice WebSocket synthetic-audio run traversed Nginx, local Whisper, internal LLM service, Remote provider, and local TTS. The Local service was separately exercised in the prior local self-test. The browser microphone run remains unverified. |
| Live remote provider | PASS (live text scope) | With the key mounted only in the LLM service, a real DeepSeek response streamed through OpenCode Go. The answer completed without provider errors. Live tool use is separately evidenced below. |
| LangChain agent and context | PARTIAL | Actual pinned `create_agent` graph tests cover contextual planning, paired tool messages, bounded history, and isolation with deterministic tests. The persisted live smoke completed a normal Dota turn, contextual follow-up, and search turn in one session. Live session isolation/clearing and tool-pair trimming against the provider remain unverified. |
| Real web search | PARTIAL | Live Remote request emitted a tool call, five search sources, and a grounded answer with no service error. Full spoken browser answer/source rendering and failure/throttle behavior against a live backend remain unverified. |
| Voice streaming | PARTIAL | Synthetic audio traversed the real Nginx WebSocket, Whisper, Remote provider and Silero path and returned 23 audio packets. No physical-microphone browser session or audible Mac playback was confirmed; first audio before full answer generation is not established by the current evidence. |
| Interruption and recovery | PARTIAL | Deterministic tests cover cancellation boundaries, stale turns, error/EOF, and retry behavior. Live interruption during provider/search/playback, provider-side cancellation, reconnect, and stop/restart remain unverified. |
| Configurable interruption | PASS (deterministic scope) | The rendered browser checkbox was verified checked by default and toggled off/on. WebSocket tests in Local and Remote flows verify that disabled interruption lets the active answer finish, enabled interruption accepts a new turn, and a later turn is accepted after a failure. The full connected browser/microphone flow remains unverified. |
| Local regression | PARTIAL | Local mode was previously started healthy and its real-model synthetic STT -> Local LLM -> TTS self-test passed. The current stack is Remote. This did not verify physical microphone capture or audible browser playback. Inherited SPEC-001 acceptance remains incomplete. |
| Network and credentials | PARTIAL / REQUIREMENT CONFLICT | Runtime inspection confirms the key exists only in Remote LLM, is sourced from ignored `.env`, and is absent from voice/Nginx. Remote LLM alone has the dedicated `llm_egress` network; Nginx's ordinary ingress bridge also permits general outbound traffic, so strict “only remote provider service” egress is not met. |
| Data and tool boundaries | PARTIAL | UI disclosure, in-memory context, local audio path, constrained search schema/URL filtering, no shell tool, and disabled external tracing are implemented and inspected. Rendered Remote UI and runtime storage were not exercised. Transcript/reply logs retain the inherited SPEC-001 privacy conflict. |
| Reproducible deployment | PARTIAL | Both Compose configurations validate and the Remote ARM64 stack is healthy with Nginx publishing loopback port 8765. The supplied public tunnel returns Remote status over HTTP 200 only when certificate verification is disabled; the public certificate chain needs repair/verification. |
| Focused checks | PASS (deterministic scope) | The current-container Python suite passed 50 tests after the new cancellation regression; both Node suites passed; both Compose configurations validated. Live provider/search and synthetic voice checks are separately recorded above. |
| Live browser acceptance | NOT VERIFIED | Remote UI is loaded, but its microphone session returned to MIC OFF. The ten-turn run, browser source rendering, real-device playback, search/no-search latency set, interruption timing, reconnect, headphones, and switch-back run are incomplete. Public status responds, but TLS certificate validation fails. |
| Criterion-by-criterion report | PASS | This report records every original SPEC-002 acceptance item and distinguishes deterministic evidence from pending deployment/live acceptance. |

## Remaining acceptance work

The ignored `.env` now contains the key copied from the user's API environment file;
the secret should remain local and must not be pasted into chat, logged, or committed.
The live Remote provider and search work, and a synthetic audio request returned
local TTS packets. Remaining acceptance requires a connected browser microphone and
audible playback, at least ten representative turns with latency measurements,
interruption/reconnect/stop-start checks, a Local/Remote switch-back, and valid
TLS verification on the public endpoint. Resolve the Nginx egress-capability conflict against the
unchanged Network and credentials criterion before marking the overall task accepted.
