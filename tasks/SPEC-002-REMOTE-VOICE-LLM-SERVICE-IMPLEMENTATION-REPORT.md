# SPEC-002 implementation report — Remote Voice and LLM provider service

Status: **implemented; live provider, Docker and browser acceptance not verified.**

Branch: `DeepSeek-Flash`, base commit `03c4547`.
Specification: [`SPEC-002-REMOTE-VOICE-LLM-SERVICE.md`](SPEC-002-REMOTE-VOICE-LLM-SERVICE.md).

This report separates deterministic/offline evidence from live evidence. The environment used
for implementation had:

- Python 3.10 venv at `.venv`, downloaded models at `models/`;
- **no** Docker daemon (Compose files were validated with `docker compose config` only);
- **no** browser or microphone;
- **no** `OPENCODE_GO_API_KEY`, so the live OpenCode Go endpoint was never exercised;
- working outbound DuckDuckGo access (`ddgs 9.16.1`).

Mocked checks never substitute for live acceptance. No statement below claims a live provider
response, a browser run or a Docker image build.

## Dependencies and configuration

New direct pins in `requirements.in`, all present in the regenerated `requirements.lock`
(CPython 3.12 / Linux ARM64, generated with uv 0.11.14 via `sh tools/lock.sh`):

| Package | Version |
| --- | --- |
| `langchain-core` | 1.7.4 |
| `langchain-openai` | 1.7.3 |
| `openai` | 3.45.0 |
| `ddgs` | 9.16.1 |

Provider configuration (Remote mode only):

| Setting | Default used | Verified live? |
| --- | --- | --- |
| `OPENCODE_GO_API_KEY` | blank; must be supplied in ignored `.env` | No key available |
| `OPENCODE_GO_BASE_URL` | `https://opencode.ai/zen/go/v1` | Unauthenticated `GET /models` answered 200; no authenticated completion |
| `OPENCODE_GO_MODEL` | `deepseek-v4.1-flash` | Model appears in the unauthenticated model list; no completion |
| `x-opencode-session` | fresh UUID per LLM-service WebSocket session | Header requirement observed in provider docs/error; unique per conversation is unit-tested, live verification needs a key |

The provider documentation advertises `https://opencode.ai/inference/go/openai/v1` and asks
agent clients for a stable `x-opencode-session` header. The service sends that header (and an
explicit `User-Agent`) and allows overriding the base URL. It does **not** silently change the
requested endpoint or model.

## Commands and results

All commands run from the repository root.

### Static and regression checks

| Command | Result |
| --- | --- |
| `python -m compileall -q workspace llm_service tests` | exit 0 |
| `node --check workspace/static/app.js` (+ capture-worklet, playback, resampler) | exit 0 |
| `EL_VOICE_MODELS=$PWD/models python workspace/tests/test_contracts.py` | `RESULT: PASS` |
| `EL_VOICE_MODELS=$PWD/models python workspace/tests/test_pipeline_context.py` | `RESULT: PASS` (9 checks) |
| `EL_VOICE_MODELS=$PWD/models python workspace/tests/test_vad_parity.py` | `RESULT: PASS` in the development venv (official silero-vad 5.1.2); `RESULT: SKIP` in the image because the package is a dev-only test dependency |
| `EL_VOICE_MODELS=$PWD/models python tools/test_tts_integrity.py` | `OK` (5 tests) |
| `for test in workspace/tests/*.test.mjs; do node "$test"; done` | all three `RESULT: PASS` |
| `docker compose -f compose.yaml -f compose.local.yaml config --quiet` | exit 0 |
| `docker compose -f compose.yaml -f compose.remote.yaml config --quiet` | exit 0 |

### New deterministic checks

| Command | Result |
| --- | --- |
| `EL_VOICE_MODELS=$PWD/models python tests/test_llm_service.py` | `RESULT: PASS` (31 checks) |
| `EL_VOICE_MODELS=$PWD/models python tests/test_remote_backend.py` | `RESULT: PASS` (22 checks) |
| `python tests/test_search_live.py` | `RESULT: PASS` (4 mocked checks; live skipped) |
| `python tests/test_search_live.py --live` | `RESULT: PASS` (3 live results in 4.82 s; a later retry took 17.41 s after transient throttling) |

`tests/test_llm_service.py` starts the real `create_app` on a random loopback port with a
fake backend and drives HTTP/WebSocket. It proves: deltas/done, start statistics, context
ownership across turns, bounded history dropping oldest pairs, source/tool events,
cancel-mid-stream then next-turn recovery, error event recovery, reset, two-connection
isolation, remote-mode-without-key readiness/generate errors, and content-free `/status`.

`tests/test_remote_backend.py` injects a LangChain-style fake chat model and fake search. It
proves: typed `web_search` binding, tool-call query extraction, bounded sources, private URL
filtering, final-answer-only deltas, no tool args in spoken text, assistant tool call plus
`ToolMessage` pairing by `tool_call_id`, empty-result honesty, 3-iteration bound, search
failure reporting, missing-key error, and redaction of keys/bearer headers.

### End-to-end local check (both services, real models)

```sh
EL_VOICE_MODELS=$PWD/models .venv/bin/python workspace/tests/selftest.py --models $PWD/models
```

Result: `RESULT: PASS (24 checks)`.

The check starts the real LLM service (local Gemma backend) and the real voice app in one
process, then drives the browser WebSocket with synthesised Russian utterances. It verified:
one complete turn, first audio, barge-in flush, no stale audio after the flush, follow-up
context statistics, conversation limits, monotonic turn ids across sessions, abrupt
disconnect release, latency counters, local mode reporting, LLM-service readiness through the
voice status, and search disabled in local mode. Sandbox measurements:

| Measurement | Result |
| --- | --- |
| Speech end → first audio dispatch | p50 3696.5 ms, p95 11279.2 ms (n=4) |
| Barge-in onset → flush dispatch | p50 0.9 ms (n=1) |
| Local Gemma first token after STT | ~863 ms |
| Live DuckDuckGo search | 3 results in 4.82 s |

These are development-sandbox numbers, not target-Mac acceptance measurements. Remote
latency was not measurable.

## Criterion-by-criterion status

### P0 — Required behavior

| Criterion | Status | Evidence / gap |
| --- | --- | --- |
| **Two launch modes** | **Partial** | `compose.local.yaml` and `compose.remote.yaml` merge cleanly with the base; local LLM gets workspace/models/private network, remote LLM gets workspace/provider network and env key with no models. Exact start/stop/switch commands and blank placeholders are in README. Docker daemon/target Mac unavailable, so neither graph was run. |
| **Separate LLM boundary** | **Pass (deterministic)** | Voice `engines.py` no longer loads Gemma or has `stream_chat`; app/client route every turn through `elvoice/llm_client.py` to the internal WebSocket service. Contract tests cover turn ids, server-owned context, deltas, tool/source events, errors, done and cancellation. |
| **Live remote provider** | **Not verified** | No `OPENCODE_GO_API_KEY`. Missing-key path is proved to produce an actionable redacted error with no fallback. Endpoint/model/streaming/tool calling against a real account remain unverified. |
| **LangChain agent and context** | **Partial / Pass for deterministic parts** | LangChain drives the remote model/tool loop. Server-owned bounded history, trimming, session isolation, reset and pairing survive the loop in deterministic tests. A live contextual follow-up after a real search was not run. |
| **Real web search** | **Partial** | Real `ddgs` search from the sandbox works and returns bounded public results. Typed tool schema, result/snippet/time/iteration bounds, empty-result honesty, failure reporting and URL filtering are tested with fakes. Live model grounding and browser source display are not verified. |
| **Voice streaming** | **Partial** | Local end-to-end run shows incremental Gemma deltas → TTS/audio through the service boundary. Remote first-audio-before-completion and real browser playback were not observed live. Tool arguments/control fields are never delta text, and a one-chunk lookahead is dropped when a tool call is detected; a provider that streams prose before tool arguments in the same message could still surface that prose and must be checked live. |
| **Interruption and recovery** | **Partial** | Server cancel stops the backend iterator; local Gemma checks the token between tokens; tests cover cancel-mid-stream then next turn, error recovery, disconnect and reset. Remote provider-side cancellation is explicitly unverified. Browser stop/flush behavior is covered by node tests but not a real browser. |
| **Local regression** | **Partial / local pass** | The 21-check local end-to-end run passes with no remote key and no search (`llm.search.enabled=false`). The local LLM service runs on the internal `private` network. External-access-unavailable behavior was not tested in Docker; no remote/search fallback exists in code. |
| **Network and credentials** | **Partial** | Only the remote `llm` service joins the egress network and receives the key. Key is not committed; errors/status are redacted and tested; logs do not contain prompts/answers. Docker runtime network isolation and an actual provider key were not tested. |
| **Data and tool boundaries** | **Partial** | UI has a mode badge, remote disclosure and clickable sources; sources are never spoken; search accepts only a short query, never a URL/shell command; private/loopback/metadata result URLs are filtered; no tracing or persistent agent memory. Browser rendering was not observed. Developer test output prints only transcript lengths and status, never transcript text; no
audio or transcript content is persisted or written to runtime logs. |

### P1 — Verification and reproducibility

| Criterion | Status | Evidence / gap |
| --- | --- | --- |
| **Reproducible deployment** | **Partial** | New direct dependencies pinned, lock regenerated with hashes, rebuild requirement documented, image now copies `llm_service`, `.env` placeholders documented. Image build was not run (no daemon). |
| **Focused checks** | **Pass** | New service/agent/search tests and the updated self-test run; fake failure paths are clearly separated from the one live search check. |
| **Live browser acceptance** | **Not verified** | No browser/microphone. The required 10 representative remote turns, latency p50/p95, search/follow-ups and interruption measurements were not run. |
| **Criterion-by-criterion report** | **This file** | Pass for the report artifact; statuses above are honest and do not mark live items complete. |

## Known implementation gaps

- No successful live OpenCode Go completion, streaming or tool call was observed. Endpoint
  compatibility and account privacy/region settings remain unverified. The requested
  `/zen/go/v1` base answered an unauthenticated model list, but that does not prove chat
  completions work.
- Provider-side cancellation after a remote barge-in is not verified. The service aborts the
  local HTTP stream and drops the turn; it does not claim provider work stopped remotely.
- The remote model's search-triggering behavior, grounding quality and clickable-source
  rendering have not been exercised in a browser.
- Docker image build and both Compose graphs were not started; only static Compose
  validation was possible.
- The conversation history is per LLM-service WebSocket; a reconnect creates a fresh history.
  This is intentional for session isolation but should be checked against the browser
  reconnect flow during live acceptance.
