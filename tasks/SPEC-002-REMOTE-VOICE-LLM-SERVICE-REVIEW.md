# SPEC-002 Independent Review

Reviewer: GPT-6.1 Sol High

Scope: review the implementation on branch `codex/spec-002-remote-voice` against
[`SPEC-002-REMOTE-VOICE-LLM-SERVICE.md`](SPEC-002-REMOTE-VOICE-LLM-SERVICE.md),
including the separate LLM service, Local and Remote Compose modes, provider and
search boundaries, cancellation, tests, and deployment documentation. Do not edit
the specification or implementation. Record each review cycle and its verdict here.

## Cycle 1

Pending independent review.

### Independent GREEN_REVIEW outcome

**Verdict: FAIL / REQUEST_CHANGES.** The code is not ready for acceptance. The
ordinary one-search LangChain path fails at its default recursion limit, and
additional reproducible contract, context, cancellation, and deployment defects
remain. Full SPEC-002 acceptance is also unverified because the user will build
and activate the new deployment and supply the runtime key later.

Reviewed the current working diff on `codex/spec-002-remote-voice`, based on
`0411ead`, against the original request, the canonical SPEC-002, project
`AGENTS.md`, the spec-driven-tdd skill, and the acceptance-boundary guide. The
original request requires two complete Compose modes, one internal LLM boundary,
Gemma locally, LangChain/OpenCode Go remotely, real configurable search, local
speech processing, and provider credentials/egress only in the remote service.
The implementation report was absent at review time; the implementer plans to
write it after this review. No production code, tests, or specification were
changed by the Reviewer. No commits, Docker builds/restarts, microphone capture,
live search, or live OpenCode requests were performed.

### Verification performed

- `docker compose --env-file /dev/null -f compose.yaml -f compose.local.yaml config --quiet`: PASS.
- `OPENCODE_GO_API_KEY=review-placeholder docker compose --env-file /dev/null -f compose.yaml -f compose.remote.yaml config --quiet`: PASS. A synthetic placeholder was used; no real credential was printed or inspected.
- The same Remote configuration without a key: expected interpolation error. This also prevents the README's initial Remote model-preparation command from running before the key is configured.
- `PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python -m unittest discover -s workspace/tests -p 'test_remote_voice.py'`: 14 tests PASS, Python 3.12; installed LangChain 1.4.8, FastAPI 0.162.0.
- `docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_*.py'`: 32 tests PASS in 4.503 seconds, using the existing container and mounted source. This does not test the proposed new image or deployment graph. Docker socket access required sandbox escalation; the approved command only executed deterministic tests in the existing container.
- `node --test workspace/tests/audio_capture.test.cjs workspace/tests/browser_playback.test.cjs`: both suites PASS. These use the existing browser harness, not a rendered browser with the new Remote UI flow.
- `git diff --check`: PASS.
- An attempted full host run of the 32 tests failed with five environment errors: missing `uvicorn`, `onnxruntime`, and the container-only `/opt/el-voice/models.json` path. Those are not reported as production regressions; the corresponding container run passed.
- Additional deterministic reproductions below exercise the real pinned LangChain graph, application response generator, tokenizer loader, and tool-message types. No external request was made by these reproductions.

### Acceptance matrix

Status meanings: PASS means the cited behavior was actually demonstrated at the
stated boundary; PARTIAL means some required behavior has evidence; FAIL means a
concrete defect prevents the criterion; NOT VERIFIED means live evidence is absent.

| Criterion | Action, evidence, and boundary | Review status |
| --- | --- | --- |
| Two launch modes | Both merged Compose configs validate. Both select voice/LLM/proxy; only proxy publishes loopback port; Remote LLM has no models mount; Local config needs no key. README has start/stop/switch commands. Fresh preparation before key setup fails (F8). Neither graph was built, started, or switched by this review. | PARTIAL / F8 |
| Separate LLM boundary | `VoiceRuntime.llm_stream` calls `LLMServiceClient`; production voice no longer instantiates Gemma. `StreamingContractTests.test_streaming_context_isolated_sources_and_history_lifecycle` POSTs the service API and asserts tool/source/delta/done events, follow-up history, separate-session empty history, and DELETE removal. Shared streaming/cancellation schema is implemented but lacks a complete documented contract; F3 and F4 affect correctness. There is no voice-to-client-to-service proving test. | PARTIAL / F3, F4 |
| Live remote provider | `ChatOpenAI` receives explicit requested URL/model/key, streaming enabled, retries disabled, custom async HTTPX client with redirects disabled. No account call, endpoint, actual streaming, or actual tool compatibility was verified. The documented endpoint correction is rejected (F2). | NOT VERIFIED / F2 |
| LangChain agent and context | Real `create_agent` reproduction fails a single normal search at default settings (F1). Context unit test checks retained recent pairs; service HTTP test checks isolation/history/delete against a generic FakeProvider. Tool-pair test accepts an orphaned call (F4); search planner receives no history (F6). No actual remote contextual follow-up test exists. | FAIL / F1, F4, F6 |
| Real web search | Typed LangChain `web_search` invokes DDGS; count/snippet/payload limits and timeout mapping have isolated fake tests. No live search or grounded spoken result was observed. Default real agent loop fails (F1), and cancellation leaves the synchronous worker running (F5). | FAIL; live portion NOT VERIFIED |
| Voice streaming | Existing `TurnRegressionTests` and Node harness demonstrate incremental speech/playback using legacy fake runtimes. Production audio still uses local Whisper and `b_lez`; remote events are forwarded to the voice pipeline. No deterministic test crosses the actual LLM client/service/provider path into speech output. No new rendered Remote browser run or real conversation was performed. | PARTIAL / NOT VERIFIED |
| Interruption and recovery | HTTP cancel test starts BlockingProvider, POSTs cancel, and observes coroutine cancellation and discarded history. Existing regressions exercise voice stale-result/queue/playback behavior with fakes. These do not cancel an actual synchronous search worker (F5), and backpressure can strand the service response generator (F3). Provider-side cancellation, real search/generation/playback barge-in, reconnect and restart remain unverified. | FAIL / F3, F5 |
| Local regression | Existing 32-test suite passes; Local service uses Gemma, no remote fallback, internal-only network. Local model loading and full offline conversations were not run through the extracted service. New token counting attempts an external asset fetch with a cold cache (F7). Inherited SPEC-001 browser/performance/privacy acceptance remains open. | PARTIAL / F7 |
| Network and credentials | Compose inspection shows key and `llm_egress` only on remote LLM; voice/private and proxy/private+internal ingress; no LLM host port. `.env` is Git-ignored and Docker build context is an allowlist excluding `.env`, source, models, and data. Errors whitelist codes, do not interpolate keys. Async provider HTTP redirects disabled. No actual new runtime network or log inspection was performed. | PASS for static topology/secret wiring; runtime NOT VERIFIED |
| Data/tool boundaries | UI source identifies mode and text/search disclosure; source links use textContent, HTTPS URLs, noopener/noreferrer. Search has a typed query only, no shell or model-URL fetch tool; result URL filter rejects private literal addresses/local suffixes. In-memory context only and LangSmith tracing disabled. No new rendered UI/audio acceptance run. Requested text logging remains and retains the inherited SPEC-001 privacy conflict. | PARTIAL |
| Reproducible deployment | Direct dependencies pinned and hash lock updated; Python base digest pinned; source mounts retained; blank key placeholder present. New image build/ARM64 dependency installation is owned by user and was not verified. README preparation/configuration problems remain (F2, F8). | PARTIAL |
| Focused checks | 14 service/helper tests and existing regressions pass. Generic FakeProvider bypasses real LangChain, actual HTTP client integration, assembled streamed tool calls, and native search cancellation. Targeted graph reproduction finds a failure missed by all 32 tests. No new rendered mode/source test. | FAIL as complete proving evidence |
| Live browser acceptance | No ten-turn remote run, search/no-search p50/p95, interruption p50/p95, headphones, reconnect, mode switching, or physical output checks performed. User has not supplied the key or activated the new stack. | NOT VERIFIED |
| Criterion-by-criterion report | Required `tasks/SPEC-002-REMOTE-VOICE-LLM-SERVICE-IMPLEMENTATION-REPORT.md` absent at review time. The final report must include every criterion, exact commands/versions, and explicit unverified live scenarios. | NOT YET PROVIDED |

### Findings to fix

1. **P1 — Default LangChain graph cannot complete a normal search.**
   `workspace/llm_service/remote_provider.py:71`, `:106-109` use a default of 3 as
   LangGraph's recursion limit. With the real pinned `create_agent`, an ordinary
   model tool call, one successful tool result, and final model answer exhaust
   this limit. Deterministic reproduction below yields `GraphRecursionError` for
   3 and succeeds for 4. Consequently every normal requested-search turn can fail
   before sources or answer are emitted. The error also occurs outside the
   provider exception-mapping `try` at `:149`, so it becomes generic
   `generation_failed`. Use a graph-step budget that permits the complete
   required flow, bound actual tool/model work deliberately, and add a test of
   the real graph with deterministic model/search fakes. Also test authentication,
   rate-limit, timeout, and malformed tool behavior during the search phase so
   they retain actionable redacted codes.

2. **P1 — Advertised endpoint correction cannot be configured.**
   `workspace/llm_service/remote_provider.py:39-47` accepts only the exact requested
   `/zen/go/v1` path and default model. `README.md:38` tells users to override the
   endpoint if their account requires a correction; the canonical spec explicitly
   calls out `/inference/go/openai/v1`. Passing that HTTPS same-provider base URL
   to `validate_provider_config` raises `ProviderConfigurationError`. Preserve
   the requested default, but support an explicit validated same-provider HTTPS
   correction without silently switching it. Document the actual supported
   overrides and retain credential-safe no-redirect handling.

3. **P2 — Stream completion can be dropped under bounded-queue backpressure.**
   `workspace/llm_service/app.py:191-205` uses `put_nowait` for the sole terminal
   `finished` marker and silently drops it when the 12-item queue is full. The
   consumer then waits forever in `queue.get()` after draining all events.
   Reproduction: request a stream from a provider producing 20 immediate deltas,
   consume the first response-body item, pause 50 ms, then consume remaining
   items with small pauses. Actual: 20 deltas and `done` are delivered, but the
   next iteration times out instead of EOF. The shipped client closes on `done`,
   which mitigates this normal-success path for that client, but the service
   contract itself leaves the response/lock open until disconnect; clients that
   read through EOF hang. Make terminal completion non-droppable and verify
   success, failure, and cancellation with a stalled consumer at the HTTP/service
   boundary.

4. **P2 — Tool-message pairing retains orphaned tool calls.**
   `workspace/llm_service/context.py:101-106` retains the entire AIMessage whenever
   any one of its calls has a result. With calls `good` and `orphan` but only the
   result for `good`, the returned AIMessage still includes both calls. This
   breaks the specified pairing invariant and can create a provider-rejected
   message sequence. `workspace/tests/test_remote_voice.py:49-57` explicitly
   asserts that invalid output instead of checking call/result IDs. Keep only
   complete valid tool-call groups or rebuild messages without orphan calls,
   preserving ordering, and assert exact IDs using real LangChain messages.

5. **P1 — Cancelling search leaves local network work running.**
   `workspace/llm_service/search.py:67` runs DDGS in `asyncio.to_thread` and cancels
   only the awaiter. The synchronous worker continues after turn cancel or the
   wrapper timeout. Rapid interrupted turns can accumulate obsolete running or
   executor-queued searches; no service-owned concurrency bound or cancellable
   worker lifecycle exists. Reproduction below starts a blocking `_search_sync`,
   cancels `search()`, and confirms that its worker remains active after the
   cancellation task has completed. The DDGS timeout is a useful request bound,
   but does not establish cancellation of all local search work or prevent an
   executor backlog. Implement cancellable owned work or a clearly bounded
   worker/admission lifecycle that retires obsolete searches before replacements
   accumulate, and prove replacement-turn recovery with a real worker-boundary
   fake. Do not describe this as provider-side cancellation.

6. **P1 — Contextual search planning discards the conversation.**
   `workspace/llm_service/remote_provider.py:97-109`, `:142-143` sends the search
   agent only the latest utterance. A follow-up such as “search for more current
   details about that” after a hero/item search has no referent in the planner,
   although the final answer stage receives prior history. The search query
   therefore cannot reliably retrieve the subject the user is discussing. Pass
   bounded useful session context to the search-planning model, and prove an
   explicit contextual search follow-up through the service plus real LangChain
   graph with a deterministic downstream model/tool. The current HTTP follow-up
   test only records generic FakeProvider input and never plans a query.

7. **P2 — Offline Local Voice attempts to download tokenizer data at runtime.**
   `workspace/llm_service/context.py:11-17` calls
   `tiktoken.get_encoding('cl100k_base')` without bundling its encoding asset.
   Neither Dockerfile nor model preparation primes that cache. With a cold cache,
   the installed tokenizer tries to load the OpenAI-hosted encoding asset before
   reaching the fallback. Local LLM has no egress, so this violates its no-remote-
   calls behavior and subjects history accounting to network failure/delay on
   repeated requests. Reproduction below intercepts the loader and proves the
   attempted asset URL without issuing a request. Bundle/verify the asset during
   preparation or use a fully local tokenizer/accounting strategy, preferably the
   local model's tokenizer for its budget. Verify a cold-cache offline turn.

8. **P2 — README's Remote preparation order requires an unset key.**
   `README.md:9-14` instructs Remote download before configuring the provider key;
   `compose.remote.yaml:25` makes that key required during interpolation of the
   complete graph, even for `run --rm download`. With `--env-file /dev/null`,
   configuration fails before model preparation starts. The later start section
   is the first instruction to populate the key. Give exact working preparation
   instructions that either set the key first or use a credential-independent
   preparation graph. Verify from a clean configuration, not an already-populated
   `.env`.

### Deterministic reproduction details

Run the following from the repository root with the temporary Python environment
used for the 14 contract tests. The environment location is incidental; a clean
Python 3.12 environment with the pinned core/provider dependencies is equivalent.
These commands use synthetic credentials/messages and do not call external APIs.

Real LangChain single-search flow, expected success at the default budget:

```sh
PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python - <<'PY'
import asyncio
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from llm_service.remote_provider import OpenCodeLangChainProvider
from llm_service.search import WebSearch, make_langchain_search_tool
class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self
class Search(WebSearch):
    async def tool_payload(self, query): return '{"results":[]}'
async def run():
    for limit in (3, 4):
        provider = object.__new__(OpenCodeLangChainProvider)
        provider.model = Model(responses=[
            AIMessage(content='', tool_calls=[{
                'id':'call-1', 'name':'web_search', 'args':{'query':'latest patch'}}]),
            AIMessage(content='Search complete')])
        provider.search_tool = make_langchain_search_tool(Search())
        provider.tool_iterations = limit
        try:
            await provider._search_context('latest patch')
            print(limit, 'PASS')
        except Exception as error:
            print(limit, type(error).__name__)
asyncio.run(run())
PY
```

Actual: `3 GraphRecursionError`; `4 PASS`.

Queue completion, expected EOF after `done`:

```sh
PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python - <<'PY'
import asyncio
from llm_service.app import LLMService, TurnRequest, create_app
class Burst:
    async def stream(self, text, history, cancelled):
        for index in range(20):
            yield {'type':'delta', 'text':str(index)}
async def run():
    application = create_app(LLMService(Burst(), 'remote'))
    endpoint = next(route.endpoint for route in application.routes
                    if getattr(route, 'path', '').endswith('/stream'))
    response = await endpoint('a'*32, 'b'*32, TurnRequest(text='burst'))
    body = response.body_iterator
    items = [await anext(body)]
    await asyncio.sleep(.05)
    try:
        while True:
            items.append(await asyncio.wait_for(anext(body), .3))
            await asyncio.sleep(.005)
    except StopAsyncIteration:
        print('EOF')
    except asyncio.TimeoutError:
        print('HUNG', len(items), items[-1].decode().strip())
    finally:
        await body.aclose()
asyncio.run(run())
PY
```

Actual: `HUNG 21 {"type":"done"}`. This intentionally isolates the response
iterator to impose deterministic backpressure; add a production HTTP-boundary
regression for the fix.

Search worker cancellation, expected terminated/retired work before replacement:

```sh
PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python - <<'PY'
import asyncio, threading
from llm_service.search import WebSearch
async def run():
    search = WebSearch()
    started, ended, release = threading.Event(), threading.Event(), threading.Event()
    def blocking(query):
        started.set()
        release.wait(1)
        ended.set()
        return []
    search._search_sync = blocking
    task = asyncio.create_task(search.search('current fact'))
    while not started.is_set(): await asyncio.sleep(.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    print('worker remains after cancel:', not ended.is_set())
    release.set()
asyncio.run(run())
PY
```

Actual: `worker remains after cancel: True`. The release event bounds and cleans
up the reproduction; it is not supplied by production cancellation.

Cold-cache tokenizer, expected no runtime external asset lookup in Local mode:

```sh
PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python - <<'PY'
import os, tempfile
from unittest.mock import patch
from llm_service.context import token_count
with tempfile.TemporaryDirectory() as cache:
    with patch.dict(os.environ, {'TIKTOKEN_CACHE_DIR': cache}):
        with patch('tiktoken.load.read_file', side_effect=RuntimeError('offline')) as read:
            print('fallback:', token_count('Hello'))
            print('asset requested:', read.call_args.args[0])
PY
```

Actual intercepted asset:
`https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`.
No network request is performed because the loader is replaced.

### Risks and unverified acceptance

- No provider account request was authorized or made. Actual endpoint/model
  compatibility, streamed tool-call assembly, search access, provider failure
  messages, and upstream cancellation are unverified.
- The proposed Linux ARM64 image has not been built by the Reviewer. Existing
  container tests cannot establish the new dependency lock installs successfully
  or the new Local/Remote process graphs become healthy.
- New mode disclosure, clickable sources, remote audio streaming, interruption,
  headphone behavior, reconnect, switching back to Local, and ten-turn latency
  measurements require a rendered browser and activated deployment. Existing
  generic fake tests do not complete those criteria.
- Context ownership and event/error/cancellation behavior need a documented
  internal service contract, including empty/abruptly-ended streams. Current
  client treats EOF without `done` as ordinary completion; there is no focused
  regression proving a truncated provider stream is rejected.
- The new LocalGemmaProvider loads GGUF directly and does not reproduce the old
  all-model receipt verification in voice startup. Preserve and verify the
  intended model-integrity boundary when completing Local regression.
- The intentional recognized-text/reply logging remains explicitly requested by
  the user. It retains the unchanged SPEC-001 no-content-logs acceptance conflict;
  SPEC-002 must report that conflict rather than claim the earlier criterion is met.
- No real key was read, embedded in the review, or emitted by verification. Remote
  Compose was validated with a synthetic placeholder, and no runtime `.env` or
  container secret dump was taken.

**Final Cycle 1 result: GREEN_REVIEW FAIL.** Address the concrete findings, rerun
focused real-graph/client/worker regressions plus the existing suites, and request
independent re-review. A later code-review PASS must remain separate from full
live acceptance until the user-owned build, key, and browser scenarios exist.

## Cycle 2

**Independent GREEN_REVIEW: FAIL / REQUEST_CHANGES.** The eight Cycle 1 findings
have targeted implementation fixes, and the normal search graph, pairing,
backpressure, worker termination, configuration, and client success paths now
pass their checks. Two reproducible error/cancellation defects and required
contract/readiness documentation remain. Full live acceptance remains unverified
under the user's explicit no-live-call/no-stack-change constraints.

Review used the same original user request, canonical acceptance criteria, project
AGENTS guidance, spec-driven-tdd skill, and acceptance-boundary guide as Cycle 1.
The spec diff changes only implementation ToDo checkboxes; its acceptance criteria
were not weakened. The implementation report is intentionally pending until after
this review. No source, tests, specification, deployment, model, or credentials
were changed by the Reviewer, and no build, restart, live inference/search,
microphone capture, or commit was performed.

### Verification and results

- Focused Python 3.12 environment, with pinned LangChain 1.4.8 and HTTPX 0.28.1:
  - `PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python -m unittest discover -s workspace/tests -p 'test_remote_voice.py'`: 18 PASS.
  - Same command with `test_langchain_provider.py`: 2 PASS.
  - Same command with `test_llm_client.py`: 2 PASS.
- `docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_*.py'`: 40 tests, 4 expected dependency skips, PASS in 5.406 seconds. Existing container and mounted source only; output captured in a temporary review log. This does not establish the new image installs or runs.
- `node --test workspace/tests/audio_capture.test.cjs workspace/tests/browser_playback.test.cjs`: both suites PASS.
- Both Compose modes validate using `--env-file /dev/null` with no key, as does Remote configuration using a synthetic placeholder. `--profile setup` JSON topology inspection confirms neither voice nor download receives `OPENCODE_GO_API_KEY`; only LLM receives it and `llm_egress`; only proxy publishes a port.
- `git diff --check`: PASS.
- Reviewer-run real subprocess reproduction: five concurrent search requests with `max_concurrent=2` spawn exactly two sleeping owned workers. Cancelling all requests terminates both workers, produces no additional workers, and completes within a two-second guard. No DDGS or live network activity occurs.
- Reviewer-run tokenizer check: in a fresh Python process, copy the known encoding asset into a newly created cache directory, replace `tiktoken.load.read_file` with an assertion, then call production `token_count`. Actual count 8, network loader calls 0. Dockerfile now primes this asset at build time, copies `/opt/tiktoken` to runtime, and sets the runtime cache there. This resolves the runtime cold-cache download design issue; actual new image build remains user-owned and unverified.
- The implementer reports byte-for-byte lock regeneration and 53-package compatibility. Those two checks were not independently repeated in Cycle 2; dependency pins/lock and Dockerfile were inspected.

### Cycle 1 closure matrix

| Cycle 1 finding | Exact new evidence | Outcome |
| --- | --- | --- |
| F1 default agent recursion | `LangChainGraphTests.test_real_agent_runs_one_search_and_plans_from_bounded_history` invokes production provider streaming using the actual pinned `create_agent`, one real typed fake search tool, a tool-call AIMessage, final model messages, and asserts `tool_call`, `sources`, `delta`. Production graph limit is now `2 * tool_iterations + 1`, default 7. | FIXED for normal one-search path; failure-code issue C2-F1 below |
| F2 endpoint override | Validator accepts the two exact HTTPS same-provider base paths and retains the requested default/model; configuration test covers both. Host/userinfo/query/fragment/port and path checks inspected. README documents only the explicit correction. | FIXED |
| F3 queue EOF | `test_stream_reaches_eof_after_backpressured_burst` runs the Cycle 1 slow-consumer setup and observes all 20 deltas, `done`, and EOF. Completion now uses a separate event rather than a droppable queue marker. Error/empty/cancel HTTP tests also finish. | FIXED |
| F4 orphan tool calls | Fake helper test and real LangChain AIMessage/ToolMessage test assert exact retained call/result IDs. Mixed AIMessage is copied and pruned, including additional wire kwargs/chunks. | FIXED for tested mixed/orphan grouping |
| F5 search worker cancellation | Search now owns a disposable subprocess and holds a bounded semaphore slot through termination. Existing fake-process test passes; Reviewer real sleeping-process reproduction confirms two slots, actual process exit, and no replacement backlog after cancellation. Worker environment whitelist excludes provider key. | FIXED; client read-task cleanup is separately C2-F2 |
| F6 contextual planner | Real graph test asserts prior user and assistant messages reach the planning model along with the contextual latest utterance. Service supplies token-trimmed history and planner caps it at eight messages. | FIXED |
| F7 runtime tokenizer download | Encoding asset is primed in builder and copied into the immutable runtime cache. Fresh-process copied-asset test with loader blocked succeeds. | FIXED by design and cache test; new image NOT VERIFIED |
| F8 keyless Remote preparation | Required interpolation removed; keyless Remote config validates; setup profile contains download command limited to STT/TTS and no key environment. Missing key produces 503 configuration readiness error through HTTP test. | FIXED |

### Full SPEC-002 acceptance matrix

| Criterion | Boundary/evidence reviewed | Status |
| --- | --- | --- |
| Two launch modes | Keyless merged config validation; correct complete graph, model mounts, networks, health dependencies, loopback proxy; exact README prepare/start/stop/switch commands. No new deployment run or actual mode switch. | STATIC PASS; runtime NOT VERIFIED |
| Separate LLM boundary | Production VoiceRuntime delegates to HTTP client. `LLMClientIntegrationTests.test_client_streams_through_service_api_and_clears_context` sends the actual client request through HTTPX ASGITransport to real service routes, asserts delta content, retained history, DELETE removal. ASGITransport buffers the response, so this proves request/response integration rather than live network streaming timing. Required protocol documentation absent (C2-F3); no full voice/audio-to-service end-to-end check. | PARTIAL |
| Live remote provider | Explicit requested default/model, two validated same-provider HTTPS paths, no redirects, zero automatic retries. Missing-key HTTP failure tested. Actual account, streaming/tool support and invalid-key live response not tested. | NOT VERIFIED |
| LangChain agent/context | Real graph one-search/context test and exact pairing tests pass. HTTP service tests verify complete-pair history budgets, isolation, and clear. Actual remote follow-up remains unverified. | DETERMINISTIC PASS; live NOT VERIFIED |
| Real search | Bounded typed query/result payload, process ownership, semaphore, timeout, URL filtering inspected and faked tests pass. Search-specific failures are lost at the actual provider graph boundary (C2-F1). No live search/grounded spoken response. | PARTIAL / C2-F1 |
| Voice streaming | Existing incremental speech and browser playback regressions pass; remote text/tool/source forwarding remains implemented. No activated Remote browser, real Whisper-to-provider-to-b_lez turn, or streamed tool-chunk/provider reasoning compatibility run. | PARTIAL; live NOT VERIFIED |
| Interruption/recovery | Service cancel/discard-history HTTP test, queue EOF regression, process exit/admission reproduction, existing stale-turn/playback/disconnect checks pass. Client coroutine cancellation leaves child read pending (C2-F2). Upstream provider-side cancellation is not verified. | PARTIAL / C2-F2 |
| Local regression | Existing container suite passes; extracted Gemma and offline topology inspected; tokenizer asset packaged. Full Gemma conversation through extracted service has not run; inherited SPEC-001 audio/performance/privacy limitations remain. | PARTIAL |
| Network/credentials | Static profile-inclusive Compose inspection confirms secret and egress isolation; HTTPS/no redirect and redacted codes inspected; build-context allowlist and ignored `.env` retained. No actual new runtime network/log inspection. | STATIC PASS; runtime NOT VERIFIED |
| Data/tool boundaries | UI mode/disclosure/source-link implementation retained, no arbitrary shell or URL fetch tools, local STT/TTS, ephemeral context, tracing disabled. No new rendered UI/physical audio check. Requested content logging still conflicts with inherited SPEC-001 criterion. | PARTIAL |
| Reproducibility | Dependencies/hash lock/base digest pinned; image build primes tokenizer; read-only mounted-source workflow retained. No new image build/ARM64 installation run by Reviewer. | STATIC PASS; build NOT VERIFIED |
| Focused checks | 18 contract/helper, 2 real graph, 2 actual-client API tests pass on host; 40 container tests with 4 dependency skips; 2 Node suites. Error mapping and cancellation gaps remain as concrete repros below. | PARTIAL |
| Live browser acceptance | No ten remote turns, search/no-search latency split, interruption p50/p95, headphones, reconnect, physical output, or Local mode switch measured. | NOT VERIFIED |
| Criterion report | Required implementation report intentionally follows this review. It must keep all live/build limitations explicit and distinguish existing container tests from new deployment validation. | PENDING |

### Findings

1. **C2-F1 — P2: Search-specific failures become a generic provider failure.**
   `workspace/llm_service/remote_provider.py:165-179` maps `SearchFailure` by its
   class name and ultimately replaces it with `provider_request_failed`. Thus
   `search_rate_limited`, `search_timeout`, `search_unavailable`, and invalid-query
   errors never reach the service's existing safe-code handling when raised from
   the real search graph. A rate-limited search is incorrectly presented as an
   unspecified provider request failure, leaving no actionable search diagnosis.
   Reviewer reproduction uses the real `create_agent` and typed tool, raises
   `SearchFailure('search_rate_limited')`, and observes
   `RuntimeError('provider_request_failed')`. Preserve recognized SearchFailure
   codes through this boundary and add real-graph-to-service tests for throttling,
   timeout, and unavailable search. Do not expose exception bodies or credentials.

2. **C2-F2 — P2: Cancelling a voice client task can leak its child stream read.**
   `workspace/llm_client.py:59-76` creates a separate `anext(lines)` task. It only
   cancels/awaits that task when the polling loop itself notices `cancel_event`;
   cancellation of the consuming coroutine while inside `await asyncio.wait`
   bypasses that branch. The response context closes, but the owned read task is
   never cancelled or gathered. The voice disconnect flow sets turn cancellation
   and then cancels turn tasks, so this is a relevant teardown shape. With a valid
   deterministic HTTPX blocking stream, setting the turn event and cancelling its
   consumer closes the response but leaves one pending child read. Repeated
   teardown can retain tasks or produce unhandled stream exceptions. Cancel and
   await each owned read in a finally block for all exit paths, and add a client
   cancellation test asserting no child tasks remain after response closure and
   cancel notification. No claim about actual provider-side abort is required.

3. **C2-F3 — P2: Required internal streaming/cancellation contract is undocumented.**
   The spec Design Statement requires one documented contract. README, AGENTS,
   `llm_service/__init__.py`, and service/client module docstrings provide no
   protocol description beyond implementation code. Document the stream request
   and identity format, event ordering/schema, context owner/budget/lifecycle,
   success/error/empty/truncated EOF semantics, cancel/delete behavior, and local
   versus unverified provider-side cancellation. The implementation ToDo currently
   marks this complete; a reviewer or client author should not have to derive it
   by reading private handlers. A project-local protocol section/file is sufficient.

4. **C2-F4 — P2: README overstates invalid-key readiness behavior.**
   `README.md:40` states an invalid key leaves Remote Voice unready, while
   `workspace/llm_service/app.py:66-72` marks every constructed provider ready and
   provider configuration validates only a nonempty key. A nonempty invalid key
   therefore reports ready until inference fails, and remains ready afterward.
   README's earlier configuration-only health disclaimer is accurate; align the
   later claim with it. State that empty keys fail configuration readiness and
   invalid nonempty keys produce a redacted error when an actual request occurs.
   Do not add an unauthorized live account probe merely to make the prose true.

### Reproductions for remaining defects

Use the same temporary Python 3.12 environment or an equivalent pinned environment,
from the project root. Both reproductions are deterministic and make no network
requests. They use the actual pinned LangChain/HTTPX application abstractions.

Search code propagation, expected `search_rate_limited`, actual
`provider_request_failed`:

```sh
PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python - <<'PY'
import asyncio
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from llm_service.remote_provider import OpenCodeLangChainProvider
from llm_service.search import WebSearch, SearchFailure, make_langchain_search_tool
class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs): return self
class Search(WebSearch):
    async def tool_payload(self, query): raise SearchFailure('search_rate_limited')
async def run():
    provider = object.__new__(OpenCodeLangChainProvider)
    provider.model = Model(responses=[AIMessage(content='', tool_calls=[{
        'id':'one', 'name':'web_search', 'args':{'query':'latest patch'}}])])
    provider.tool_iterations = 3
    provider.search_tool = make_langchain_search_tool(Search())
    try:
        print([event async for event in provider.stream('latest patch', [], asyncio.Event())])
    except Exception as error:
        print(type(error).__name__, str(error))
asyncio.run(run())
PY
```

Read-task ownership, expected zero pending children, actual one:

```sh
PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python - <<'PY'
import asyncio, threading, httpx
from llm_client import LLMServiceClient
class Stream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False
    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b''
    async def aclose(self): self.closed = True
async def run():
    baseline = set(asyncio.all_tasks())
    stream, cancel = Stream(), threading.Event()
    def handle(request):
        if request.url.path.endswith('/cancel'):
            return httpx.Response(202, json={'cancelled': True})
        return httpx.Response(200, stream=stream)
    client = LLMServiceClient('http://llm', transport=httpx.MockTransport(handle))
    async def consume():
        return [event async for event in client.stream('a'*32, 'b'*32, 'hello', cancel)]
    task = asyncio.create_task(consume())
    await stream.started.wait()
    cancel.set()
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
    await asyncio.sleep(.01)
    pending = [item for item in asyncio.all_tasks() if item not in baseline and not item.done()]
    print('closed:', stream.closed, 'pending child reads:', len(pending))
    for item in pending: item.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
asyncio.run(run())
PY
```

Actual: `closed: True pending child reads: 1`. The reproduction explicitly cleans
up the leaked task; production currently does not.

### Remaining risks and acceptance limits

- The requested URL/model, real account key, actual streaming/tool behavior and
  live DDGS connectivity remain unverified by instruction. No key was inspected,
  printed, copied to the review, or passed to browser/voice/download.
- Worker cancellation is now supported by actual local process exit evidence;
  provider-side remote request cancellation remains unverified.
- The new image and service graph have not been built/activated by Reviewer.
  Existing container tests use the old dependencies and skip four new tests;
  those four tests pass separately against the pinned host core packages.
- Fresh-cache tokenizer behavior is proven for the copied asset and inspected
  image recipe, not by a new built-container run.
- No new rendered Remote UI, real audio conversation, ten-turn latency measurement,
  interruption/headphone/reconnect or mode-switch acceptance was performed.
- Inherited SPEC-001 text-log privacy conflict and outstanding live acceptance
  remain open. Preserve that distinction in the implementation report.

**Final Cycle 2 verdict: GREEN_REVIEW FAIL / REQUEST_CHANGES.** Fix C2-F1 through
C2-F4 and repeat independent code review. Live/build acceptance remains a separate
pending stage even if the code review subsequently passes.

## Cycle 3

**Code-review verdict: PASS / APPROVED.** All concrete Cycle 1 and Cycle 2 code
findings are resolved in the reviewed implementation, and the relevant
deterministic checks pass. No additional high-confidence merge-blocking code
finding was identified. **Full SPEC-002 acceptance remains NOT COMPLETE**: the
new image, configured account, live search, actual browser voice journey, and
required measurements have not been tested under the user's explicit constraints.
This verdict approves the implementation at the code-review boundary; it does not
claim the outstanding live acceptance criteria have passed.

Reviewed the same original user request and unchanged acceptance criteria using
project AGENTS guidance, the spec-driven-tdd skill, and acceptance-boundary guide.
The acceptance text remains unchanged; the only spec diff is implementation ToDo
checkboxes. The implementation report is intentionally pending until after this
review. Reviewer modified only this review artifact, made no commits, and did not
build, restart, stop, or change the running stack, use a real key, call OpenCode or
live search, or start microphone capture.

### Verification performed

All commands ran from the repository root.

| Check | Result |
| --- | --- |
| `PYTHONPATH=workspace /tmp/el-voice-spec002-check/bin/python -m unittest discover -s workspace/tests -p 'test_remote_voice.py'` | 20 tests PASS, 1.021 seconds |
| Same command with `test_langchain_provider.py` | 3 tests PASS, 0.235 seconds; actual pinned LangChain 1.4.8 graph |
| Same command with `test_llm_client.py` | 3 tests PASS, 0.183 seconds; actual pinned HTTPX 0.28.1 client |
| `docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_*.py'` | 44 tests, 6 expected dependency skips, PASS in 5.414 seconds; existing container only |
| `node --test workspace/tests/audio_capture.test.cjs workspace/tests/browser_playback.test.cjs` | Both suites PASS |
| `docker compose --env-file /dev/null -f compose.yaml -f compose.local.yaml config --quiet` | PASS without remote credentials |
| Same Compose command with `compose.remote.yaml` | PASS without credentials for model preparation; runtime key validation remains in the LLM application |
| `git diff --check` | PASS |
| Reviewer additional real-graph search-error reproduction | `search_rate_limited`, `search_timeout`, `search_unavailable`, `invalid_search_query` each survive the actual typed-tool/agent/provider boundary and `safe_error_code` unchanged; all four PASS, no live calls |

The six container skips are the three LangChain and three HTTPX tests, because the
old image lacks the new pinned packages. All six ran successfully in the focused
Python 3.12 environment. The existing container suite does not verify the new
image build, image dependencies, selected mode topology, or real model/provider
integration. Earlier lock compatibility/reproducibility and no-network tokenizer
cache checks were inspected/reused as described in Cycle 2, not rerun needlessly.

### Cycle 2 closure matrix

| Finding | Exact behavior/test evidence | Verdict |
| --- | --- | --- |
| C2-F1 search errors lost | `remote_provider.py` imports SearchFailure, recognizes known errors in tool error messages/payloads, and rethrows SearchFailure rather than remapping it. `LangChainGraphTests.test_search_rate_limit_survives_real_agent_and_service_error_mapping` sends a real service HTTP request through production provider and actual LangChain graph with a failing typed search tool; response ends in `error.code == search_rate_limited`. Reviewer repeats all four recognized codes through the actual graph and safe mapper. | FIXED |
| C2-F2 child read leak | `LLMServiceClient.stream` cancels/awaits the owned `anext(lines)` task in its per-read finally block. `LLMClientIntegrationTests.test_consumer_cancellation_closes_owned_read_and_notifies_service` starts a blocked HTTPX response, sets the turn event, cancels its consumer, then asserts the read was cancelled, read task is done, response is closed, and cancel endpoint was notified. | FIXED |
| C2-F3 missing protocol docs | README Internal LLM service contract now documents private endpoint/path, JSON text input and hex IDs, context ownership/budget/lifecycle, NDJSON events/order, terminal success/error/empty/truncated EOF, cancel/delete, local Gemma/search cancellation, and the unverified upstream cancellation limit. | FIXED |
| C2-F4 readiness claim | README now distinguishes an empty key causing unready configuration from a nonempty invalid key producing an authentication error on a real request. Configuration-only health disclaimer remains. No live probe was introduced. | FIXED |
| Additional Local model-integrity concern | LocalGemmaProvider constrains the GGUF path to the configured model root/manifest asset and runs the existing downloader's `--verify --only` before importing/loading llama.cpp. New tests assert verification is invoked and failure prevents Llama load. Verifier checksum/receipt behavior was inspected. | RESOLVED at code/test boundary; actual Gemma service startup NOT VERIFIED |

Cycle 1 fixes remain intact: complete default real agent flow, contextual search
history, exact allowed endpoint correction, non-droppable completion/EOF,
pruned tool call/result IDs, owned cancellable search processes with bounded
concurrency, prepackaged tokenizer encoding, and keyless Remote preparation.
The passing focused suites exercise their regressions. Cycle 2's actual process
termination/admission and copied tokenizer asset checks remain relevant evidence;
those implementations have not changed in Cycle 3.

### SPEC-002 acceptance matrix

| Criterion | Code/test evidence and practical boundary | Current acceptance status |
| --- | --- | --- |
| Two launch modes | Both complete merged configs validate without keys; service dependencies, mounts, health checks, internal networks, Remote-only egress, loopback proxy, and README mode/switch commands inspected. | Code/static PASS; actual build/start/switch NOT VERIFIED |
| Separate LLM boundary | Voice production runtime uses the shared HTTP client; Local and Remote use one internal API. Actual client-to-FastAPI integration asserts returned text, context storage, and DELETE removal. Protocol documented in README. ASGITransport buffers the response, so this is API integration evidence rather than network streaming-timing evidence. | Code/API PASS |
| Live remote provider | Explicit requested default URL/model and documented same-provider alternative; async redirect following disabled; missing-key HTTP configuration test; provider errors are safely mapped. | Configuration/error code review PASS; real account streaming/tool compatibility NOT VERIFIED |
| LangChain agent/context | Actual create_agent tool graph with contextual history, complete call/result pairing, successful final answer, and HTTP service isolation/budget/clear tests pass. | Deterministic PASS; live follow-up quality NOT VERIFIED |
| Real web search | Typed DDGS tool, bounded payload/count/snippets/time, process cancellation/admission, public URL filtering; real graph-to-service throttling test and Reviewer four-code error reproduction pass. | Deterministic/code PASS; live search and grounded spoken result NOT VERIFIED |
| Voice streaming | Existing incremental TTS/audio/playback regressions pass; only delta text enters the speech path; tool/source forwarding present. No actual DeepSeek streamed reasoning/tool-chunk behavior or Remote browser-to-Whisper-to-provider-to-b_lez journey tested. | Deterministic partial; live NOT VERIFIED |
| Interruption/recovery | Service HTTP cancellation/history discard, backpressure EOF, client child-read cleanup, owned process exit/admission, existing stale-turn/playback/disconnect regression checks pass. Local native cancellation remains token-boundary/cooperative as documented. | Deterministic PASS for tested paths; real barge-in/provider-side cancellation/reconnect/restart NOT VERIFIED |
| Local regression | Existing regressions pass; Gemma extracted, manifest/receipt gate restored, tokenizer encoding prepackaged, internal-only network and no remote fallback. Full real Local conversation through the new service has not run. | Code/regression PASS; full Local runtime NOT VERIFIED |
| Network/credentials | Static topology remains only Remote LLM gets key/intentional egress; voice/download/proxy do not get key; LLM has no published host port; ignored `.env` and allowlisted image context; redirects disabled; redacted safe errors. | Static PASS; new runtime network/log inspection NOT VERIFIED |
| Data/tool boundaries | Mode/text-search disclosure and safe source-link implementation, local audio processing, no shell or arbitrary URL-fetch tool, ephemeral context and tracing disabled; requested text diagnostics retained. | Code PASS; rendered Remote UI and runtime storage inspection NOT VERIFIED; inherited privacy conflict remains |
| Reproducible deployment | Pinned dependencies/hash lock/image recipe inspected; build-time tokenizer asset cache; mounted-source workflow and exact mode commands documented. | Recipe/static PASS; new Linux ARM64 image build NOT VERIFIED |
| Focused checks | 20 contract/helper, 3 real graph, 3 HTTP client tests pass on pinned host core packages; existing container 44 with 6 dependency skips; 2 Node suites pass. | PASS at declared deterministic boundaries; no claim of full end-to-end live proof |
| Live browser acceptance | No ten-turn live run, search/no-search latency split, interruption p50/p95, headphones, physical output, reconnect, or actual mode switch. | NOT VERIFIED |
| Criterion-by-criterion report | Implementer explicitly plans to write the report after this review. It must preserve all statuses and evidence limits above. | PENDING |

### Findings

None. No unresolved high-confidence blocking code issue remains from the reviewed
scope. The outstanding live/build/report requirements above remain explicit
acceptance work and are not converted into passing criteria by this verdict.

### Risks and handoff limits

- The user must build and activate the new configuration before runtime
  acceptance. Neither the old running stack nor host fake integrations establish
  that the new Linux ARM64 image starts successfully.
- Actual OpenCode URL/model/account streaming/tool compatibility and DDGS search
  availability remain unverified. No key was inspected or exposed in review
  output; all checks use synthetic/no credentials and deterministic fakes.
- Live conversation, first audio before full generation, provider-specific
  reasoning/tool isolation, interruption, headphones, recovery, Local switching,
  and ten-turn search/no-search/interrupt measurements remain unverified.
- Actual upstream provider cancellation after local HTTP closure remains
  unverified; local client/search cancellation is directly covered.
- Requested transcript/reply logs retain the unchanged inherited SPEC-001 privacy
  conflict. Keep it explicit in the implementation report.
- Required implementation report is pending and must distinguish code review,
  deterministic checks, new-image validation, and live acceptance.

**Final Cycle 3 code-review result: PASS / APPROVED. Full SPEC-002 live acceptance:
NOT COMPLETE.** The implementation now supplies the requested two-mode architecture
and deterministic service/tool/client behavior; the remaining user-owned
build/account/browser scenarios must establish the final runtime outcome.
