# SPEC-001 Implementation Review

Status: Changes requested.

Parent task: [Local Dota voice conversation](SPEC-001-LOCAL-VOICE-CONVERSATION.md).
Evidence report: [Implementation report](SPEC-001-LOCAL-VOICE-CONVERSATION-IMPLEMENTATION-REPORT.md).

## Scope

Complete the current implementation in this repository on `codex/local-voice`.
Keep the current local models, Docker deployment, conversation context, and working
microphone capture. Implement the corrections and checks described below.
The original specification and its acceptance criteria remain unchanged. Record
unverified or failing criteria in the implementation report.

## P0: Required corrections

- [ ] **Fix session teardown under backpressure.** In `workspace/app.py`, a producer
  can remain blocked on a full outgoing queue while disconnect cleanup waits for
  turn tasks and the sender stops consuming. Cancel and await session tasks,
  release blocked producers, and bound cleanup. Verify disconnect with a full
  queue and a failed or stalled sender; a new session must then start successfully.
- [ ] **Verify actual inference cancellation.** Current Whisper and TTS calls keep
  computing after turn invalidation. Use engine cancellation or terminable workers
  where necessary, or demonstrate a measured bound on residual work. Cover LLM
  prompt evaluation as well as generation; do not claim a one-token bound from
  checks between streamed fragments. Repeated interruptions must not accumulate
  obsolete work or prevent the replacement turn from running.
- [ ] **Bound browser audio buffering.** Apply backpressure based on queued audio
  duration, with an explicit configurable limit. Interruption, stop, and disconnect
  must clear pending playback and reject all stale turn results.

## P1: Required improvements

- [ ] Add speech-text sanitization and configurable incremental chunking, including
  a short first chunk. Handle Markdown, URLs, emoji, punctuation, and long phrases
  without delaying all audio until generation completes.
- [ ] Validate settings at startup: types, ranges, sample rates, VAD parameters,
  thread counts, and chunk limits. Expose configuration failures through readiness.
- [ ] Put turn identity, sequence, and sample rate inside each binary audio packet;
  validate the packet before playback. Preserve unique identities across sessions.
- [ ] Add a reproducible pipeline self-test and focused regression checks for
  teardown, stale results, cancellation, and microphone resampling. Keep reusable
  check sources in `workspace/tests/`; temporary artifacts belong in `work/`.
- [ ] Add playback volume/mute and useful audio diagnostics. Preserve the existing
  browser-error reporting and requested transcript/reply/error logs.
- [ ] Separate session orchestration, model inference, and configuration where it
  makes these fixes easier to maintain; avoid an unrelated application rewrite.

## Regression checks

- [ ] Verify that 44.1 kHz and 48 kHz input both produce 16,000 output samples per
  input second, allowing for buffered frame tails. Every supported capture path
  must consume all input samples across callbacks.
- [ ] Verify that stop/start and reconnect cannot reuse message identities or append
  new replies to messages from a previous session.
- [ ] Preserve the selected VAD model's required input context and recurrent state;
  validate the wrapper against the bundled model interface.
- [ ] Verify that WebSocket closure immediately clears scheduled playback.
- [ ] Preserve in-session conversation history and test a contextual follow-up.
- [ ] Keep LLM token and audio queues bounded under a slow or stalled consumer.

## Acceptance evidence still required

- [ ] Run the original browser scenarios through Nginx on the target Mac: repeated
  turns, interruption during generation and playback, headphones without false
  self-interruption, stop/restart, and disconnect/reconnect with released resources.
- [ ] Verify relevant Russian Dota answers, follow-up context, and audible `b_lez`
  playback beginning while the LLM is still generating later text.
- [ ] Measure at least 10 representative turns and interruptions; report p50/p95
  for response and interruption latency, then agree targets. Distinguish server
  dispatch, browser scheduling, and actual output; report measurement limitations.
- [ ] Inspect runtime storage and logs. The user requested transcript/reply logging;
  this conflicts with the unchanged no-content-logs criterion. Report that criterion
  as unmet while those logs are enabled. Never log or persist conversation audio.
- [ ] Update `SPEC-001-LOCAL-VOICE-CONVERSATION-IMPLEMENTATION-REPORT.md` with
  commands, outcomes, remaining failures, and a status for every original acceptance
  criterion. Mark acceptance complete only after observation; an automated WebSocket
  test is not a microphone or physical audio-output test.

## Review evidence

The teardown finding comes from `workspace/app.py`: disconnect cleanup waits for
turn tasks before cancelling the sender, while a producer can remain blocked on
its outgoing queue. An isolated check reproduced a producer still waiting after
the session was marked stopped. The current model calls, buffering, settings, and
speech-text handling were also inspected in `workspace/app.py`,
`workspace/voice_pipeline.py`, and `workspace/static/app.js`.

Full browser acceptance was not rerun during this review. Record actual results
and remaining limitations in the linked implementation report.


## RED_REVIEW cycle 6 - 2026-11-04

Reviewed RED sequence through commit `8cad24c`. The approved parent task and the
acceptance checklist above remain unchanged. RED_REVIEW verdict: **FAIL**.

### Commands and observed results

- `docker compose run --rm --no-deps -e PYTHONDONTWRITEBYTECODE=1 voice python -m unittest discover -s /workspace/tests -p 'test_review_*.py'`: existing image; 13 test methods; exit 1; 15 failure records and 2 timeout errors (subtests account for multiple failure records).
- `node workspace/tests/audio_capture.test.cjs`: exit 0.
- `node workspace/tests/browser_playback.test.cjs`: exit 1; seven independent scenarios fail. The socket-close scenario currently passes with legacy metadata/raw PCM.
- Target-specific failures include missing packet framing, missing browser-consumption backpressure, unsanitized speech, ignored chunk limits, nine invalid configurations reaching ready, stale old-socket text, absent controls/diagnostics, STT cancellation forwarding, and blocked full-queue disconnect cleanup.
- Independent bundled VAD reference, log sink, history propagation, first audio before generation end, token-queue backpressure, and capture resampling/continuity checks pass.

### Concrete remaining RED blockers

1. Actual engine cancellation remains uncovered. The STT fake checks cancellation forwarding; LLM/TTS residual duration is supplied by `time.sleep(0.15)` in the fixture. These are orchestration tests, not measured bounds on Whisper, Silero, or Gemma CPU work. Add production-runtime/worker-boundary proofs for STT, TTS, LLM prompt evaluation, and LLM generation. Repeat interruptions, assert termination or independently measured residual bounds, bounded obsolete workers, replacement completion, and rejected stale results. Preserve real runtime locking or test the actual worker layer rather than replacing it with lock-free methods.
2. The corrected browser packet contains one second of PCM, but the harness drains a source whenever buffering reaches approximately two seconds, before offering further chunks. Consequently the fixture itself prevents overflow; an unbounded scheduler that emits a `playback_queue` message can pass. Exercise a genuinely stalled/over-capacity input boundary or a connected client/server credit loop. Assert both pending and scheduled duration, validate acknowledgement duration/identity fields, and prove eventual delivery without drops. The new real WebSocket server credit test is valid and currently fails because the third chunk arrives without an acknowledgement.
3. The socket-close scenario still feeds legacy metadata/raw PCM, while the required transport is self-describing binary packets. Use valid new-format audio for that scenario so protocol migration cannot cause an unrelated setup failure. Add delayed audio after stop/disconnect and a real stop/start journey that proves track release, fresh session identities, and no old message/audio mutation.
4. Mute/volume routing assertions improved, but markup regex plus fabricated DOM elements does not prove rendered visibility. Add rendered-browser visibility and activation assertions; verify playback through the controlled gain to its output. Physical audibility remains a separate manual check.
5. Add the required reproducible real-model pipeline self-test and an application/WebSocket two-turn contextual follow-up. Current history checks invoke `process_turn` directly and assert an assistant role rather than the exact prior response reaching the next real application turn.

### Acceptance-list summary

Python below means `workspace/tests/test_review_pipeline.py`; Browser means
`workspace/tests/browser_playback.test.cjs`; Capture means
`workspace/tests/audio_capture.test.cjs`.

| Checklist item | Test/action/assertion and review assessment |
| --- | --- |
| Teardown under backpressure | Python `WebSocketCleanupTests`: real socket, stalled sender, full queue and blocked putter; disconnect requires completed producer/sender tasks, released putters, replacement admission and unique IDs. Valid RED: session-release deadline expires. |
| Actual inference cancellation | Python STT and LLM/TTS tests cover fake orchestration only. Real engines, distinct prompt/generation stages and repeated interruptions remain uncovered; blocker 1. |
| Browser audio duration bound | Python `BrowserAudioBackpressureTests` receives two one-second chunks, requires no third before a consumption acknowledgement, then expects remaining chunks as acknowledgements arrive. Valid application-boundary RED. Browser harness covers eight one-second packets but pre-drains capacity; blocker 2. |
| Sanitization/configurable incremental chunking | Python `SpeakableResponseTests` checks synthesized text and metadata, fragmented Markdown/URL/emoji, configured first/max chunk limits and retained useful words. Early-audio gate test passes. Current sanitization/chunk failures are relevant RED. |
| Startup settings/readiness | Python `StartupValidationTests` runs lifespan and ASGI status requests; nine invalid type/range/rate/VAD/audio/chunk cases must expose error readiness, valid control must reach ready. Relevant RED; models are stubbed and real startup remains GREEN verification. |
| Binary packet identity/sequence/rate and validation | Python `AudioPacketTests` uses production synthesis/queue/sender path and asserts exact header/PCM. Browser independently specifies valid/truncated/invalid-rate/out-of-order/interrupted packet behavior. Packet prerequisites currently fail; relevant RED. |
| Pipeline self-test/focused regressions | Committed sources/commands are reproducible and focused tests exist. Complete real-model pipeline self-test remains missing; blocker 5. |
| Volume/mute, diagnostics, retained logging | Browser specifies source-to-gain routing and operating controls; rendered visibility still missing. Diagnostics case requires buffering row and browser error message, but currently fails before its error assertion. Python actual logger-sink transcript/reply/error/no-audio-sentinel test passes with valid even-length PCM. |
| Maintainable separation/no rewrite | RED commits change test sources only. Inspect production module responsibilities and scope during GREEN review. |
| 44.1/48 kHz resampling | Capture asserts 16000 emitted-plus-tail samples, less than one buffered frame and identical varying-audio output across callback partitions at both rates. Pass; appropriate algorithm boundary. |
| Stop/start/reconnect isolation | Real WebSocket teardown specifies new session/turn IDs; Browser old-socket text scenario reproduces stale appending. Stop/start with actual capture tracks and stale audio remains missing; blocker 3. |
| VAD context/recurrent state | Python wrapper is compared with separate direct bundled ONNX invocation carrying explicit context and recurrent tensors. Pass; independent reference is suitable. |
| Socket closure clears playback | Browser connects production socket handlers, schedules legacy audio, invokes closure and asserts sources stopped. Pass today; update protocol fixture and assert delayed results rejected. |
| Conversation history/contextual follow-up | Python two direct `process_turn` calls show prior user content and assistant role reaching stub history. Supplemental regression passes; exact application-boundary follow-up and real contextual quality remain pending. |
| Bounded token/audio queues | Python token test stalls consumer, asserts producer count at most six and exit after cancellation; pass. Server credit RED now proves duration gating is missing. Client pending-duration/acknowledgement proof remains blocker 2. |
| Original browser scenarios through Nginx | Mandatory target-Mac microphone/acoustic/lifecycle evidence remains deferred: repeated turns, both interruption stages, headphones without self-interruption, stop/restart, reconnect and resource release. Synthetic tests do not establish completion. |
| Russian Dota answers and audible overlapping b_lez | Real-model answer relevance/context, selected audible voice and playback before generation completion through Nginx require target-Mac observation; deferred. |
| Ten-turn latency and agreed targets | At least ten representative response/interrupt measurements, p50/p95, dispatch/scheduling/output distinctions, limitations and subsequent target agreement remain deferred. |
| Runtime storage/log inspection | Application sink test covers selected records without its audio sentinel. Inspect actual storage/application/proxy logs after a session. Keep no-content-logs criterion unmet while requested text logs are enabled; never persist/log conversation audio. |
| Honest report for every original criterion | Final report must record commands, results, failures and every original status. Physical microphone/output outcomes stay unverified until observed. GREEN/handoff requirement pending. |

The deferred physical acceptance evidence is not itself the reason for this RED
rejection. The remaining automated proving gaps above independently prevent PASS.
Only this review-cycle entry was appended; no source, tests, parent specification,
or acceptance checklist was edited, and no commit was created by the Reviewer.


## GREEN_REVIEW cycle 1 - 2026-11-04

Reviewed the implementation working tree based on `8cad24c`, including the final
configuration-load handling and first-token cancellation measurement additions.
Verdict: **REQUEST_CHANGES**. The original business task is not yet resolved.

### Verification

- Re-ran `docker compose run --rm --no-deps -e PYTHONDONTWRITEBYTECODE=1 voice python -m unittest discover -s /workspace/tests -p 'test_review_*.py'`: 14 tests, exit 0.
- Re-ran `node workspace/tests/browser_playback.test.cjs` and `node workspace/tests/audio_capture.test.cjs`: both exit 0.
- `git diff --check`: exit 0.
- Ran independent read-only probes against production Python and production browser JavaScript using the existing image and browser API harness. Results below demonstrate defects not caught by the passing suites.
- Independently opened the rendered page through `http://localhost:8765/`: Start, Mute, volume slider, headphone notice and ready state are visible; End session is hidden while idle. No microphone access or recording was initiated. This verifies rendering, not current server deployment, physical output or acoustic behavior.
- Inspected the offline real-model self-test and the final report. The implementer reports STT residuals 1867/2003/1523 ms; LLM generation 44/49/38 ms; first-token/prompt 26 ms with cancellation before the first token; TTS 1143 ms. These model runs were not independently repeated by the Reviewer. The report honestly leaves microphone, physical output, live interruption, ten-turn latency and requested text-log privacy acceptance incomplete.

### Findings to fix

1. **P1: A synthesized chunk longer than the audio limit can never be sent.** `workspace/app.py:synthesize_and_queue` requires `browser_buffered_ms + duration_ms <= max_buffered_ms` before emitting any bytes. A three-second TTS result with the default two-second limit blocked until the probe deadline with `outgoing.qsize() == 0`. No acknowledgement can make that chunk fit, even in an empty browser. Split synthesized PCM into duration-bounded framed packets, or otherwise guarantee every outgoing packet fits the configured limit. Add an application-boundary regression using speech audio longer than the limit and assert first audio and completion.
2. **P1: Browser/server credit accounting stalls normal playback and can release unsatisfied reservations.** The browser only sends `playback_queue` when scheduling audio, not when playback completes. For two one-second packets at `currentTime=0`, production JavaScript emitted sequence 0 with `consumed_ms=0`, `buffered_ms=1014.9999999999999`; sequence 1 with `consumed_ms=14.999999999999886`, `buffered_ms=2014.9999999999998`. Completing both sources emitted zero further consumption acknowledgements. The server rejects the second acknowledgement because it exceeds 2000 ms. With production Python, sequence 0's 1015 ms snapshot also overwrote an existing 2000 ms reservation, forgetting other queued/in-flight audio. `consumed_ms` is validated but unused. Track outstanding audio by packet/consumption, acknowledge actual completion, distinguish scheduling lead-in from queued audio, and retain reservations newer than an acknowledgement. Add a connected production client/server test that drains a response longer than the limit without fabricated acknowledgement payloads; include delayed, duplicate, stale and invalid acknowledgements and inspect the actual duration bound.
3. **P1: Incremental chunking corrupts text and bypasses configured limits.** `_speakable_chunks('Hello ')` strips the trailing separator; appending `'world'` produces `'Helloworld'`. The probe `_speakable_chunks('word ' * 25 + '.', max_chars=20, min_chars=12)` emits a 126-character first chunk because punctuation is considered before the maximum. Preserve separators/raw partial-token state across deltas and cap every emitted/final synthesis chunk regardless of punctuation. Test whitespace-only deltas, word boundaries and a long punctuated sentence delivered in one fragment, asserting actual TTS input.
4. **P2: Valid JSON with a non-object root still crashes before readiness exists.** Patching `_load_settings` to return `[]` caused `VoiceRuntime.__init__` to raise `AttributeError` at `self.config.get`, before `_validate_settings` can reject it. Validate the root type before accessing it, retain an error status, and test ASGI health/status for list/null/scalar settings as well as malformed JSON. Align accepted TTS rates with the browser's hardcoded 24000 Hz policy or support every accepted rate end to end.
5. **P1: Cancellation evidence is still incomplete and GREEN changed the failing fake behavior.** The LLM/TTS tests now add their own locks and cooperative event waits, replacing the prior uncooperative operations. Those fixture changes make the tests pass without proving native inference stops or has a production-enforced residual bound. The real-model script adds useful observed samples, but it does not test repeated replacement turns or representative long input/context; it prints PASS without asserting the prompt-stage observation occurred or that residual work satisfies a stated measured bound. Its TTS residual is measured from call start, approximately 20 ms before cancellation. Keep cooperative forwarding tests, restore an uncooperative worker-boundary regression, test repeated real cancellation/replacement without accumulating obsolete workers, measure from the cancellation timestamp, and report the tested input/context limits and remaining limitations.
6. **Acceptance remains incomplete in the running application.** The report confirms the Python service has not been restarted, so the changed browser protocol is paired with the old in-memory server protocol. Activate the corrected implementation and verify updated readiness/WebSocket behavior through Nginx, then complete the original microphone, audible overlap, interruption, headphone, recovery, storage/log and ten-turn latency scenarios before requesting full GREEN acceptance. Keep the no-content-logs criterion explicitly unmet while user-requested transcript/reply logging remains enabled; do not alter the parent criterion.

### Review-checklist acceptance matrix

Python means `workspace/tests/test_review_pipeline.py`; Browser means
`workspace/tests/browser_playback.test.cjs`; Capture means
`workspace/tests/audio_capture.test.cjs`.

| Checklist item | Evidence and GREEN assessment |
| --- | --- |
| Teardown under backpressure | Python `WebSocketCleanupTests` passes real socket/stalled sender/full-queue cleanup, released tasks/putters and replacement admission. Local source cancels sender/turn tasks and bounds turn cleanup. Automated correction verified; live resources still require runtime acceptance. |
| Actual inference cancellation | Cancellation-aware locks/segment checks and observed real-model residuals are useful. Native operations remain in flight, repeated real replacement testing is absent and fake locks/event waits weakened the RED. Finding 5; partial. |
| Bounded browser audio buffering | Python credit test and Browser scheduling test pass separately, but independent production probes reproduce oversize deadlock and missing/inconsistent consumption credit. Findings 1-2; fail. |
| Sanitization/configurable chunking | Existing tests pass removal/fragmented input/short-first limits and early audio. Independent probes show separator loss and a 126-character chunk with maximum 20. Finding 3; fail. |
| Settings/readiness validation | Nine invalid cases and valid control pass through ASGI with model loading stubbed. Non-object JSON crashes construction; accepted rate policy conflicts with browser validation. Finding 4; partial. |
| Self-describing binary packets | Actual synthesis-to-sender test passes UUID/sequence/rate/header/PCM assertions; Browser rejects malformed/rate/order cases. Session IDs remain UUIDs. Framing correction verified at automated boundary; deployed protocol remains old. |
| Reproducible self-test/focused regressions | Offline real-model self-test source now exists and implementer records success. It bypasses session/chunking/credit/browser integration and does not prove all cancellation requirements. Automated sources remain reusable; partial product evidence. |
| Volume/mute, diagnostics and logs | Source routes playback through GainNode; Node tests exercise gain and diagnostics/errors. Reviewer rendered inspection confirms controls/notice; implementer reports accessible mute toggle. Actual log sink test passes text/error/no-audio sentinel. Physical gain/audibility not observed. |
| Maintainable separation/no rewrite | Existing app/runtime split retained; no dependency or architecture replacement. Configuration validation lives in runtime. No blocking broad rewrite found. |
| 44.1/48 kHz resampling | Capture verifies counts, tails and partition-invariant varying samples; pass. |
| Stop/start/reconnect isolation | Real socket replacement UUID test and Node old-socket text/audio rejection pass. Node stop clears scheduled sources/context. Real microphone track release and repeated live stop/start remain unobserved. |
| VAD context/recurrent state | Independent direct ONNX reference comparison passes; automated regression satisfied. Real microphone threshold/noise/endpoint tuning remains pending. |
| Socket closure clears playback | Updated Browser test uses framed packets and confirms sources stop; pass at browser API harness boundary. |
| In-session contextual follow-up | New `WebSocketConversationTests` runs two real WebSocket turns and asserts exact prior user/assistant exchange reaches second inference. Pass with stub models; real Russian contextual quality remains pending. |
| Bounded token/audio queues | Token producer/backpressure/cancellation test passes. Outgoing count queue is bounded, but duration-credit and browser consumption defects remain; finding 2. |
| Original target-Mac scenarios through Nginx | Rendered page observed. Updated server not activated; repeated live turns, both interruption stages, headphones, restart/reconnect/resource release remain unverified. Incomplete. |
| Russian Dota answers and audible overlapping b_lez | Model self-test proves nonempty stages only; answer relevance/context quality and physical audible incremental playback remain unverified. Credit deadlocks currently prevent acceptance. |
| Ten-turn latency and agreed targets | No required end-to-end baseline or target agreement. Incomplete. |
| Runtime storage/log inspection/privacy conflict | Logger sink test omits audio marker; implementer reports offline self-test wrote no files. No full post-conversation runtime inspection. Requested text logging explicitly violates unchanged no-content-logs clause; criterion unmet as written. |
| Honest report for every original criterion | Report includes every original criterion and describes unverified/failed outcomes candidly. Add the independently reproduced defects and avoid saying self-test source is committed while it remains untracked. Full task completion must not be claimed. |

### Original-spec acceptance matrix

| Original acceptance criterion | Review status |
| --- | --- |
| Browser session and continuous listening | Partial: rendered controls and synthetic lifecycle checks; updated live microphone/resource journey pending. |
| Local speech onset and endpointing | Partial: real ONNX state check; pre-roll/noise/threshold tuning against microphone pending. |
| Accurate final transcription | Partial: implementer reports synthesized-speech Whisper round trip; representative microphone accuracy pending. |
| Local Dota response | Partial: real Gemma nonempty output reported; Russian relevance/context not assessed. |
| Incremental response playback | Fail: server early-queue unit test passes, but duration/credit defects block actual response streaming. |
| Automatic barge-in | Partial: cancellation forwarding/isolation mechanisms; repeated native work and live interruption acceptance incomplete. |
| Turn isolation | Automated UUID/stale-socket checks pass; acoustic replacement scenario not observed. |
| Echo behavior | Headphone notice rendered; no physical headphone/no-self-interruption verification. |
| Browser-to-container routing | Page reachable through proxy; changed server code has not been activated/verified. |
| Local operation and privacy | Not met as written: intentional text logs remain. Offline model operation reported; full runtime storage/log inspection pending. |
| Readiness and recovery | Partial: regression suite passes selected cases; non-object settings crash and updated deployment/recovery unverified. |
| Latency measurements | Not complete: ten-turn p50/p95 baseline and agreed targets absent. |
| Conversation/interruption scenarios | Not complete: required actual-browser scenarios remain outstanding. |
| Reproducibility | Automated commands and source reviewed; real-model report available, but complete running-product verification outstanding. |

Only this GREEN review entry was appended by the Reviewer. Prior review entries,
the acceptance checklists and the parent specification were preserved. No source
or tests were edited and no commit was created by the Reviewer.


## GREEN_REVIEW cycle 2 - 2026-11-04

Reviewed the full current implementation working tree and updated implementation
report against the unchanged parent spec and cycle 1 findings. Verdict:
**REQUEST_CHANGES**. Code-review PASS is not warranted because two production
credit-protocol defects remain. Original product acceptance is separately
incomplete and must not be inferred from passing automated checks.

### Verification and exact reproductions

- Existing-image command `docker compose run --rm --no-deps -e PYTHONDONTWRITEBYTECODE=1 voice python -m unittest discover -s /workspace/tests -p 'test_review_*.py'`: 17 methods, exit 0.
- `node workspace/tests/browser_playback.test.cjs`, `node workspace/tests/audio_capture.test.cjs`, and `git diff --check`: exit 0.
- Read-only production chunking probes now preserve `Hello ` plus `world` as `Hello world`. With maximum 20, the prior long punctuated input emits lengths `[19, 19, 19, 19, 19, 19, 6]`. Settings root `[]` now constructs an error/not-ready runtime instead of raising `AttributeError`.
- Production browser JavaScript was executed with the existing browser API harness, without editing files. After scheduling packets 0 and 1, it sent scheduling acknowledgements with sequences 0 and 1. Completing both sources sent completion acknowledgements with sequences **0 and 0**. For one-second fixture packets, exact completion payloads were `{sequence: 0, consumed_ms: 1015, buffered_ms: 999.9999999999998}` and `{sequence: 0, consumed_ms: 999.9999999999998, buffered_ms: 0}`. Both are older than the latest scheduling sequence 1.
- Production Python confirmed a completion acknowledgement at sequence 0 after scheduling acknowledgement 1 is rejected as out of order; reserved duration remains 1615 ms. Separately, with packet 0 sent, packet 1 reserved but unsent, ledger `{0: 800, 1: 800}` and reserved duration 1600 ms, an acknowledgement of packet 0 with buffered 815 ms changed the reserved total to **815 ms**, even though ledger `{1: 800}` remained. The correct total must retain the later 800 ms reservation (1615 ms in this setup).
- The implementer reports a separate offline/no-network real-model pass: STT residuals 1821/1590/1353 ms, LLM generation 31/28/56 ms, first-token/prompt 24 ms before first token, TTS 962 ms, generated speech duration 52.65 s. Reviewer inspected the script/report but did not repeat those model runs. Residual samples characterize the tested input; they are not a universal worst-case latency guarantee.

### Blocking findings

1. **P1: Completion acknowledgements carry the wrong packet sequence.** `workspace/static/app.js` stores the last received packet sequence in `audioSequence`, but `source.onended` sends `(audioSequence.get(turnId) || 1) - 1`. After packets 0 and 1, this reports 0, while the server has already accepted scheduling acknowledgement 1. The server rejects completion credit as out of order, so a response longer than available credit stalls. Report the actual highest received/acknowledged packet sequence consistently, and add a regression connecting production browser completion payloads to `receive_control`. The current Python completion helper uses the server's next sequence minus one, which does not match this production client expression and masks the bug.
2. **P1: Acknowledgements forget reserved but unsent audio.** `workspace/app.py:receive_control` sums later packet durations only when `packet_sequence <= audio_sent_sequence`. `synthesize_and_queue` reserves duration before enqueueing/sending, so a delayed earlier acknowledgement can erase reservations for packets still in the outgoing queue. This creates excess credit and permits later delivery beyond the configured duration bound. Preserve all later reservations, distinguish reserved/queued/sent state where needed, and add a stalled-sender test that processes an earlier acknowledgement while another reserved packet has not been sent.

### Recheck of cycle 1 findings

| Prior finding | Current evidence and outcome |
| --- | --- |
| Oversize TTS chunk deadlock | Addressed: PCM packetization uses configurable duration; application-boundary test drains a three-second synthesis as 15 x 200 ms packets with full byte count and completion. Pass for this correction. |
| Missing/inconsistent playback credit | Partly addressed: source completion now emits credit, per-packet duration ledger exists and acknowledgements are validated. Wrong completion sequence and unsent-reservation loss remain; findings 1-2. |
| Separator loss and chunk cap bypass | Addressed: prior exact probes now pass; new regression preserves separator and asserts hard caps. Sanitization/configured chunk suites pass. |
| Non-object settings crash/rate mismatch | Addressed: root type is checked during construction and retained as readiness error; tests cover non-object roots. Accepted TTS rate is restricted to browser-supported 24000 Hz. |
| Cooperative-only cancellation fixtures | Improved at the requested runtime boundary: new uncooperative leaf-engine test invokes production `VoiceRuntime` locking/iterator behavior across three replacement attempts for STT/LLM/TTS, proving no overlapping native calls and successful replacements. Native-call interruption remains cooperative/bounded by the current operation; reported real-model residual samples remain limited to tested inputs. TTS residual timestamp now starts at cancellation. |
| Running deployment and original acceptance | Still incomplete: voice Python process has not been restarted, and no microphone session was started. Report correctly retains partial/unmet statuses. This prevents full SPEC-001 acceptance independently of code-review status. |

### Review-checklist acceptance matrix

Python means `workspace/tests/test_review_pipeline.py`; Browser means
`workspace/tests/browser_playback.test.cjs`; Capture means
`workspace/tests/audio_capture.test.cjs`.

| Checklist item | Evidence and status |
| --- | --- |
| Teardown under backpressure | Python real WebSocket stalled-sender/full-queue test verifies task/putter release, replacement admission and fresh IDs; passes. |
| Actual inference cancellation | Production runtime locks are now tested with uncooperative leaf engines over repeated replacements; offline real-model residuals are reported. No claim of immediate native cancellation; actual microphone interruption remains unverified. |
| Bounded browser audio buffering | Packetization regression passes, but completion sequence and unsent reservation defects break real credit flow/bounds; fail. |
| Sanitization/configurable incremental chunking | Fragmented formatting/URL/emoji, useful text preservation, first/subsequent chunk limits, space preservation and early audio tests pass; prior probes fixed. |
| Startup settings/readiness | Invalid settings ASGI tests and non-object root checks pass; 24000 Hz rate policy matches browser. Updated live startup not performed. |
| Self-describing packets | Actual synthesis/queue/sender packet header/payload checks and browser rejection tests pass. New deployed WebSocket protocol remains inactive. |
| Pipeline self-test/regressions | Reusable offline real-model source exists; 17 focused regressions pass. Full browser-to-server flow remains insufficiently integrated, as the ACK reproduction demonstrates. |
| Volume/mute, diagnostics/logs | Browser harness verifies gain routing/control/error diagnostics; rendered controls were observed during cycle 1. Actual logger sink checks text/error records without audio marker. Physical output unverified. |
| Maintainable separation/no unrelated rewrite | Existing app/runtime/config architecture and deployment preserved. No unrelated rewrite found. |
| 44.1/48 kHz resampling | Capture count/tail/partition-continuity checks pass. |
| Stop/start/reconnect isolation | Synthetic stop/context cleanup, stale old-socket audio/text rejection and replacement UUID checks pass; real microphone release/restart remains pending. |
| VAD context/recurrent state | Separate direct bundled ONNX reference comparison passes. |
| WebSocket closure clears playback | Browser framed-audio close regression passes. |
| Conversation history/follow-up | Real Uvicorn/WebSocket two-turn test asserts exact prior exchange reaches next model request; passes with stub models. Real Russian contextual relevance unverified. |
| Bounded token/audio queues | Token queue/backpressure test passes; outgoing count bound exists. Duration reservations still fail in the unsent-packet case; finding 2. |
| Original browser scenarios through Nginx | Rendered page verified previously; updated Python process not restarted. Actual repeated turns, generation/playback interruption, headphones, stop/restart and reconnect/resource release remain unverified. |
| Russian Dota answers and audible b_lez overlap | Real models return nonempty output in reported self-test; answer quality/context and physical audible overlap remain unverified. |
| Ten-turn latency/targets | Required response/interruption baseline, p50/p95 and agreed targets remain absent. |
| Runtime storage/log privacy | Logger unit sink is clean of its audio sentinel; full post-conversation runtime inspection remains pending. User-requested transcript/reply logging keeps unchanged no-content-logs criterion unmet. |
| Honest implementation report | Report candidly identifies unactivated deployment, no microphone test, native residual work and partial/unmet original acceptance. Add the two independently reproduced credit defects and their fixes/results before handoff. |

### Original acceptance evidence still outstanding

- Updated Compose/Nginx/WebSocket deployment, actual continuous microphone capture and released tracks/resources after stop/disconnect/reconnect.
- Real microphone VAD onset/endpoint/pre-roll/noise tuning, transcription accuracy, relevant Russian Dota answers and contextual follow-up.
- Audible incremental `b_lez` playback while generation continues; barge-in during generation and playback; headphone operation without false self-interruption; stale speech never resuming.
- At least ten representative response/interruption measurements, p50/p95, distinction between scheduling and actual output, and agreed performance targets.
- Runtime storage/log inspection after a real session. No audio may persist or enter logs. The intentional text-logging conflict remains explicitly unmet under the unchanged parent spec.

Only this cycle-2 review entry was appended by the Reviewer. Prior review entries,
the acceptance checklists, source/tests/report and the parent specification were
preserved. No commit was created by the Reviewer.


## GREEN_REVIEW cycle 3 - 2026-11-04

Code-review verdict: **PASS**. No remaining high-confidence blocking code findings
were identified in the reviewed SPEC-001 correction scope. The cycle-2 credit
findings and cycle-1 implementation defects are addressed.

Full original SPEC-001 acceptance remains **NOT COMPLETE**. This code-review PASS
does not certify the unactivated running deployment, microphone/acoustic behavior,
physical audio, ten-turn performance baseline, or the unchanged privacy criterion.
It is not evidence that the complete original business task has been observed and
accepted end to end.

### Verified commands and results

- `docker compose run --rm --no-deps -e PYTHONDONTWRITEBYTECODE=1 voice python -m unittest discover -s /workspace/tests -p 'test_review_*.py'`: 18 methods, exit 0, using the existing image.
- `node workspace/tests/browser_playback.test.cjs`: exit 0.
- `node workspace/tests/audio_capture.test.cjs`: exit 0.
- `git diff --check`: exit 0.
- Independent production-JavaScript probe: after packets 0 and 1, completing both sources now emits completion acknowledgements with sequences **1 and 1**, consumed durations approximately 1015/1000 ms and buffered durations approximately 1000/0 ms. Sequence no longer moves behind the latest scheduling acknowledgement.
- New `PlaybackAcknowledgementTests.test_acknowledgement_keeps_reserved_unsent_packet_duration` invokes production `receive_control` with packet 1 still in the outgoing queue. It preserves ledger `{1: 800.0}`, queue size 1, and effective reservation **1615.0 ms** after an earlier 815 ms client snapshot. The test passes.
- The real Uvicorn/WebSocket long-audio credit test now calculates completion acknowledgements from the latest packet actually received, matching the production browser protocol instead of server producer state. It drains all 15 x 200 ms packets from three seconds of audio and reaches turn completion.
- Reported offline/no-network real-model evidence remains STT residuals 1821/1590/1353 ms, LLM generation 31/28/56 ms, first-token/prompt residual 24 ms with cancellation before first token, TTS residual 962 ms, and synthesized duration 52.65 s. Inference source has not changed since that run. The Reviewer inspected the source/report but did not independently repeat that model run.

### Prior findings rechecked

| Prior finding | Cycle-3 result |
| --- | --- |
| TTS result larger than duration budget never sent | Fixed: duration-limited PCM packetization; real WebSocket three-second/200 ms packet regression passes with preserved byte count and completion. |
| No completion credit / wrong completion sequence | Fixed: source completion reports the latest received sequence via `audioSequence.get(turnId) ?? 0`. Independent probe and Node monotonicity/sequence-1 completion assertions pass. |
| Earlier acknowledgement drops queued reservations | Fixed: all later reserved packet durations are retained regardless of sent sequence. New queued-reservation regression passes. |
| Stream separator loss / punctuated chunk cap bypass | Fixed: separator-preservation and hard maximum regressions pass; prior `Hello world` and maximum-20 reproductions remain corrected. |
| Non-object settings crash / incompatible accepted sample rates | Fixed: root failures remain available as readiness errors; accepted TTS rate matches browser-supported 24000 Hz. Startup tests pass. |
| Cooperative-only replacement-worker tests | Addressed at production runtime boundary: uncooperative leaf-engine tests exercise actual runtime locking across three replacements for STT/LLM/TTS. Model residual measurements are reported honestly; immediate native termination is not claimed. |
| Updated deployment / full original acceptance | Outstanding verification, explicitly disclosed in the report. Voice service has not been restarted and no new microphone session was started. |

### Acceptance matrix

Python means `workspace/tests/test_review_pipeline.py`; Browser means
`workspace/tests/browser_playback.test.cjs`; Capture means
`workspace/tests/audio_capture.test.cjs`.

| Review checklist item | Code evidence / remaining acceptance limitation |
| --- | --- |
| Session teardown under backpressure | Python real WebSocket full-queue/stalled-sender cleanup and replacement admission regression passes. Live microphone resources still require observation. |
| Actual inference cancellation | Production lock/uncooperative replacement regressions and reported real residual samples available. Native STT/TTS finish their active operation; measurements characterize tested inputs, not universal worst-case latency. Live barge-in remains unverified. |
| Bounded browser audio buffering | Packetization, source-completion credit, monotonic sequences, acknowledgement validation and queued reservation preservation now pass focused regressions. |
| Sanitization/configurable incremental chunking | Actual synthesis arguments, fragmented Markdown/URL/emoji, first/max chunk limits, separators and early audio covered by passing tests. |
| Settings validation/readiness | ASGI readiness invalid/valid configuration tests, non-object roots and protocol rate policy pass. Updated live service initialization remains pending. |
| Self-describing binary packets | Synthesis-to-sender framing and Browser packet validation/isolation regressions pass. |
| Reproducible self-test/regressions | Reusable real-model self-test source and 18 focused regressions present; existing image used. Full running-browser journey remains outstanding. |
| Volume/mute, diagnostics/logs | Gain-routing/control/diagnostic tests pass; rendered controls observed in earlier review; actual logger sink checks omit audio payload marker. Physical output is unverified. |
| Maintainable separation/no unrelated rewrite | Existing local runtime/app/config architecture and Docker deployment preserved; no blocking scope/maintainability issue identified. |
| 44.1/48 kHz resampling | Capture counts, tails and partition-continuity regressions pass. |
| Stop/start/reconnect identity isolation | UUID replacement and stale socket audio/text/stop cleanup regressions pass; actual microphone track release/restart remains unverified. |
| VAD context/recurrent state | Independent direct bundled ONNX reference regression passes; microphone threshold/noise tuning remains unverified. |
| WebSocket closure clears playback | Framed-audio Browser regression passes. |
| In-session contextual follow-up | Real Uvicorn/WebSocket two-turn test passes exact prior exchange into second model request. Relevant real Russian follow-up quality remains unassessed. |
| Bounded token/audio queues | Token backpressure/release, duration credit and queued reservation regressions pass. |
| Original browser scenarios through Nginx | Page rendering observed, but updated server not activated; repeated microphone turns, both interruptions, headphone self-trigger behavior and recovery remain unverified. |
| Relevant Russian Dota answers / audible overlapping b_lez | Reported model round trip is nonempty; response relevance/concision, actual audible selected voice and playback overlap remain unverified. |
| Ten-turn latency / agreed targets | Required end-to-end response/interruption measurements, p50/p95 and target agreement remain incomplete. |
| Runtime storage/log privacy | Application logging regression omits audio marker. Full post-session runtime inspection remains pending. Requested transcript/reply logs keep unchanged no-content-logs criterion explicitly unmet. |
| Honest original-criterion report | Report preserves partial/unmet statuses, no-microphone/no-restart disclosure and limits of real-model measurements. It does not claim full acceptance. |

### Findings

None blocking the reviewed code correction pass.

### Remaining original acceptance evidence

Activate the reviewed server code and verify readiness and the updated WebSocket
conversation through Nginx. Then observe continuous capture and track release,
VAD/pre-roll/noise behavior, microphone transcription, relevant Russian Dota and
contextual answers, audible incremental b_lez output, both barge-in stages,
headphone/no-self-interruption behavior, stop/restart/reconnect, and stale-speech
isolation. Collect the ten-turn latency baseline and agree targets. Inspect actual
runtime storage and logs after a session. Keep the intentional text-log conflict
unmet under the unchanged specification unless the user authorizes a requirement
change; do not mark full SPEC-001 acceptance complete from this code review.

Only this cycle-3 review entry was appended by the Reviewer. Existing entries,
acceptance checklists, source/tests/report and the parent spec were preserved.
No commit was created by the Reviewer.
