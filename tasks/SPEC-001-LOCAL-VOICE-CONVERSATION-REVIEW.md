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

## RED_REVIEW cycle 4 - 2026-11-04: configurable interruption

**Verdict: FAIL / REQUEST_CHANGES.** The RED failure is valid for the missing
checkbox, but the proving tests do not yet cover the new user task: keeping an
active answer running when interruption is disabled. GREEN must wait for the
missing behavioral RED coverage below.

Reviewed commit `6c08330` on `codex/voice-interruption-toggle`, the user's request,
the new SPEC-001 User-controlled barge-in criterion, the corresponding SPEC-002
Configurable interruption criterion, project guidance, the spec-driven-tdd skill,
and the acceptance-boundary guide. Production source is unchanged by this RED
commit. The Reviewer changed only this review artifact and created no commit.

### Verification

`node --test workspace/tests/browser_playback.test.cjs` exits 1 with the expected
failure: `the interrupt checkbox must be visible and checked by default`.
Existing cases in that file run without another reported failure. This is a
feature-specific RED rather than a dependency, syntax, or fixture error.

### Acceptance-list summary

| New acceptance behavior | Reviewed test/action and assertion | RED verdict |
| --- | --- | --- |
| Visible checked-by-default checkbox and explanation | HTML regex checks checkbox type, checked attribute, and exact label. No rendered visibility or help-text assertion; fixture hardcodes `checked = true`. | PARTIAL; rendered evidence missing |
| Default preference sent at session start | FakeWebSocket start-message assertion requires `interrupt_enabled === true`. | Covered at browser-script harness boundary |
| User can select off before starting | No test starts a session after selecting off. | MISSING |
| Live checkbox updates active session | Harness directly calls change listener and asserts false/true messages. No server receives either message. | PARTIAL |
| Disabled speech preserves generation/search and playback, completes the reply, and starts no competing turn | No new test sends control/audio through the voice WebSocket or observes response completion/cancellation. | MISSING |
| Enabled speech cancels active work and accepts replacement speech | Existing browser test consumes a prebuilt interrupt event; no test proves VAD/control-to-server cancellation. | MISSING |
| A later utterance is accepted after the protected reply finishes | No new server/app-boundary test. | MISSING |
| Reconnect retains the selected off/on value | No new connection is opened after selecting off. | MISSING |
| Same preference works in Local and Remote Voice | No mode-parameterized WebSocket behavior test. | MISSING |

### Findings

1. **P1 - Add real voice-WebSocket proving tests for the business behavior.**
   `workspace/tests/browser_playback.test.cjs:261-283` can become GREEN after only
   adding a checkbox and sending a JSON flag, while `receive_control` still
   ignores the flag and `receive_audio` still unconditionally cancels the old
   turn. Add deterministic tests through `/ws/voice` using production session,
   control, VAD dispatch, and turn processing; fake only inference/VAD leaves.
   Start an answer, disable interruption, send detected speech, release the
   answer producer, and assert the original turn emits its remaining text/audio
   and completion, its cancellation event stays unset, no interrupt is sent, and
   no competing transcription/generation starts. Also cover speech during queued
   playback, normal next-utterance admission after playback finishes, and
   re-enabled speech causing a real interrupt/replacement turn. Parameterize the
   shared behavior for Local and Remote mode. These tests must fail for the
   current unconditional cancellation behavior rather than merely asserting a
   new field exists.

2. **P1 - Prove the actual visible control at a rendered/browser boundary.**
   The HTML regex at `workspace/tests/browser_playback.test.cjs:262-266` accepts a
   hidden checkbox, and the fake document at line 79 supplies the default value
   without parsing the production DOM. It cannot prove that the user sees the
   checkbox, its explanation, or can operate it. Add a rendered browser/app test
   that opens the production HTML/script, asserts the labeled checkbox is visible
   and checked, observes the nearby disabled-mode explanation, clicks the actual
   control, and observes the corresponding session message or behavioral effect.
   A synthetic socket/audio setup is sufficient; real microphone capture is not
   required for this UI evidence.

3. **P2 - Cover off-before-start and reconnect persistence.**
   The new harness test checks only a default-on initial handshake and live
   true/false messages. Select off before a session starts and assert its start
   message is false; then reconnect after selecting off and assert the replacement
   connection again sends false. Repeat enabling to establish that current user
   choice, rather than a constant/default reset, controls each new connection.

The RED commit contains no production implementation and has a trustworthy
missing-feature failure. Acceptance coverage is incomplete, so this review does
not authorize GREEN yet. No Docker build/restart, live provider/search request,
or microphone capture was performed.

## RED_REVIEW cycle 5 - 2026-11-04: configurable interruption

**Verdict: FAIL / REQUEST_CHANGES.** Commit `bde3dc4` adds a useful real voice
WebSocket regression and closes the missing pre-session/reconnect assertions at
the browser-script harness boundary. The remaining gaps are protection of a
still-generating answer, a live transition from enabled to disabled, and rendered
control evidence. The required behavior has not yet received complete RED
coverage, so GREEN remains blocked.

### Verification

- `node --test workspace/tests/browser_playback.test.cjs`: expected exit 1 for
  the absent checked-by-default checkbox.
- `docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_review_pipeline.py' -k interrupt_toggle`:
  expected exit 1; both Local and Remote subtests fail because detected speech
  produces `interrupt` instead of `speech_ignored`. The direct Docker attempt was
  sandbox-blocked; the same narrowly scoped test ran after approved escalation.
  No image, service, provider, search, or microphone state was changed.
- Inspected the full new test and production session/control/audio path. No
  production implementation is present in either RED commit.

### Acceptance-list summary

| New acceptance behavior | Evidence in the revised RED | Verdict |
| --- | --- | --- |
| Visible checked default, label, explanation, actual user operation | HTML regex plus fake Element/listener invocation only. Explanation string now asserted. | PARTIAL; rendered boundary still missing |
| Initial default-on handshake | FakeWebSocket start message asserts true. | COVERED at script harness boundary |
| Off selected before starting | Second connection asserts false. | COVERED at script harness boundary |
| Live disabling during an active reply | Node emits false, but real WebSocket test starts false and only applies true live. | MISSING at server boundary |
| Disabled speech preserves an unfinished generation/search and remaining reply output | Noise arrives only after `turn_complete`; the generator has already finished. | MISSING |
| Disabled speech does not cancel active playback | Real WebSocket test reaches playback-started, expects speech_ignored and uncancelled original turn. | Useful target-specific RED; playback continuation/completion not observed |
| Protected answer reaches playback completion | Test itself sends playback_finished rather than observing browser audio sources complete. | NOT PROVED |
| Later utterance is admitted | After the synthetic completion acknowledgement, test expects a fresh turn and transcription count 2. | COVERED at app/WebSocket boundary |
| Re-enabled speech interrupts/replaces | Real WebSocket setting=true round trip then VAD frames require interrupt, cancelled old turn, and fresh replacement completion. | COVERED at app/WebSocket boundary |
| Selected value survives reconnect | Script harness closes socket and asserts replacement handshake remains false. | COVERED at script harness boundary |
| Shared Local and Remote behavior | WebSocket test repeats against both fake runtime modes. | COVERED for shared voice policy; no live provider claim |

### Findings

1. **P1 - Exercise disabled interruption before generation completes.**
   `test_interrupt_toggle_preserves_active_reply_when_off_and_interrupts_when_on`
   waits for `turn_complete` before it sends the protected speech. Its fake model
   has already emitted its entire answer, so the test cannot catch cancellation
   of in-flight generation/search or discarded remaining answer chunks. Hold a
   deterministic model/remote stream open after producing initial text/audio,
   send the detected speech while it is still active, then release the stream.
   Assert the original turn remains uncancelled, remaining text/audio arrives,
   its completion arrives, and no competing turn starts. Keep the existing
   playback-only case. A browser audio instrumentation test should establish
   that scheduled sources finish without being stopped: sending playback_finished
   from the test only proves that the server accepts that claim.

2. **P1 - Add the requested rendered control check.**
   Cycle-4 finding 2 remains unresolved. The revised Node test still relies on an
   HTML regex and a fake document whose `checked` value is supplied by the test;
   it would accept a hidden input and never clicks the rendered checkbox. Open
   production HTML/script in an isolated browser, assert the checkbox and helper
   text are visible and default checked, click the actual checkbox, and observe
   the setting reaching a synthetic socket or app boundary. No real microphone
   is needed.

3. **P2 - Test switching interruption off in the active server session.**
   The real WebSocket test starts with false and tests only the live true update.
   A server that accepts the initial false but ignores later false changes could
   pass. Start enabled, issue false over the active WebSocket before the protected
   speech, observe its acknowledgement, and prove the same uncancelled completion
   behavior. Retain the separate initial-false handshake coverage.

Only this cycle-5 section was appended. Existing review history, specs, source,
and tests were preserved; no commit was made by the Reviewer.

## RED_REVIEW cycle 6 - 2026-11-04: configurable interruption

**Verdict: FAIL / REQUEST_CHANGES.** Commit `2c19458` substantially improves the
WebSocket proof: the first model stream remains blocked after initial audio, and
the test requires additional synthesis/audio and completion after the protected
speech. Two concrete issues remain: its start fixture is still explicitly false
while asserting the default is true, and the rendered browser/control/output
proof requested in cycles 4 and 5 is still absent.

### Verification

The Reviewer independently reran
`docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_review_pipeline.py' -k interrupt_toggle`.
Both Local and Remote subtests fail for the intended current production defect:
`interrupt != speech_ignored`, while the first answer generator is still blocked
after delivering its initial PCM packet. Test cleanup releases the blocked fake
generator. The existing container was used without a build or service restart.
The unchanged Node test's earlier missing-checkbox RED remains relevant; it was
not rerun needlessly in this cycle.

### Acceptance-list summary

| Behavior | Revised test evidence | Verdict |
| --- | --- | --- |
| In-flight reply survives disabled speech | Initial PCM arrives while fake LLM waits; original turn must stay uncancelled; released stream must yield more synthesis/PCM and completion without another transcription. | COVERED at shared app/WebSocket boundary; target RED reproduced |
| Live enabled-to-disabled transition | Test sends false live, but its start message still also contains false. | NOT CORRECTLY PROVED |
| Server default is enabled | Later assertion expects true although explicit false was supplied at start. | INVALID fixture/assertion pairing |
| Enabled cancellation/replacement and later idle admission | Existing real WebSocket continuation checks preserved for both fake modes. | COVERED at declared app boundary |
| Checkbox visible/default checked/helper/click | Existing HTML regex and fake Element listener remain the only evidence. | Rendered proof MISSING |
| Selected off value before start and after reconnect | Existing script-harness assertions preserved. | COVERED at script boundary |
| Browser reply actually finishes without stopped sources | Test client sends playback_finished after transport completion; no browser observes source completion. | Rendered audio-path proof MISSING |

### Findings

1. **P1 - Correct the initial setting in the live-transition test.**
   `workspace/tests/test_review_pipeline.py:1015` still sends
   `"interrupt_enabled": False` in the start payload, whereas the test later
   asserts `started.interrupt_enabled is True` and describes a default-on
   enabled-to-disabled transition. A correct implementation honoring the supplied
   false will fail that assertion; the live false control is currently false-to-
   false. Omit the initial preference to test the default (or explicitly send
   true and assert that configured initial state), then send false live. Retain
   the independent initial-false coverage.

2. **P1 - Add the rendered checkbox and uninterrupted audio-path proof.**
   Cycles 4/5's rendered-boundary finding remains unchanged. The Node regex can
   accept a hidden control, its fake document supplies checked state, and no user
   clicks a real DOM checkbox. Add an isolated browser test using production
   HTML/script that observes the visible checked control and helper, clicks it,
   and observes the setting message. With browser-level synthetic audio/socket
   instrumentation, also verify protected reply audio sources are not stopped
   and reach their completion path. The WebSocket test proves transport and
   cancellation policy; its client-generated playback_finished is not evidence
   that the browser played the answer through. No physical microphone or speaker
   observation is required for this deterministic browser proof.

Only this cycle-6 section was appended. No production source, tests, specification,
or earlier review entry was changed, and the Reviewer created no commit.

## RED_REVIEW cycle 8 - 2026-11-04: configurable interruption

**Verdict: FAIL / REQUEST_CHANGES.** Reviewed through `a28edc8`. The in-flight
generation, live toggle, queued audio preservation, and manual rendered RED gaps
are now resolved. One acceptance branch remains unproved: the service honoring an
unchecked preference supplied in the initial WebSocket start message.

### Verification and resolved findings

- The start payload in the live-transition scenario now omits the preference,
  correctly testing default-on followed by a live false update.
- The Reviewer independently ran the targeted real-WebSocket test in the existing
  container. Local and Remote both fail for the intended current behavior,
  `interrupt != speech_ignored`, while generation remains blocked after initial
  PCM output. No unrelated setup failure occurs.
- `node --test workspace/tests/browser_playback.test.cjs` still fails specifically
  for the absent checkbox. Its new queued-audio case schedules both sources before
  ignored speech, requires neither to stop, and requires natural completion of
  both before the browser emits playback_finished.
- An in-memory mutation making speech_ignored call stopPlayback is rejected by
  the queued-audio test with `ignored speech must not stop current assistant
  audio`. The probe changed no repository file.
- The Reviewer independently opened the production static page at
  `http://127.0.0.1:18765/static/index.html` in a browser. Start, Mute, volume,
  and conversation controls rendered; the exact interrupt checkbox role count
  was 0. This is valid expected rendered RED. No microphone permission or capture
  was initiated, and the review tab was closed. Positive rendered visibility,
  default, actual clicking, and output checks remain mandatory in GREEN.

### Acceptance-list summary

| Behavior | Evidence | RED verdict |
| --- | --- | --- |
| Visible/default control and explanation | Target-specific Node failure plus independently observed absent checkbox in the rendered production page. | COVERED as RED; positive UI verification required at GREEN |
| Default-on and live enabled-to-disabled control | Real WebSocket default handshake, live false acknowledgement, protected onset and unfinished generation. | COVERED |
| Disabled response continues through tail audio and completion | Real WebSocket blocked/released model stream plus production browser handler/audio-source completion harness. Mutation probe rejects an erroneous stop. | COVERED at stated deterministic boundaries |
| Re-enabled cancellation, replacement, and later idle utterance | Real WebSocket continuation assertions for both fake runtime modes. | COVERED |
| Client sends selected false before session/reconnect | Node script harness asserts initial-false and reconnect-false messages. | COVERED for the client only |
| Service honors initial selected false | Real WebSocket test only starts with the preference omitted, then uses the separate live setting message. | MISSING |

### Finding

1. **P1 - Prove the initial unchecked preference at the service boundary.**
   The Node case verifies outbound `start.interrupt_enabled = false`, but no real
   voice-WebSocket case sends that field and proves the service uses it. A server
   that ignores the start field, always initializes true, and correctly supports
   subsequent interrupt_setting messages would pass the current behavioral tests.
   A user who unchecked the control before starting (or reconnecting) would still
   be interrupted. Add a real `/ws/voice` start with false, assert its session
   acknowledgement is false, and send speech during an active response to assert
   that the turn remains uncancelled and completes without an interrupt. Reuse the
   existing deterministic scenario where practical; keep the default-on/live-
   transition case as well.

Only this cycle-8 section was appended. Source, tests, specifications, and earlier
review entries were preserved. No commit, image build/restart, live provider/search
request, or microphone capture was performed by the Reviewer.

## RED_REVIEW cycle 9 - 2026-11-04: configurable interruption

**Verdict: PASS / APPROVED for RED.** Reviewed through `abb7721`. All concrete
RED findings from cycles 4-8 are addressed. The tests now demonstrate the missing
feature at the client, voice WebSocket, and browser playback boundaries, with
independent rendered absence evidence. The Implementer may proceed to GREEN.
This is not approval of an implementation or completion of live voice acceptance.

### Verification

The Reviewer independently ran
`docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_review_pipeline.py' -k initially_disabled`.
Both Local and Remote subtests fail specifically with `interrupt !=
speech_ignored`: the initially unchecked preference is ignored by current
production code during active playback. The new case also asserts the start
acknowledgement is false, original turn is current and uncancelled, and no second
transcription starts. It uses the actual Uvicorn `/ws/voice` endpoint and leaf
VAD/model fakes, without embedding the feature in its fixture.

Earlier independently reproduced evidence remains applicable: the default-on/live
toggle case fails during an unfinished generation; the Node control case fails
for the absent checkbox; a buggy in-memory ignored-speech handler that stops
playback is rejected by the queued-source case; the actual rendered production
static page has no interrupt checkbox. Production files remain unchanged across
the RED commits. `git diff --check` passes.

### Acceptance-list summary

| New acceptance behavior | Proving evidence | RED verdict |
| --- | --- | --- |
| Visible checkbox, default checked, label and explanation | Node markup/script assertions plus independent actual-browser absence observation. | PASS as missing-feature RED; positive rendered check required in GREEN |
| Default enabled at service start | Real WebSocket case omits the setting and asserts its start acknowledgement is enabled, alongside the later live false policy checks. | PASS |
| Off choice before start or reconnect | Client tests assert false start/reconnect messages; new real WebSocket initial-false case asserts service acknowledgement and protected playback. | PASS |
| Live disable while answer is generating/playing | Real WebSocket control, blocked stream, initial PCM, ignored onset, uncancelled original, remaining PCM/synthesis and completion. | PASS |
| Disabled speech starts no competing turn; later speech is accepted | WebSocket tests assert transcription counts and later fresh-turn completion after playback acknowledgement. | PASS |
| Enabled voice interrupts and replaces active turn | Live true acknowledgement followed by VAD onset requires actual interrupt, old cancellation, and replacement completion. | PASS |
| Protected browser playback reaches completion | Production handler harness has two active sources at ignored speech, requires no stop, no early completion, and playback_finished only after both natural end callbacks. | PASS at browser API harness boundary |
| Shared Local and Remote policy | Both real WebSocket scenarios repeat with Local and Remote fake runtime modes. | PASS for the shared voice control; no external provider claim |

### Findings

None blocking RED. The reviewed tests are relevant, have valid setup and
feature-specific failures, and do not implement the requested production policy.

### GREEN requirements

Run the reviewed proving tests and relevant regressions after implementation.
Independently verify the rendered checkbox is visible, checked on a fresh page,
has its explanation, and changes through an actual click. Confirm the implemented
client sends the selected setting and reconnect retains it. Keep synthetic audio
and shared WebSocket evidence distinct from physical microphone/headphone and
live provider acceptance. The latter is not required to prove this narrow toggle,
but must not be claimed from these tests.

Only this cycle-9 section was appended. No source, test, spec, or earlier review
entry was changed, no commit was created, and no build/restart, real provider/search
request, or microphone capture was performed by the Reviewer.

## GREEN_REVIEW cycle 10 - 2026-11-04: configurable interruption

**Verdict: FAIL / REQUEST_CHANGES.** The normal interruption-toggle journeys pass
the reviewed regressions, but disabling interruption introduces a reproducible
error-recovery lockout: a failed turn remains current, so every subsequent speech
onset is ignored. The implementation must release failed turns before acceptance.

Reviewed the current working diff based on `abb7721`, the original user request,
the two canonical task specs, project guidance, reviewed RED sequence, code, tests,
and implementation reports. The setting belongs to the shared voice session; no
Local/Remote provider architecture or dependencies were changed by the feature.

### Verification performed

- `node --test workspace/tests/audio_capture.test.cjs workspace/tests/browser_playback.test.cjs`:
  both suites PASS, including the visible-control script assertions, initial and
  live setting messages, reconnect selection, protected audio sources, and natural
  playback completion.
- `docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_review_pipeline.py'`:
  20 test methods PASS in 5.615 seconds. These include the two new Local/Remote
  voice-WebSocket scenarios; they are 20 mixed regression methods, not 20 separate
  WebSocket tests.
- `git diff --check`: PASS.
- Independent in-memory production-path failure probe in the existing container:
  create VoiceSession with interruption false, send start/end VAD frames, let
  transcribe raise a synthetic error, await process_turn, then send a new onset.
  Observed first events `speech_started`, `speech_ended`, `error`; the next event
  is `speech_ignored` for the failed turn, and `session.current_turn` still points
  to that failed turn. No real model/provider was used.
- The Implementer reports an actual rendered-browser checked-default/off/on check.
  The Reviewer's attempt to reopen the temporary static URL was connection-refused
  because that server was no longer running; positive rendering was not repeated
  independently in this cycle. The earlier rendered RED observation remains
  recorded, and the source/Node wiring was inspected.

### Acceptance evidence

| Behavior | Evidence | Current result |
| --- | --- | --- |
| Visible checked-default control and explanation | Production checkbox/label/helper inspected; author reports real browser interaction; script tests pass. | PASS at inspected/source/author-rendered boundary |
| Initial default/off choice and reconnect | Boolean start validation/ack; real WebSocket startup-false/default cases and browser reconnect tests pass. | PASS |
| Live off while answer is generating/playing | Shared session policy, blocked fake stream, unchanged turn, remaining PCM and completion tests pass in both modes. | PASS for normal deterministic paths |
| Re-enable and automatic replacement | Real WebSocket true update, old cancellation, and replacement completion pass. | PASS |
| No stopped/early-finished protected audio | Production handler harness keeps current/queued sources and reports completion only after both end. | PASS at browser API harness boundary |
| Error recovery with interruption off | Independent probe shows all subsequent speech ignored after a terminal error. | FAIL |
| Honest remaining acceptance | Reports preserve full connected-mic/provider limits. One report calls all 20 mixed regression methods WebSocket tests. | Limits retained; correct the test-count description |

### Findings

1. **P1 - Release a terminally failed turn when interruption is disabled.**
   `process_turn` catches STT/LLM/TTS failures and queues an error, but leaves
   `session.current_turn` set. The browser error handler does not send a completion
   acknowledgement. The new `receive_audio` guard therefore treats that failed
   turn as an active reply and sends speech_ignored for every later onset when
   the checkbox is off. A transient recognition/provider/TTS failure permanently
   silences the session until the user re-enables interruption or restarts it.
   End failed-turn ownership and clean up its pending work/audio safely so normal
   listening resumes. Add a real voice-WebSocket regression, with interruption
   off, that makes the first inference fail, then proves a subsequent utterance
   is accepted and can complete without a checkbox/session workaround; cover both
   modes and account for failure after partial audio where applicable.

2. **P2 - Keep speaking status consistent with the unchecked control.**
   `playPCM` still unconditionally sets `Speaking · you can interrupt`, even when
   interruption is disabled. This contradicts the selected control and its helper
   until a noise event overwrites the status. Render a status matching the active
   choice and verify it in the browser script/output flow.

Only this cycle-10 section was appended. Production source, tests, specs, reports,
and earlier review history were preserved. No commit, build/restart, real external
request, or microphone capture was performed by the Reviewer.

## GREEN_REVIEW cycle 11 - 2026-11-04: configurable interruption

**Verdict: PASS / APPROVED for the interruption-control change.** Both Cycle 10
findings are resolved. The original narrow user task is implemented: the browser
offers a default-on checkbox, off protects the active reply from detected speech,
and on restores voice interruption in the shared Local/Remote voice path. There
are no remaining high-confidence blocking findings in this feature diff.

Full connected microphone, physical audio, live provider/search, and the broader
original task acceptance remain unverified as reported; this approval does not
convert those outstanding criteria to PASS.

### Independent verification

- `docker exec -w /workspace el-voice-voice-1 python -m unittest discover -s tests -p 'test_review_pipeline.py'`:
  **21 test methods PASS**, 6.262 seconds. This includes startup-false, default-on,
  live off/on, protected in-flight generation/playback, replacement, next-turn
  admission, and failed-partial-answer recovery in Local and Remote fake modes.
- `node --test workspace/tests/audio_capture.test.cjs workspace/tests/browser_playback.test.cjs`:
  **both suites PASS**, including setting-dependent speaking status, preserved
  current/queued sources, delayed failed-turn acknowledgement, and reconnect.
- Original no-audio STT failure rechecked with in-memory production-path probes:
  server now emits turn_failed; the production browser handler acknowledges a
  failed turn with no sources; playback_finished releases server ownership and a
  subsequent onset becomes speech_started. Both probes PASS without file changes.
- Independently served unchanged production static files at a loopback-only URL,
  opened the actual page in Chrome, and verified the checkbox and explanation
  visible, initial checked=true, click-to-false, and click-back-to-true. Screenshot
  and accessibility state confirmed the layout. No microphone or session start
  was requested. The temporary tab and static server were closed afterward.
- `git diff --check`: PASS. Acceptance wording was not weakened to match GREEN;
  task/report updates describe the new feature and retain live-acceptance limits.

### Finding closure and acceptance matrix

| Requirement or prior finding | Exact evidence | Result |
| --- | --- | --- |
| Cycle 10 error lockout | process_turn emits turn_failed after errors; frontend waits for queued sources or immediately acknowledges when none exist. Real WebSocket regression completes a later utterance with interruption still off in both modes; no-audio probes also pass. | FIXED |
| Cycle 10 misleading status | playPCM and live checkbox change use the selected setting; Node output-flow assertions require off/on wording. | FIXED |
| Visible default-on control and explanation | Independent real rendered page, actual checkbox clicks, source label/helper and passing script assertions. | PASS |
| Initial preference and reconnect | Actual service start/default tests plus browser false-before-start/reconnect message tests. | PASS at declared deterministic boundaries |
| Live off preserves reply without competing work | Reviewed real WebSocket blocked-generation/PCM continuation test remains GREEN; queued-source browser case reaches natural completion. | PASS |
| Re-enable interrupts and accepts replacement | Shared WebSocket cancellation/replacement tests remain GREEN for Local/Remote fake modes. | PASS |
| Later utterance and failure recovery | Normal completion and partial-failure WebSocket scenarios accept the next fresh turn while off. | PASS |
| Reproducible and honest evidence | Reports now correctly describe 21 mixed pipeline test methods, browser checks, and outstanding connected-mic/provider scenarios. | PASS for this feature report |

### Findings

None blocking this feature. The implementation follows the existing shared
session/protocol and browser architecture without adding packages or provider
changes. Reviewed RED now passes for the intended reason, and the accepted outcome
addresses the noisy-environment use case at the tested app/browser boundaries.

Only this cycle-11 section was appended. No source, tests, specs, reports, or prior
review entries were edited by the Reviewer, and no commit or Docker build/restart
was performed. All inference/search inputs used deterministic fakes; no physical
microphone or live external provider request was used.
