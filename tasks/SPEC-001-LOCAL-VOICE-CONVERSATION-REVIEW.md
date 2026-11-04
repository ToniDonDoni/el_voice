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
