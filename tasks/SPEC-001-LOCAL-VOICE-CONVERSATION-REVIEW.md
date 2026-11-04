# SPEC-001 Implementation Review

Status: Changes requested.

Parent task: [Local Dota voice conversation](SPEC-001-LOCAL-VOICE-CONVERSATION.md).
Evidence report: [Implementation report](SPEC-001-LOCAL-VOICE-CONVERSATION-IMPLEMENTATION-REPORT.md).

## Scope

Continue the Luna implementation on `codex/local-voice`. Reuse selected ideas from
`DeepSeek-Flash` after adapting and checking them. Keep the current local models,
Docker deployment, conversation context, and working microphone capture.
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

## P1: Improvements to adapt from DeepSeek

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

## Import safeguards

The comparison found the following issues in DeepSeek. Do not introduce them when
adapting its components:

- At 44.1 kHz, its worklet produced 14,700 samples per input second instead of
  16,000. Its fallback capture also drops input after returning the first frame.
- Session turn counters restart at one while browser message maps retain old IDs.
- Its Silero v5.1.2 wrapper omits the preceding 64-sample input context used by the
  upstream wrapper. Keep the working bundled VAD unless a replacement is justified.
- WebSocket closure does not immediately flush scheduled playback.
- Each model request omits previous conversation turns; preserve Luna's history.
- The LLM token queue is unbounded; preserve bounded queues.

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
  criterion. Mark acceptance complete
  only after observation; an automated WebSocket test is not a microphone or
  physical audio-output test.

## Review evidence

Code from both working trees was compared; acceptance criteria, Dockerfile,
Compose, Nginx configuration, and dependency files were identical. Short isolated
checks reproduced the resampling discrepancy, repeated DeepSeek session IDs, and
Luna's blocked producer after stop. Full browser acceptance and a controlled
performance comparison were not rerun. Neither implementation is fully accepted.
