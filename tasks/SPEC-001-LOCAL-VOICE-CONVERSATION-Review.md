# Review notes: SPEC-001 local Dota voice conversation

Status: **not started**. This file records follow-up work found while reviewing the
implemented pipeline on branch `DeepSeek-Flash` (commits `b96ef16`, `18dbb23`, `1be543f`,
`c3400ac`). Nothing here has been implemented or verified yet.

Scope note: R1 extends the behaviour described in
[`SPEC-001-LOCAL-VOICE-CONVERSATION.md`](SPEC-001-LOCAL-VOICE-CONVERSATION.md). The current
implementation is stateless per turn on purpose ("pass each finalized utterance once").
When R1 is done, the SPEC problem/design text, acceptance criteria and README must state
that all turns of a session share one conversation context.

## R1. Single conversation context for every message (main task)

**Current behaviour.** `workspace/elvoice/pipeline.py::_generate` sends only
`[system prompt, current utterance]` to Gemma. The transcript shown in the browser is
display-only; earlier questions and answers are not passed back to the model.

**Problem.** Follow-up questions lose their referent ("how good is he on mid?" after a
previous hero question). Every message is processed in isolation.

**Work.**
- Keep a bounded, in-memory conversation history per voice session: user final
  transcripts and assistant final answers, in order, with no audio and no persistence.
- Build the LLM prompt as `system + trimmed history + current utterance`, dropping the
  oldest complete pairs when a token/character budget is exceeded; never include
  cancelled, empty or barge-in-truncated turns as assistant context.
- Add configuration and validation in `workspace/settings.json` and
  `workspace/elvoice/settings.py`, e.g. `conversation.history_turns` and
  `conversation.max_prompt_chars`; expose the limits (not the content) in `/api/status`.
- Clear history on session stop and disconnect; keep it out of logs, `/data` and metrics.
- Update README (architecture, privacy, latency) and SPEC-001 acceptance wording.
- Extend `workspace/tests/selftest.py` with a follow-up turn whose answer depends on the
  previous one; add a unit check for history trimming and a prompt-length bound.

**Acceptance.** Three or more consecutive turns in the browser where the last answer uses
earlier context; history stays within the configured bound; no transcript content in logs
or storage; added first-audio latency measured and reported (targets agreed from the
target-Mac baseline, not invented here).

## R2. Correct audio resampling for 44.1 kHz capture

**Current behaviour.** `workspace/static/capture-worklet.js` downsamples by averaging
`inputSampleRate / 16000` samples: exact 3:1 at 48 kHz, but at 44.1 kHz the ratio is
2.75625 and the code alternates 2/3-sample groups without an anti-aliasing filter. That is
a crude decimation, not a resampler.

**Work.** Replace the averaging loop with a proper 44.1/48/32/96 kHz to 16 kHz resampler
(polyphase or linear interpolation with a low-pass step) in the worklet; keep the
512-sample frame contract. Add a check that a known-duration capture produces the expected
16 kHz sample count, and compare STT/VAD behaviour at 44.1 and 48 kHz.

**Acceptance.** Sixty seconds of 44.1 kHz capture yields 960,000 ± a few samples; STT
accuracy and VAD onsets do not regress versus 48 kHz; documented in README.

## R3. Turn identifiers must not repeat across sessions

**Current behaviour.** `SpeechSegmenter._turn_seq` starts at 1 in every session. Reconnect
creates a new session with repeated ids ("turn 1" again). Combined with R5 this can mix or
mis-flush already scheduled audio from a previous session.

**Work.** Decide and implement one of: include a session epoch in the audio routing,
reset the browser state on every `ready`, or make turn ids monotonic for the application
process. Make flush/playback checks robust against equal ids from different sessions.
Cover "audio in flight during reconnect" in the browser scenarios and in the self-test.

**Acceptance.** A reconnect never plays old audio over a new session and a `flush` cannot
stop the wrong turn; no mixed audio observed in a pulled-plug test.

## R4. VAD parity with the official Silero wrapper

**Current behaviour.** The raw Silero VAD v5.1.2 ONNX graph is fed 512-sample frames with a
persistent state tensor. The official `silero-vad` wrapper's framing, context handling and
reset semantics were not compared during implementation; a review note says the wrapper's
context is skipped.

**Work.** Inspect the v5.1.2 `OnnxWrapper`/`VADIterator` implementation, align input
framing, context and reset behaviour, and document any intentional deviation. Add a parity
test using the same WAV through both paths; re-tune onset/end thresholds only against real
microphone input.

**Acceptance.** Parity test within an agreed tolerance on probabilities and segment
boundaries, or a documented reason for the difference; README states the chosen semantics.

## R5. Stop scheduled playback when the connection is lost

**Current behaviour.** The browser schedules `AudioBufferSourceNode`s ahead of time. On an
unexpected WebSocket close, error or reconnect, `workspace/static/app.js` does not stop
already scheduled sources; audio can keep playing while the client reconnects, and the
server cannot flush it.

**Work.** On socket close/error and at the start of a reconnect: stop all sources, clear the
playhead and the current turn id, and mark the old turn ended. On the server, cancel the
active turn and send `flush` whenever the socket is still usable. Keep `stop` semantics
unchanged. Cover forced disconnect, network drop and reconnect in the browser scenarios.

**Acceptance.** After a forced disconnect no assistant audio continues for more than a
short, documented bound; reconnect starts from a clean playback state.

## Out of scope

Persistent conversation history across sessions, speaker-playback acoustic support, Dota
game control, external APIs, and anything already declared out of scope in SPEC-001.

## Verification for the follow-ups

All items require the real container and a browser on the Mac, plus the existing
content-free checks: `python -m compileall -q workspace`, `node --check` for the browser
scripts, and `workspace/tests/selftest.py` inside the voice container. Record commands,
measurements, and remaining limitations in README before marking any item complete.

## Implementation status

Implemented on branch `DeepSeek-Flash` with local evidence:

- R1: bounded in-memory context (`workspace/elvoice/context.py`), wired into
  `pipeline._generate`; prompt statistics are exposed through `/api/status` without content.
  Verified by `workspace/tests/test_contracts.py`, `tests/test_pipeline_context.py` and the
  end-to-end `workspace/tests/selftest.py` (3+ consecutive turns, `history_turns=3`).
- R2: streaming resampler `workspace/static/resampler.js` with a two-stage low-pass filter
  and fractional interpolation; used by the capture worklet. Verified by
  `tests/resampler.test.mjs` (44.1/48/32/96 kHz, 60 s count, 10 kHz alias rejection) and
  `tests/capture-worklet.test.mjs`.
- R3: process-wide turn id allocator (`workspace/elvoice/turnids.py`) used by every session;
  verified by `tests/test_pipeline_context.py` and the selftest reconnect checks.
- R4: VAD adapter now carries the official 64-sample context and 576-sample ONNX input;
  verified by `tests/test_vad_parity.py` against a reference of the v5.1.2 wrapper.
- R5: playback is isolated in `workspace/static/playback.js` with per-turn flush and
  `stopAll()` on socket close; verified by `tests/playback.test.mjs` and the selftest
  abrupt-disconnect check.

Additional evidence and remaining gaps:

- R4 parity now uses the official `silero-vad==5.1.2` package `OnnxWrapper` (not a
  hand-written reference); probabilities and reset are compared on the real ONNX model.
  Segment-boundary parity is documented as a deviation: the project uses its own
  endpointing segmenter, so only probabilities and streaming state are compared.
- R2 STT/VAD 44.1-vs-48 parity is not automated yet; the alias suppression is verified
  numerically (10 kHz residual RMS 0.092 at the 16 kHz output). A WAV-based STT/VAD
  comparison remains a browser/audio-acceptance item.
- R5 client-side evidence is module level (`static/playback.js` with a fake
  AudioContext) plus the server abrupt-disconnect check. The client calls `stopAll()`
  synchronously in the socket `onclose` handler; the expected audible bound is one
  event-loop turn (well under 100 ms) but must be confirmed in a real browser.
  Manual browser checks: force-close/reload the tab during playback, disable the
  network mid-answer, reconnect and confirm no old speech continues; measure the stop
  delay.

Browser/microphone acceptance (headphones echo behaviour, audible interruption, real
latency) still has to be run on the target Mac; Docker is unavailable in this environment.
