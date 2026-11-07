# SPEC-006: Reduce Voice CPU usage

## 1. Problem and current state
- Voice sessions showed 394-470% CPU during processing; idle CPU is not yet measured.
- With speaker filtering enabled, Nomo-PVAD processes audio every 160 ms, even in silence.

## 2. Required state
- Lower CPU usage without degrading transcription, speaker filtering, or interruptions.

## 3. Acceptance criteria
- [ ] Nomo-PVAD is not called during confirmed silence.
- [ ] Compare CPU, memory, and response latency against `main` in idle, silence, and speech.
- [ ] Enrolled speaker acceptance, other-speaker rejection, quiet speech, and barge-in still work.
- [ ] No sustained CPU work remains after session stop; Whisper keeps `beam_size=3`.

## 4. Analysis
- Excessive silence scoring is confirmed in code; exact CPU attribution still needs profiling.

## 5. Proposed changes
- Gate Nomo-PVAD with Silero VAD, retaining pre-roll and fail-closed speaker checks.
- Limit STT/TTS CPU threads to 2; preserve Whisper `beam_size=3`.
- Add scoring counters and regression tests.

## 6. ToDo
- [ ] Luna reviews changes and verifies acceptance criteria on the target Mac.
