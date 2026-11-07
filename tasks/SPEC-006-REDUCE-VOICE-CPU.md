# SPEC-006: Reduce Voice CPU usage

## 1. Problem and current state
- 394-470% CPU observed during voice processing.
- Idle CPU usage has not been measured.
- Nomo-PVAD runs every 160 ms, including silence, when speaker filtering is enabled.

## 2. Required state
- Reduce CPU usage during silence and speech.
- Preserve hands-free listening and interruption.
- Preserve enrollment and speaker rejection.
- Preserve transcription and speech synthesis.
- Support both local and remote modes.
- Prevent unbounded buffering, false speaker acceptance, and lost speech onset.
- Keep model assets and WebSocket audio contract unchanged.
- Keep raw audio processing local.

## 3. Acceptance criteria
- [ ] No Nomo-PVAD inference during confirmed silence.
- [ ] Compare CPU, memory, and response latency with `main`.
- [ ] Test idle, silent listening, and active speech.
- [ ] Verify enrolled speaker acceptance and other-speaker rejection.
- [ ] Verify quiet speech, speech onset, and interruption.
- [ ] No sustained CPU work after session stop.
- [ ] Whisper retains `beam_size=3`.

## 4. Analysis
- Code confirms repeated Nomo-PVAD scoring during silence.
- Per-component CPU costs still require profiling.

## 5. Proposed changes
- Gate Nomo-PVAD with Silero VAD.
- Keep pre-roll and fail-closed speaker verification.
- Limit STT/TTS CPU threads to 2; keep `beam_size=3`.
- Add inference counters and regression tests.

## 6. ToDo
- [ ] Luna reviews changes.
- [ ] Luna compares `main` and PR on the target Mac.
- [ ] Validate acceptance criteria before merge.
