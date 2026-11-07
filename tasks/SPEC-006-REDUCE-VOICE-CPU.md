# SPEC-006: Reduce EL Voice CPU usage during listening and speech processing

## 1. Problem and current state

EL Voice uses the Mac CPU through the Linux ARM64 voice container. During a recent voice turn, the container reportedly consumed about 394% CPU, and a two-second thread sample accumulated about 470% CPU. These are observations during activity, **not** proof of high steady-state CPU in silence. On an Apple Silicon laptop, sustained multithreaded inference causes heat and battery drain.

Current code paths:
- The browser sends 16 kHz PCM frames throughout an active voice session, even when nobody is speaking.
- Silero ONNX VAD evaluates 512-sample frames (32 ms) with one intra-op thread. This is expected background listening work.
- If `Only accept enrolled speakers` is enabled, the Nomo-PVAD speaker scorer is invoked on every 160 ms audio batch, including silence. Each invocation reprocesses a rolling window of up to 2.4 seconds.
- Faster-Whisper uses `beam_size=3` and 4 CTranslate2 CPU threads during transcription.
- Silero TTS is configured for 4 PyTorch threads; speaker model initialization subsequently calls the process-wide `torch.set_num_threads(2)`. This makes actual PyTorch threading dependent on initialization order.
- No verified per-stage CPU profile or steady-state silence measurement is available yet. Do not label any one stage the proven sole culprit.

## 2. Required state

Keep hands-free listening, interruption, enrollment, speaker rejection, transcription, synthesis, and both local and remote modes functional. Minimize expensive CPU inference when the user is silent and reduce the CPU burst during speech/response processing without introducing uncontrolled buffering, false speaker acceptance, or lost utterance starts. Preserve the existing model assets, WebSocket audio contract, and local-only handling of raw audio.

## 3. Acceptance criteria

- [ ] A 30-second silent listening session with speaker filtering enabled does not continuously invoke Nomo-PVAD. Silero VAD remains active to detect speech. Record speaker-score invocation counts as evidence.
- [ ] On the same Mac and settings, the average voice-container CPU during 30 seconds of silence with the filter enabled is at most half of the baseline measured on `main`. Report actual percentages and variance, not assumed results.
- [ ] During 10 short Russian utterances, the enrolled speaker is accepted, a non-enrolled speaker is rejected, and neither first syllables nor speech-end transitions are lost. Include quiet speech and pause/resume cases.
- [ ] With the filter disabled, the ordinary Silero VAD path works unchanged and does not invoke speaker scoring.
- [ ] After stopping the session, persistent voice-container CPU returns near its no-session baseline; there are no accumulating background workers, stale batches, or lingering transcription jobs.
- [ ] Keep Whisper `beam_size=3` in the new branch and verify transcripts are at least as good as the baseline. Compare p50/p95 speech-end-to-first-audio latency, transcription quality, voice output, container CPU peaks, and memory before and after the changes. Any accuracy or latency regression must be documented and reviewed before merge.
- [ ] Automated tests cover silent suppression, speech onset, endpoint/reentry, speaker gating, and configured inference concurrency.

## 4. Analysis

A continuous microphone stream is expected for hands-free voice activity detection; continuous heavy speaker-model inference is not necessary for silence. The cheap Silero detector can act as a front gate for optional speaker recognition. Filtering **must fail closed**: VAD may decide whether to *run* speaker recognition, but may never itself authorize a speaker. Maintain a short pre-roll so speech beginning before the VAD decision can be used by the speaker recognizer.

The rolling Nomo window recomputes up to 2.4 seconds for each 160 ms chunk while speech is active. Reducing inference frequency or replacing it with true incremental scoring would require model validation and should not be assumed safe without quality measurements.

Whisper beam search affects recognition accuracy and should be preserved at `beam_size=3` until a separate, quality-controlled experiment warrants a change. CPU-thread settings affect peak CPU and latency, so benchmark them independently without reducing beam search. PyTorch CPU thread counts are process-global: they cannot safely be treated as independent TTS and speaker pools.

## 5. Proposed solution

1. Add a per-session Silero VAD gate **before** optional Nomo-PVAD scoring. Use a separate detector instance from the turn-ending VAD, preserve existing audio frames/pre-roll for recognition, and retain fail-closed behavior. Do not run the Nomo network on confirmed silence; reset per-speaker rolling state across inactive periods so stale audio does not contaminate a new utterance.
2. Bound inference parallelism: start with 2 CTranslate2 threads while preserving `beam_size=3` for Whisper; configure PyTorch inference consistently at 2 threads for the current TTS/speaker combination. Benchmark latency and quality before accepting the thread-count changes. Do not reduce beam size as part of this optimization.
3. Do not add an unconditional spin/sleep loop. Keep queue waits event-driven and continue to drain or discard stale audio under overload.
4. Add targeted tests plus repeatable **before/after** measurement instructions. Do not claim measured improvements until the target Mac runs the measurements.
5. Preserve current defaults for user privacy and session controls; no external model or dependency updates.

## 6. ToDo and verification

- [ ] Implement the voice activity gate for speaker recognition, preserving speaker rejection and interruption behavior.
- [ ] Update inference settings conservatively and document trade-offs.
- [ ] Write automated unit/regression tests for the gate.
- [ ] On the target Mac, run baseline `main` and this PR separately using the same mode, model assets, speaker profile, microphone, and Docker settings.
- [ ] For each build, capture `docker stats --no-stream el-voice-voice-1` repeatedly over **30 seconds without a session**, **30 seconds of session silence with speaker filter off**, **30 seconds of session silence with filter on**, and **10 comparable spoken turns**. Compare average/peak CPU and memory.
- [ ] Check logs for `speech_started`, `speech_ended`, `user_transcript`, `turn_complete`, `speaker_scoring_failed`, queue drops, and session cleanup. Keep private text out of shared benchmark artifacts.
- [ ] Measure p50/p95 response and interruption latency via `/api/status`, inspect transcript quality and voice output, and test enrolled/non-enrolled voices plus whispered/quiet speech.
- [ ] Have Luna review real device results and explicitly approve or reject the CPU/quality trade-offs before merge.

No benchmark results are asserted by this specification.
