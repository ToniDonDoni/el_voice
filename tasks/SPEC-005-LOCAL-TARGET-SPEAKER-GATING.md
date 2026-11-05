# SPEC-005: Local Target-Speaker Gating

## 1. Problem and Current State
- Browser AudioWorklet sends 16 kHz mono PCM in 512-sample frames through `/ws/voice`.
- `StreamingSileroVAD.accept()` in `workspace/voice_pipeline.py` detects any speech. `receive_audio()` in `workspace/app.py` uses it to open/end turns and trigger barge-in.
- Continuous TV speech can open unwanted turns and prevent the silence timer from ending the user's utterance.

## 2. Required State
- A hold-to-record **Detect me** control registers the owner's voice. Recording starts on press and stops on release; only a successful enrollment replaces the previous profile. This creates a voice embedding, not a trained model.
- The owner can explicitly enroll or remove permitted conversation partners. Only the owner and enabled, enrolled partners may open, extend, or interrupt a turn. TV and other voices must not reset the end-of-utterance timer.
- The UI shows enrollment progress, active protection, errors, and a way to delete saved profiles. Before enrollment, existing VAD behavior remains available and the UI clearly says speaker filtering is off.
- Models, speaker embeddings, and inference remain on the local Mac/Docker host in both Local and Remote Voice modes. No runtime downloads, API keys, cloud audio, or stored raw enrollment recordings. Filtering is not an authentication or source-separation guarantee.

## 3. Acceptance Criteria
- [ ] **Detect me** works with pointer, touch, and keyboard hold/release; enrollment never triggers STT/LLM. Insufficient clean speech (target: at least 10 seconds), cancellation, and disconnect retain the old profile and report failure.
- [ ] Successful enrollment enables filtering immediately; embeddings survive container/session restarts under ignored `/data`, and deletion removes them. Profiles and enrollment audio are never logged or sent remotely.
- [ ] With TV speech only, no `speech_started`, STT, or barge-in occurs during a 10-minute playback test. A registered user speaking alone starts and completes a turn.
- [ ] While TV speech continues, the registered user's completed utterance reaches STT within 1.2 seconds after they stop; unregistered speech cannot hold the turn open. Simultaneous TV speech may still contaminate the transcribed audio.
- [ ] Explicitly enrolled and enabled guests can trigger turns; removing or disabling a guest revokes access. Other voices remain excluded.
- [ ] On the target Apple Silicon Linux ARM64 CPU container, the speaker-scoring path keeps up with real time (p95 processing under 160 ms per 160 ms chunk for the owner profile), uses bounded buffering, and reports latency/false-activation metrics. Record measured values; do not assume performance.
- [ ] Browser, WebSocket, gating/endpoint, persistence, offline startup, and existing barge-in/Local/Remote regressions are covered by deterministic tests and a real-microphone/TV check.

## 4. Analysis
Silero ONNX VAD is still useful as a speech signal but cannot identify who spoke. Speaker identity must gate both turn onset and endpoint; checking identity only after transcription would leave the TV-induced hanging-turn problem intact. Existing 32 ms audio frames must be accumulated into 160 ms scoring chunks without blocking WebSocket reads. Multiple allowed speakers increase CPU work; keep the allowlist bounded and benchmark it.

## 5. Proposed Solution
- Use [tuya/nomo-pvad](https://github.com/tuya/nomo-pvad), pinned to `release-1.1` (Apache-2.0). It performs streaming target-speaker VAD on 16 kHz PCM every 160 ms using an enrolled 192-dimensional embedding. Unlike Picovoice Eagle, it needs no license server or AccessKey.
- Extend `models.json` and `tools/download_models.py` to prefetch and verify both the Nomo-PVAD checkpoint and its separate upstream ERes2NetV2 enrollment checkpoint, with pinned artifact revisions/checksums; verify availability before loading and pass explicit local model paths. Lock compatible CPU `torch`/`torchaudio` dependencies for Python 3.12 and Linux ARM64. Never permit ModelScope's implicit first-run download.
- Add a shared model loader and per-session speaker scorer to `workspace/voice_pipeline.py`. Buffer five 512-sample frames per 160 ms score; score configured allowed profiles, combine by maximum probability, and use configurable hysteresis (initial start: two consecutive scores >= 0.75; end: 640 ms without accepted-speaker speech). Preserve pre-roll, minimum speech, maximum utterance length, and existing cancellation behavior. Silero alone must never extend a gated turn.
- Extend `workspace/app.py` WebSocket control with begin/finish/cancel enrollment and profile status. Route enrollment audio away from normal turns, atomically save only embeddings under `/data` with restricted permissions, and apply revised profiles safely between audio chunks. Update `workspace/static/index.html`, `app.js`, and `app.css` with **Detect me**, optional allowed-speaker registration/removal, and feedback.

## 6. ToDo
- [ ] Pin and verify both model assets and CPU dependencies; prove startup without internet.
- [ ] Implement persistent enrollment, per-speaker scoring, and identity-aware turn boundaries.
- [ ] Add the browser controls and WebSocket events.
- [ ] Add deterministic and real-audio tests; benchmark CPU and tune thresholds on TV overlap.
