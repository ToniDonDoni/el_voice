# SPEC-005: Local Target-Speaker Gating

## 1. Problem and Current State
- Browser AudioWorklet sends 16 kHz mono PCM in 512-sample frames over `/ws/voice`.
- `StreamingSileroVAD` in `workspace/voice_pipeline.py` detects any speaker. `receive_audio()` in `workspace/app.py` uses it for turn boundaries and barge-in. Continuous TV speech can create false turns and prevent the end-of-utterance timeout.
- `/ws/voice` currently accepts connections without an explicit WebSocket `Origin` check. Adding biometric-profile controls requires protecting that boundary.

## 2. Required State
- Hold **Detect me** to capture an enrollment sample; release to create and save the owner's voice embedding locally. This is enrollment, not model training. Invalid or canceled samples never replace an existing profile.
- The owner may explicitly enroll, enable, disable, or remove additional permitted speakers, with their consent. Only enrolled and enabled speakers open, extend, or interrupt speech turns. TV and unknown speakers cannot reset the owner's endpoint timer.
- The UI displays capture progress, filtering status, and errors; profiles can be deleted. With no profile configured, legacy VAD is available with an explicit **filtering off** status. If enabled filtering fails or its profile is invalid, fail closed rather than silently accepting everyone.
- Audio analysis and embeddings stay on the local Mac in both Local and Remote Voice; no speaker audio or biometric embedding goes to a remote provider. Existing Remote Voice disclosure of *recognized text* is unchanged. Never retain raw enrollment recordings or put embeddings in telemetry, responses, or Git.
- Speaker filtering is a convenience feature, **not authentication or anti-spoofing**. Replayed/imitated voices and overlapping TV speech remain limitations.

## 3. Acceptance Criteria
- [ ] **Detect me** works on pointer/touch/keyboard hold and release. At least 10 seconds of usable enrollment speech are required; canceled, short, silent, or interrupted capture leaves the prior profile intact and produces a clear error. Enrollment bypasses STT, LLM, and TTS.
- [ ] Owner and consented guest embeddings are created locally and persist across restart under ignored `/data`, with restricted directory/file permissions and atomic replacement. Deletion removes saved profiles and active allowlist entries. Profile bytes and enrollment PCM are never logged or exposed over APIs.
- [ ] Ten minutes of TV-only test playback produces zero turns or barge-in events. A registered owner speaking alone produces a completed STT turn. A registered guest works only while enabled; an unknown speaker never starts a turn.
- [ ] While TV continues talking, the registered user's turn closes within 1.2 seconds after their last speech; unregistered speech cannot extend it. Simultaneous TV speech may still pollute the resulting STT audio.
- [ ] Speaker inference is local and offline after one-time model provisioning. Missing, tampered, or corrupt artifacts fail readiness/filtered speech closed, with **no implicit ModelScope download**, fallback-to-any-speaker, or raw-audio transmission.
- [ ] WebSocket handshakes reject unexpected `Origin`; enrollment/profile control messages are validated, size/rate-limited, and unavailable to cross-origin pages. Only explicitly allowed local origins can manage profiles.
- [ ] On the target Apple Silicon Linux ARM64 CPU container, p95 processing stays below 160 ms per 160 ms chunk with bounded queues; capture actual inference latency, false activations, and missed speech, then tune thresholds.
- [ ] Deterministic browser/WebSocket/endpoint/persistence/security tests and real microphone + TV tests pass; Local/Remote Voice, cancellation, and barge-in regressions remain intact.

## 4. Analysis
Silero ONNX VAD cannot distinguish a nearby user from a TV. Identity must affect both onset and endpoint; filtering only after STT leaves the hanging-turn problem intact. Existing 32 ms frames must be accumulated into 160 ms target-speaker chunks off the WebSocket receive loop. The upstream Nomo-PVAD wrapper automatically downloads ERes2NetV2 if its local checkpoint is missing; its `.pt` checkpoints require safe loading. The current Compose voice service already has no external network egress and uses read-only `/models`; preserve those protections.

## 5. Proposed Solution
- Adopt [tuya/nomo-pvad](https://github.com/tuya/nomo-pvad) **release-1.1** (Apache-2.0), pinned to source commit `cb828468d183424fd574984f00b3ffbcce0be9bc`. Its `weights/nomo_pvad.pt` SHA-256 is `0acf2f3e75409b172771663fd1bee31af2760dba4e25b1353d2c26726b6412d1`. It emits one target-speaker probability per 160 ms of 16 kHz PCM. Enrollment uses the separate full ERes2NetV2 checkpoint (`iic/speech_eres2netv2_sv_zh-cn_16k-common`) to create a 192-dimensional embedding.
- Extend `models.json` and `tools/download_models.py`: pin an immutable ERes2NetV2 revision and verified hash as well as the Nomo asset; download both in setup mode, never at runtime. Use vetted, pinned source and compatible locked CPU `torch`/`torchaudio` dependencies. Reject missing files **before** constructing the upstream wrapper, force explicit local paths, use `torch.load(weights_only=True)`, and reject unverified artifacts. Do not enable SDK telemetry or model-execution downloads.
- Add a shared model and per-session scorer in `workspace/voice_pipeline.py`, consuming five 512-sample frames per 160 ms. Accept the maximum probability across enabled profiles. Initial hysteresis: two consecutive scores >= 0.75 start speech; end after 640 ms without accepted-speaker speech. Preserve pre-roll, max utterance time, and cancellation. Silero-alone activity never extends a filtered turn; do not block the WebSocket receive loop during CPU inference.
- Extend `workspace/app.py` with enrollment begin/end/cancel, profile status/delete/allowlist controls, strict WebSocket Origin validation, input bounds, and atomic restricted-permission embedding storage. Route enrollment frames outside the normal turn pipeline. Add **Detect me** and guest management to `workspace/static/index.html`, `app.js`, and `app.css`; surface profile/filtering state without returning embedding contents.

## 6. ToDo
- [x] Pin/checksum both model assets and locked dependencies; confirm offline startup and no egress.
- [x] Implement profile enrollment, restricted persistence, speaker-aware onset/endpoint, and failure handling.
- [x] Implement protected WebSocket controls and accessible browser UI.
- [ ] Complete physical microphone/TV acceptance, guest-consent checks, ten-minute TV-only playback, and CPU/TV threshold tuning.

## 7. Implementation Verification
- Linux ARM64 image build, dependency check, Compose validation, pinned speaker-asset verification, and offline model startup passed.
- Python unittest discovery passed (87 tests); Node browser/audio tests passed (2 tests).
- The real local STT -> Gemma -> TTS self-test passed. Synthetic WebSocket enrollment, profile enable/delete, and Nomo target/background examples were exercised without retaining sample audio or the temporary profile.
- The labeled Nomo example produced one accepted target-speaker turn and no turn for the background-only segment. This is synthetic evidence, not a microphone/television acceptance test.
- Latency remains unaccepted: the live status p95 was 166 ms in an unknown-only sample and rose above 400 ms while an accepted sample also exercised STT/LLM/TTS. One full-rate mixed sample emitted a bounded-queue fail-closed event; slower synthetic playback did not. Continue ARM64 CPU and threshold tuning before claiming the 160 ms criterion.
