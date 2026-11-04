# Specification 001: EL Voice — Local Dota Voice Conversation

## 1. Problem Statement

Complete the EL Voice prototype so a user can have a natural, hands-free conversation with a local Dota-focused assistant in a browser, hear responses in the selected Silero `b_lez` voice, and interrupt the assistant by speaking. The current container serves a bootstrap status page only. It does not capture microphone audio, run speech recognition or language-model inference, or conduct a conversation.

## 2. Design Statement

Implement the browser interface and voice pipeline in this project, using the existing Docker Compose deployment and locally prepared models:

- Browser microphone capture and audio playback run on the Mac. The browser communicates with the application through the existing Nginx proxy at `http://localhost:8765`; no host audio devices are mounted into Docker.
- The application runs in the existing Linux ARM64 CPU container. Keep the voice service on its private Compose network and expose the browser endpoint only through the loopback-bound proxy.
- Keep the microphone active for the duration of an explicitly started browser session. Stream audio frames to the application and use a local streaming Voice Activity Detection (VAD) component to detect speech onset and end-of-utterance. Preserve a small configurable pre-roll buffer so the start of a word is not lost.
- Select and document an upstream local VAD implementation before adding dependencies or model assets. Prefer a streaming-capable Silero VAD implementation if it fits the existing Python 3.12 / Linux ARM64 CPU environment; use Vosk partial recognition only if VAD alone cannot meet the measured interruption behavior. Pin any new package, document the model source and integrity check, and update the lock/model manifest as needed. Faster-Whisper's `vad_filter` may help process a completed utterance, but it does not by itself satisfy continuous speech-onset detection or hands-free barge-in.
- Use the existing local Whisper `small` model for final utterance transcription, Gemma 3 1B Q4_K_M for response generation, and Silero `v4_cyrillic.pt` with speaker `b_lez` for speech synthesis. No cloud inference, API keys, or runtime internet access are required.
- Begin synthesizing and playing completed text chunks while the LLM is still generating later text. Do not wait for the complete LLM response or complete TTS audio before starting playback. Use bounded audio buffering and tag every turn event and audio chunk with a `turn_id`.
- When VAD detects a new user utterance during LLM generation or TTS playback, interrupt automatically without requiring a button. Stop playback, clear queued audio, cancel the active generation and synthesis, and accept the new utterance. Ignore every delayed result carrying the cancelled `turn_id`. Use an engine-supported cancellation mechanism or an isolated, terminable worker when needed; cancelling an asyncio task alone is not proof that blocking CPU inference stopped.
- Show a browser checkbox labeled **Interrupt assistant when I speak**, checked by default. The user can change it before or during a session; the current selection must be sent to the voice service and retained across socket reconnects for that page session. When checked, speech onset during an active assistant turn interrupts it as above. When unchecked, keep listening, but ignore speech that begins while an assistant turn is active: do not cancel its generation, stop its playback, or start a competing turn. Accept the next utterance after the reply finishes. Explain this behavior beside the control. The setting must work in both Local Voice and Remote Voice.
- The supported acoustic scenario for this MVP uses headphones. Request browser microphone `echoCancellation: true` and inspect the selected track settings when available. Treat browser echo cancellation as best effort; do not claim reliable speaker playback support. Explain the headphone requirement in the UI and prevent the assistant's own playback from triggering a false barge-in in the supported setup.
- Keep audio and transcripts in memory only. Do not persist them or include their contents in logs. Record only non-sensitive timing and operational metrics needed to verify latency and readiness.
- Write an original implementation. Do not copy or import the referenced author's application code, installer, packages, model mirrors, or binary artifacts. Use upstream components and assets, documenting any newly introduced component.
- Preserve the existing read-only application source mount and writable `/data` mount. Prefer editing mounted source files so routine application changes do not require rebuilding the image.

## 3. Acceptance Criteria

Verify these in the running Compose deployment and an actual browser on the Mac. No criterion is complete until it has been observed in the running application.

### P0 — Required for a usable voice conversation

- [ ] **Browser session and continuous listening:** The page provides explicit start/stop controls and clear listening, processing, speaking, reconnecting, and error states. Once a session starts and microphone permission is granted, capture and stream audio continuously until the user stops or disconnects the session. Stopping releases the microphone and closes server-side session resources.
- [ ] **Local speech onset and endpointing:** A local streaming VAD detects the beginning of speech and the end of each utterance without requiring a push-to-talk control. It retains enough pre-roll to preserve initial phonemes, rejects silence/noise as empty turns, and can listen while the assistant is speaking. Document and tune the VAD thresholds, minimum speech duration, and end-of-speech silence handling against real microphone input.
- [ ] **Accurate final transcription:** At utterance end, transcribe the captured turn with the configured local Whisper model inside the container. VAD used for endpointing is distinct from any post-hoc Whisper `vad_filter`. No audio is sent to a cloud service.
- [ ] **Local Dota response:** Pass each finalized Russian-language utterance once to the configured local Gemma model and produce a relevant Russian-language response as a Dota-focused assistant. Inference requires no API key or external service.
- [ ] **Incremental response playback:** Stream LLM output, form speakable text chunks, and start playing the first Silero `b_lez` audio chunk while the LLM is still generating subsequent text. The browser plays received audio incrementally; the implementation does not buffer the entire response before playback.
- [ ] **Automatic barge-in:** When local VAD detects speech onset during either LLM generation or TTS playback, stop current playback and flush queued audio without a user button. Cancel current work using a mechanism that actually stops or bounds the underlying CPU inference, not only its awaiting coroutine. Process the new utterance as a new turn.
- [ ] **User-controlled barge-in:** The rendered page shows a checked-by-default interrupt checkbox. With it checked, a VAD speech onset during an active assistant turn cancels that turn and accepts the new utterance. With it unchecked, speech during an active reply does not cancel generation, stop playback, or start another turn; the reply completes and a later utterance is accepted. Changing the control during a session updates the active WebSocket session, and reconnect uses the selected value. Verify the visible control and both behaviors through browser/app and WebSocket tests in Local and Remote mode.
- [ ] **Turn isolation:** Assign a unique `turn_id` to each user turn and propagate it through recognition, generation, synthesis, and browser events. After cancellation, reject stale results and audio from the old turn; verify that no old speech resumes or overlaps the replacement answer.
- [ ] **Echo behavior:** In the supported headphone scenario, the assistant does not trigger its own barge-in while speaking. Request `echoCancellation: true`, inspect `MediaStreamTrack.getSettings()` where available, and surface a clear headphone notice. Document that browser echo cancellation is best effort and that speaker playback is unsupported for this MVP.
- [ ] **Browser-to-container routing:** Complete conversations and WebSocket upgrades work through `http://localhost:8765`; the voice service port remains unpublished on the host.
- [ ] **Local operation and privacy:** Once models and any explicitly approved VAD assets are present, the conversation runs without external inference or runtime downloads. Audio and transcripts are not persisted and their contents do not appear in application or proxy logs. Verify this by inspecting the configured storage and logs after a session.
- [ ] **Readiness and recovery:** Health and status distinguish a live HTTP process from a fully initialized pipeline and report model/VAD readiness without secrets. Handle empty or failed recognition, model errors, browser disconnect/reconnect, session stop, and repeated turns without a server crash or leaked microphone/session resources.

### P1 — Required verification and measured performance

- [ ] **Latency measurements:** Record monotonic timestamps for speech-end to first audible assistant audio and VAD speech-onset to stopped assistant playback. Measure at least 10 representative turns on the target Mac/container, report p50 and p95 for both measures, and propose performance targets from those measurements. Agree the targets before marking latency acceptance complete; do not invent thresholds before the baseline exists. Do not record transcript or audio content with these metrics.
- [ ] **Conversation and interruption scenarios:** In the actual browser, verify multiple consecutive turns; barge-in during LLM generation; barge-in during TTS playback; no false self-interruption during TTS with headphones; stop and restart a session; browser disconnect and reconnect; and no audio/transcript persistence or content-bearing logs.
- [ ] **Reproducibility:** Verify the application with the existing image or document any necessary rebuild and exact dependency/model change. Run focused syntax and pipeline checks, then complete the manual browser scenarios above. Record commands, measurements, results, and known limitations.

## 4. Analysis

The prepared environment is running on Docker Desktop for Apple Silicon. The host port is published through Nginx at `127.0.0.1:8765`, the voice service remains on a private network, all six expected model files have been verified, and standalone Silero synthesis succeeded with speaker `b_lez`. These checks establish container and TTS readiness only; they do not demonstrate STT inference, LLM inference, VAD, browser microphone capture, streaming playback, or a conversation.

The current `workspace/app.py` is an explicit bootstrap status server. The previous draft named Whisper but did not define continuous listening, local speech-onset detection, or endpointing. Faster-Whisper's `vad_filter` is a filter for speech regions during transcription of supplied audio; it is not by itself a continuous microphone-session controller. The project already locks `onnxruntime`, but the exact streaming VAD API, state handling, model source, and CPU cost still need to be confirmed before implementation. The implementation must record that choice and any new dependency or model asset rather than silently adding one.

The previous draft also described cancellation at the asyncio/task level without proving that blocking CPU inference stops, allowed sequential LLM-then-TTS implementations to pass, and did not define echo support or latency evidence. The revised requirements close those gaps with per-turn identity, actual cancellation verification, overlapped LLM/TTS streaming, a headphone-supported acoustic scenario, and measured p50/p95 timing before targets are agreed.

The browser owns microphone permission and playback on macOS; Docker must not receive a host audio-device mount. Browser echo cancellation is requested but is not guaranteed to remove all loudspeaker sound. Headphones are therefore the supported MVP scenario. Any work on reliable open-speaker use, Dota game control, external APIs, or persistent conversation history is outside this specification.

The referenced article and release are architectural reading only. The author's documented Vosk-plus-Faster-Whisper design motivates the separation between fast speech-onset/interruption detection and accurate final transcription; this project should implement that behavior independently with upstream components. Faster-Whisper documents its integrated Silero VAD filter in its [README](https://github.com/SYSTRAN/faster-whisper#vad-filter). Browser echo-cancellation constraints and their best-effort behavior are documented by [MDN](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints/echoCancellation). The article reference is [Habr](https://habr.com/ru/articles/1101954/).

## 5. Proposed Solution

Replace the bootstrap page with a browser client and an asynchronous pipeline in the mounted `workspace/`. Keep a microphone stream active for each user-started session and send audio frames through the existing WebSocket proxy. Implement a local, streaming-capable VAD with explicit onset, endpointing, pre-roll, and silence handling. Use VAD onset to trigger immediate barge-in and VAD endpoint to finalize a turn for local Whisper transcription.

Pass the finalized transcript to Gemma and stream generated text into a bounded sentence/chunk queue. Synthesize each ready chunk with Silero `b_lez` and return audio chunks tagged with the current `turn_id`, allowing browser playback to begin before LLM generation completes. On barge-in, stop browser playback, flush server and client queues, cancel/terminate the active model work using verified engine behavior, and invalidate the old turn ID. Request and verify browser echo-cancellation settings where supported and document headphones as required for the supported MVP scenario.

Implement and verify incrementally: choose and characterize the VAD and cancellation mechanisms; build the browser microphone/WebSocket session; add VAD and Whisper endpointing; add Gemma streaming and overlapping TTS playback; then harden barge-in, echo behavior, reconnects, privacy, and metrics. Keep the existing Nginx route and network boundary unless the implementation proves a change is necessary. Keep all source code, UI text, and project documentation in English.

## 6. ToDo

- [ ] **Specification approved for implementation.**
- [ ] Confirm a streaming local VAD implementation and the actual CPU cancellation behavior of the selected STT, LLM, and TTS engines. Record any new pinned dependency, model source, integrity check, and rebuild requirement before adding it.
- [ ] Implement the browser session UI, continuous microphone capture, audio constraints, reconnect behavior, and proxied WebSocket protocol.
- [ ] Implement local VAD onset/endpointing with pre-roll, silence handling, and final Whisper transcription.
- [ ] Implement per-session/per-turn orchestration with Gemma streaming, overlapping Silero `b_lez` synthesis/playback, `turn_id` validation, and effective cancellation.
- [ ] Implement the headphone-supported echo policy and verify the assistant does not trigger its own barge-in.
- [ ] Add the default-on browser interrupt control and verify enabled/disabled behavior during an active reply.
- [ ] Measure p50/p95 response and interruption latency on the target Mac, then propose targets from the measured baseline.
- [ ] Verify every P0/P1 criterion in the actual container and browser; record scenarios, commands, measurements, and known limitations.

Implementation evidence and remaining gaps: [EL Voice verification](../README.md).
