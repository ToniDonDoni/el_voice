# EL Voice Implementation Report

## Summary

Implemented the browser-to-container voice pipeline in the mounted `workspace/` source. It uses the existing image and installed packages; no Docker rebuild or dependency change is needed. The voice service must be restarted to load source edits.

The task specification was restored unchanged from the current branch. Its Problem Statement, acceptance criteria, and checklists were not edited. This report records observed results and limitations against those original criteria.

The user requested that local Docker logs show what was recognized, what the assistant answered, and errors. Application stdout now emits one-line structured JSON events for user transcripts, complete or interrupted assistant replies, browser/model/session errors, VAD turn boundaries, generated TTS chunks, and browser playback scheduling. Audio payloads are never logged. Docker may retain the text logs; they should be treated as conversation history. Nginx access logs do not contain WebSocket message bodies.

## Verification performed

- `docker compose restart voice` completed; `/api/status` reported `ready` and all four models/components (`vad`, `stt`, `llm`, `tts`) ready on Linux `aarch64`.
- `curl --fail http://127.0.0.1:8765/api/status` succeeded through the published proxy. The voice port remains unpublished on the host.
- Ran a synthetic Russian speech turn through the running WebSocket service. The backend produced a transcript, streamed assistant text, generated 14 TTS chunks totaling 3,270,000 bytes, accepted a browser-style playback acknowledgement, and completed the turn.
- Inspected `docker compose logs`: `speech_started`, `speech_ended`, `user_transcript`, `tts_chunk_ready`, `playback_started`, `assistant_reply`, `playback_finished`, and `session_stopped` appeared as structured events. No model, turn, WebSocket, or browser error occurred in that synthetic run. The transcript and reply values are present in local logs and are intentionally not repeated in this report.
- Posted a browser-diagnostic probe through Nginx; `/api/client-error` returned HTTP 204 and emitted a `browser_error` log. A later probe left the literal test message `log test` in the Docker log; it is a test record, not a browser failure.
- One synthetic measurement from speech endpoint to browser playback scheduling was about 4.9 seconds. This measures scheduling acknowledgement, not sound from the Mac's physical output; it is one run, not the required 10-run baseline.
- Browser inspection confirmed the page loads, models are ready, the microphone is off while idle, and the headphone notice is visible. Earlier in the live browser session, microphone speech was recognized and assistant text appeared, but the user reported hearing no voice. I could not verify that audio was audible from the Mac's selected physical output.
- Python source parsed successfully with Python's in-memory `compile`; `node --check workspace/static/app.js`, `python3 -m json.tool workspace/settings.json`, and `git diff --check` passed. A first `py_compile` attempt could not write `__pycache__` into the read-only workspace mount; the no-write syntax check passed instead.
- Replaced deprecated FastAPI `on_event` hooks with a lifespan handler. The latest restart reached `ready`; historic Docker output still contains warnings from the preceding startup, but the current startup no longer emits those deprecation warnings.
- No application audio or transcript was written during the conversation test. `data/tts-test.wav` is an existing standalone TTS smoke-test artifact from the earlier setup and is not a captured conversation recording.

## Acceptance criteria

| Criterion | Status | Evidence / remaining work |
| --- | --- | --- |
| Browser start/stop, continuous capture, clear states | Partial | UI and prior live microphone capture work. Must recheck start/stop and resource release in the user's browser after this restart. |
| Streaming VAD onset/endpoint, pre-roll, noise rejection | Partial | Synthetic input reached onset, endpoint, and Whisper. Thresholds are configured; microphone tuning and false-trigger testing remain. |
| Final Whisper transcription | Partial | Synthetic speech and a prior live microphone turn produced transcripts. Accuracy has not been measured against a representative sample. |
| Relevant local Dota response | Partial | Gemma generated a response, but the synthetic Dota answer was low quality and included irrelevant/inaccurate content. Improve or accept the 1B model's limits before calling this criterion complete. |
| Incremental Silero playback | Partial | The synthetic client received 14 audio chunks; logs show chunk generation and playback scheduling. Real browser audibility was not confirmed. |
| Automatic barge-in and bounded cancellation | Partial | VAD-driven turn invalidation and llama.cpp EOS cancellation are implemented. Whisper and TTS are blocking CPU calls whose current invocation cannot be forcibly stopped; their stale result is discarded. Need real interruption tests during both LLM and TTS. |
| `turn_id` stale-result isolation | Implemented; runtime scenario pending | IDs guard outbound events/audio and browser playback. Need verify with live barge-in that old audio never resumes. |
| Headphone/echo behavior | Partial | Browser requests `echoCancellation: true`; prior browser diagnostics reported it enabled and the headphone warning is visible. No controlled headphone playback/no-self-trigger test was completed. |
| Proxy and WebSocket routing | Pass for observed browser use | The page and earlier browser WebSocket session used `http://localhost:8765`; voice has no host-published port. The latest synthetic session connected directly to the voice container. |
| Local operation, privacy, and no conversation audio/transcript persistence or content-bearing logs | Not met as written | The original criterion remains unchanged. The requested debugging change logs recognized text and assistant replies, so the no-content-bearing-logs clause is not met. Audio payloads are not logged and no conversation recording was created. Docker log retention can retain conversation text. A separate old TTS sample remains in `data/`. |
| Readiness and recovery | Partial | Readiness distinguishes model load and all models reached ready after restart. Synthetic session closed cleanly. Empty STT, injected component errors, reconnect, and repeated browser turns need manual verification. |
| 10-turn p50/p95 latency and agreed targets | Not complete | Only one synthetic sample was observed: about 4.9 seconds to scheduled playback. No 10-turn target-Mac/browser baseline or agreed thresholds. |
| Required browser interruption scenarios | Not complete | I could not verify audible playback from the physical output. Multiple turns, LLM/TTS interruption, no false self-interruption with headphones, restart, disconnect/reconnect, and full audio/storage checks have not all been demonstrated. |
| Reproducibility | Partial | Existing image works with mounted source changes; syntax, readiness, and synthetic pipeline checks passed. Manual browser scenarios remain. |

## How to inspect the next live turn

In the repository root, run:

```sh
docker compose logs --follow --no-log-prefix voice
```

After speaking, check for `user_transcript`, `assistant_reply`, one or more `tts_chunk_ready` events, then `playback_started`. Any failures should appear as `turn_error`, `browser_error`, `websocket_error`, or a model initialization traceback. If the transcript and reply are logged and TTS chunks exist but there is no audible sound, `playback_started` only proves that the browser scheduled the buffer. Check the Chrome tab's audio output selection, macOS output device, mute state, and volume; browser automation cannot verify physical speaker output.
