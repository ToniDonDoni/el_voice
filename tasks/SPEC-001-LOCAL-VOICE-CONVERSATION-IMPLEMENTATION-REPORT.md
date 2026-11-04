# EL Voice Implementation Report

## Summary

Implemented the SPEC-001 review corrections in the mounted `workspace/` source. No dependency or image rebuild is needed. The running voice process has not been restarted, so its imported Python code is still the previous version; restart it to activate the changes. Browser assets are mounted and were verified in the page after cache-busting.

The user requested logs containing recognized speech, assistant replies, and errors. Those text records remain in local Docker logs, as requested; audio payloads are not logged. This means the unchanged specification's no-content-bearing-logs criterion is **not met**.

## Verification performed

- Python regression suite in the existing `el-voice:dev` image: 18 tests passed (`unittest discover -s /workspace/tests -p 'test_review_*.py'`). Coverage includes duration-limited PCM packetization and browser-credit backpressure over a real Uvicorn WebSocket, preservation of reserved duration while a later packet remains queued behind the sender, stalled-sender teardown/replacement, two contextual WebSocket turns, bounded speech chunks, startup validation, repeated replacement against non-cooperative fake inference workers using production runtime locks, VAD recurrent state, and log payload privacy.
- `node workspace/tests/browser_playback.test.cjs`: passed. Includes packet validation, stale socket audio/text rejection, playback teardown, gain routing, volume/mute logic, and monotonic queue acknowledgements from both scheduling and source completion.
- `node workspace/tests/audio_capture.test.cjs`: passed for 44.1/48 kHz resampling, callback partitioning, and buffered tails.
- `git diff --check` and Python in-memory syntax compilation passed.
- Re-ran the offline real-model self-test in a separate capped, no-network container without stopping the running service or writing audio files. Whisper recognized synthesized `b_lez` audio; Gemma returned a response; Silero produced 24 kHz audio. This run measured STT cancellation residuals of 1821, 1590, and 1353 ms; LLM generation residuals of 31, 28, and 56 ms; LLM first-token/prompt residual of 24 ms after cancellation before the first token; and TTS residual of 962 ms. STT and TTS discard stale results but do not stop an in-flight native CPU operation immediately. The generated TTS response lasted 52.65 seconds, so response concision/quality still needs evaluation.
- Rendered-browser inspection at `http://localhost:8765/` confirmed mute and volume controls visible, Start visible, End session hidden while idle, and the mute button toggled its accessible state. No microphone permission was requested and no audio was recorded.
- The app's FastAPI process was not restarted. Therefore the real-model test exercised the changed runtime directly, while the running site's current WebSocket server still has old in-memory Python code. Full new-code browser conversation through Nginx and physical audibility remain unverified. No browser microphone session was started during these checks.

## Acceptance criteria

| Original criterion | Status | Evidence and remaining work |
| --- | --- | --- |
| Browser session, continuous capture, start/stop and resource release | Partial | Browser UI and previous microphone session exist; actual start/stop and track release were not repeated against the updated server. |
| Streaming local VAD, onset/endpoint, pre-roll and noise rejection | Partial | Bundled Silero state and VAD integration tests pass; microphone noise and threshold tuning remain unverified. |
| Final local Whisper transcription | Partial | Real-model self-test transcribed generated `b_lez` speech; representative microphone accuracy remains unmeasured. |
| Relevant local Dota response | Partial | Gemma inference completed, but response quality and relevance were not reviewed; one generated response led to 52.65 seconds of TTS. |
| Incremental LLM-to-Silero playback | Partial | Regression test proves audio is queued before generation ends; physical browser playback with the updated server remains unverified. |
| Automatic barge-in and bounded cancellation | Partial | Turn events, cancellation-aware waits, locks, and stale-result rejection are covered. Real-model residual work was measured; Whisper and Silero calls still finish their current native operation. Actual browser interruption remains untested. |
| Unique turn identity and stale-result isolation | Partial | Binary packet/session tests and stale browser socket tests pass; no live stop/start acoustic run was repeated. |
| Headphones and echo behavior | Partial | UI explains headphone requirement and requests browser echo cancellation; no physical headphone/no-self-trigger test was performed. |
| Browser-to-container route through Nginx | Partial | Rendered page is reachable at the proxied address; updated Python WebSocket behavior has not been activated in the running process. |
| Local operation, no audio persistence, no content logs | Not met as written | The self-test used no network and wrote no audio. Requested transcripts/replies remain in Docker logs, which conflicts with the unchanged criterion. No audio payload is logged or saved by the self-test. |
| Readiness and recovery | Partial | Invalid settings fail readiness; real models initialized in the isolated self-test. New service restart, empty/error paths, and browser reconnect remain unverified. |
| Ten-turn latency baseline and agreed targets | Not complete | Only model cancellation samples were measured. No 10-turn end-to-end browser p50/p95 baseline or agreed targets exist. |
| Browser conversation/interruption scenarios | Not complete | Repeated actual turns, LLM/TTS interruption, echo behavior, stop/restart, reconnect, and physical audibility remain manual checks. |
| Reproducibility | Partial | Existing image and read-only mounts were used. Automated commands and a reusable offline real-model self-test are present in the working tree; end-to-end browser acceptance is outstanding. |

## Activate the implementation

The changed Python files are bind-mounted into the current container, so rebuild is unnecessary. Restart the voice service to load them:

```sh
docker compose restart voice
curl --fail http://localhost:8765/api/status
```

The reproducible real-model check is:

```sh
docker run --rm --network none --read-only \
  --tmpfs /tmp:size=512m,mode=1777 --cpus 4 --memory 6g \
  --user 10001:10001 -e EL_VOICE_MODELS=/models \
  -v "$PWD/workspace:/workspace:ro" -v "$PWD/models:/models:ro" \
  -w /workspace --entrypoint python el-voice:dev \
  /workspace/tests/real_model_pipeline_selftest.py
```

The remaining physical microphone, echo, interruption, audible-output, and 10-turn performance criteria require an actual browser session on the target Mac. The application should be restarted before those checks.
