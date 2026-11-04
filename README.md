# EL Voice

An independent local voice assistant for Docker Desktop on Apple Silicon. A browser on the
Mac captures the microphone; everything else — speech-onset detection, final transcription,
Dota-focused answer generation and speech synthesis — runs inside the Linux ARM64 container
on CPU. No cloud inference, API keys or runtime downloads are used.

The conversation pipeline is implemented in `workspace/`. The acceptance criteria in
[`tasks/SPEC-001-LOCAL-VOICE-CONVERSATION.md`](tasks/SPEC-001-LOCAL-VOICE-CONVERSATION.md)
still require a run in the real container with a real browser and microphone; the
"Verification status" section below records exactly what has and has not been observed.

No source code, installer, infrastructure package, binary build, or model mirror from the
previously reviewed assistant author is used here.

## What the image installs

| Component | Source / purpose |
| --- | --- |
| Python 3.12 on Debian Bookworm | Official Docker image, pinned by digest |
| FFmpeg, OpenMP, libsndfile, curl, CA certificates | Debian repositories; audio decoding and CPU inference |
| FastAPI, Uvicorn, WebSockets | Browser/API layer and WebSocket audio transport |
| Faster-Whisper, CTranslate2, PyAV, ONNX Runtime | Local speech recognition, streaming VAD and audio processing |
| llama-cpp-python | Official upstream Linux ARM64 CPU wheel pinned by SHA-256 |
| PyTorch CPU | Official PyTorch Linux ARM64 wheel; no CUDA dependencies |
| NumPy, Hugging Face Hub and resolved dependencies | Supporting libraries |

Direct inputs are in `requirements.in`. Python runtime dependencies and artifact hashes are
resolved in `requirements.lock` for CPython 3.12 / Linux ARM64. The voice pipeline does not
add a Python package: streaming VAD uses the already-locked ONNX Runtime and a publisher
ONNX model file, so the existing image can run the browser application without a rebuild.
Debian packages are not a complete snapshot, so this is not a byte-reproducible image.
Changing the lock requires `uv` on the development machine and `sh tools/lock.sh`.

There is no host audio-device mount, GUI toolkit, Vosk requirement, CUDA toolkit, or Git
checkout of someone else's application.

## Prepare local directories

Run commands from the repository root:

```sh
sh tools/init.sh
docker compose config --quiet
```

Initialization writes an ignored `.env` with the host account's numeric IDs. This lets the
download container write to the bind-mounted `models/` directory without running as root.
An existing `.env` is preserved. Set `EL_VOICE_PORT` there to change the host port.

| Host folder | Container folder | Access |
| --- | --- | --- |
| `workspace/` | `/workspace` | Read-only to the process; edit source on the Mac |
| `models/` | `/models` | Read-only in the voice service; writable in the download service |
| `data/` | `/data` | Writable cache and future optional application state |

The repository root, Git metadata, home directory and Docker socket are not mounted.
Only the Dockerfile and the small bootstrap tools enter the build context. Models do not
inflate image layers and survive container recreation.

## Build and start

Give Docker Desktop approximately 12 GB memory and at least 15 GB free disk space.
The runtime service is limited to 10 GB and 6 CPUs. The first image build downloads the
Python dependencies; it no longer compiles llama.cpp. No AMD64 emulation or GPU access is used.
Nginx runs from a separate official unprivileged image; Compose pulls it on first start, so
adding the proxy does not rebuild `el-voice:dev`.

```sh
docker compose build voice
docker compose up -d voice proxy
docker compose ps
curl --fail http://localhost:8765/healthz
```

Open **http://localhost:8765** in a browser and press **Start conversation**.

The only published mapping is `127.0.0.1:8765 -> proxy:8080`; Nginx forwards HTTP and
WebSocket requests to `voice:8000` over the internal network. The voice service has no
published port and no normal outbound internet route. Nginx joins a separate bridge network
for host port publishing; that bridge permits potential outbound access from the Nginx
container. Nginx's fixed upstream config forwards requests only to `voice:8000`. The
download service uses egress to fetch models, while `voice` remains isolated on its internal
network.

Application source changes are visible immediately in `/workspace` and only need
`docker compose restart voice`; they do not need an image rebuild. The exception is a change
to the model manifest or to `tools/`: those files are baked into the image and need
`docker compose build voice` (see "Models and downloads").

## Models and downloaded assets

| Model | Publisher | Approximate size |
| --- | --- | --- |
| Gemma 3 1B IT Q4_K_M GGUF | ggml-org on Hugging Face | 806 MB |
| Whisper small, CTranslate2 format | Systran on Hugging Face | 464 MB including tokenizer/config |
| Silero v4 Cyrillic, selected voice `b_lez` (male Lezghian) | models.silero.ai | 34 MB |
| Silero VAD v5.1.2 streaming ONNX | snakers4/silero-vad (MIT) | 2.3 MB |

`models.json` pins Hugging Face commit revisions and publisher checksums, plus the VAD model
URL and SHA-256. `tools/download_models.py` verifies SHA-256 for large weights and Git blob
checksums for small configuration files, resumes partial downloads, and records local SHA-256
receipts. Re-run the download service to continue an interrupted download:

```sh
docker compose run --rm download
docker compose run --rm download python /opt/el-voice/download_models.py --verify
docker compose exec voice python /opt/el-voice/check_environment.py
```

The voice application verifies every model file against the local receipt before loading it.
The receipt pins the publisher URL and SHA-256 recorded by the download service, so a
replacement file from another source is refused.

The Silero TTS publisher URL has a versioned filename, but no independently verified
publisher SHA-256 was available when preparing this recipe. Its first download trusts the
official HTTPS source; the resulting local SHA-256 receipt detects subsequent changes. This
is not a publisher signature. Silero uses an executable PyTorch package/pickle: do not load
replacement `.pt` files from arbitrary sources. The preparation script only downloads bytes.

The VAD model URL is pinned to the upstream `v5.1.2` release tag and the file is pinned by
SHA-256 in `models.json`. It is a plain ONNX graph loaded by ONNX Runtime; no pickle is
involved. The image built before this change does not contain the new manifest entry, so
`models.json` and `tools/download_models.py` in the image are older than the repository. If
the VAD file has to be (re)downloaded, rebuild the image once with `docker compose build
voice` so the download service uses the updated manifest. The voice service itself does not
read `models.json`; it verifies the file against the receipt in `models/.download-receipts.json`.

Gemma has its own terms and Silero model licensing must be checked with the publisher; these
model files are not automatically covered by our code or a Python package's license.

## Streaming VAD choice

The pipeline uses **Silero VAD v5.1.2 streaming ONNX** loaded through the already-locked ONNX
Runtime. This was selected and characterised before the implementation:

- **Why it fits:** one small ONNX graph, CPU-only, 16 kHz, no new Python package, and a true
  streaming API (no buffering of a whole utterance). It is used only for continuous
  speech-onset detection and endpointing.
- **Streaming API:** exactly 512 samples (32 ms) of 16 kHz mono audio per call, plus a
  recurrent state tensor of shape `(2, 1, 128)`. The graph returns a speech probability and
  the next state. State is kept for the whole browser session and reset only when a session
  starts or stops, so silence/speech transitions are classified in context.
- **Measured CPU cost:** 0.084 ms per 32 ms frame in the development sandbox — about 0.3 % of
  one CPU core. This is cheap enough to run continuously while the assistant generates and
  speaks, which is what makes hands-free barge-in possible.
- **Not used:** Faster-Whisper's `vad_filter` is a post-hoc filter for supplied audio and
  cannot detect a speech onset or drive an interruption, so it is disabled for final
  transcription. Vosk partial recognition was not needed because the streaming VAD meets the
  interruption behaviour.

Thresholds and endpointing live in `workspace/settings.json` and are tuned against real
microphone input (the values below are the starting point, not a guarantee):

| Setting | Value | Meaning |
| --- | --- | --- |
| `vad.threshold` | 0.5 | speech probability that starts an utterance |
| `vad.neg_threshold` | 0.35 | probability below which a frame counts as silence |
| `vad.min_speech_ms` | 250 | shorter detections are rejected as noise/empty turns |
| `vad.end_silence_ms` | 550 | silence that finalises the utterance |
| `audio.pre_roll_ms` | 480 | audio kept before the onset so the first phoneme is not lost |
| `audio.max_utterance_ms` | 20000 | hard limit that finalises an unbounded utterance |

Raise `end_silence_ms` if slow, hesitant speech is cut off; lower it for snappier turn
taking. `pre_roll_ms` is included in the audio sent to Whisper.

## Architecture

```
Browser (Mac)                                   Container (Linux ARM64, CPU)
─────────────                                   ────────────────────────────
getUserMedia (echoCancellation: true)
  │
  ├─ AudioWorklet: 48 kHz → 16 kHz, 512-sample frames
  │        │ binary PCM over WebSocket (/ws) through Nginx :8765
  │        ▼
  │   VoiceSession (asyncio) ──► SpeechSegmenter ──► Silero VAD ONNX (per frame)
  │                                    │ onset: barge-in cancel of the active turn
  │                                    │ end:   finalised utterance
  │                                    ▼
  │                              Whisper small (CTranslate2, CPU thread)
  │                                    │ final Russian transcript
  │                                    ▼
  │                              Gemma 3 1B (llama-cpp-python, worker thread)
  │                                    │ streamed text → SpeakableChunker
  │                                    ▼
  │                              Silero b_lez (PyTorch CPU, serial executor)
  │                                    │ 24 kHz PCM chunks tagged with turn_id
  │                                    ▼
  └─ Web Audio playback queue ◄──── binary audio frames over WebSocket
```

Source layout (mounted at `/workspace`):

| File | Responsibility |
| --- | --- |
| `app.py` | FastAPI app, WebSocket endpoint, status/health, static files |
| `elvoice/settings.py` | settings load and cross-component validation |
| `elvoice/models.py` | receipt/SHA-256 verification of publisher assets |
| `elvoice/vad.py` | Silero ONNX streaming wrapper and endpointing segmenter |
| `elvoice/engines.py` | model lifecycle, readiness, STT/LLM/TTS inference |
| `elvoice/tts.py` | text sanitisation and speakable chunking |
| `elvoice/pipeline.py` | per-session turn state machine, barge-in, metrics |
| `static/` | browser UI, capture worklet, WebSocket client, playback queue |
| `tests/selftest.py` | end-to-end pipeline check (synthetic speech, real models) |

Models are loaded on a background thread so the HTTP process is alive immediately and can
report readiness while several hundred megabytes of weights load. Loading order is
PyTorch first, as documented in `tools/check_environment.py`, so its bundled OpenBLAS
registers before other native inference libraries.

## WebSocket protocol

One WebSocket connection carries one browser session. JSON control messages and binary audio
frames are mixed on the same socket.

Browser → service:

| Message | Meaning |
| --- | --- |
| `{"type":"start"}` | open a session; the service answers with `ready` or `error` |
| `{"type":"stop"}` | release the session, microphone and server resources |
| binary frame | 16 kHz mono signed 16-bit little-endian PCM, any chunking (the service re-frames to 512 samples) |
| `{"type":"playback_done","turn_id":N}` | the browser drained the playback queue for turn `N` |
| `{"type":"playback_stopped","turn_id":N,"elapsed_ms":X}` | browser-side time to stop playback after `flush` |
| `{"type":"ping"}` | keep-alive, answered with `pong` |

Service → browser:

| Message | Meaning |
| --- | --- |
| `{"type":"ready", ...}` | session accepted; sample rates, VAD settings, headphone notice |
| `{"type":"state","state":"listening"}` | `listening`, `processing`, `speaking`, `idle`, `reconnecting`, `error` |
| `{"type":"user_speech_start","turn_id":N}` | VAD onset detected |
| `{"type":"user_text","turn_id":N,"text":"..."}` | final local transcription (in-memory only) |
| `{"type":"assistant_text","turn_id":N,"delta":"..."}` | streamed answer text (display only) |
| `{"type":"turn_end","turn_id":N}` | no more audio will be sent for this turn |
| `{"type":"flush","turn_id":N,"reason":"barge_in"}` | stop and discard playback for turn `N` immediately |
| `{"type":"empty_turn","turn_id":N}` | detection rejected as noise or no speech recognised |
| `{"type":"timing","turn_id":N,"response_ms":X}` | speech end → first audio dispatch (content-free) |
| `{"type":"error","code":"...","message":"...","retryable":bool}` | session/turn error |
| binary audio frame | 16-byte header + 24 kHz mono signed 16-bit PCM |

Binary audio header (`struct "<4sIIHH"`, little-endian):

| Offset | Field |
| --- | --- |
| 0 | magic `EVA1` |
| 4 | `turn_id` (uint32) |
| 8 | chunk sequence within the turn (uint32) |
| 12 | sample rate (uint16, 24000) |
| 14 | reserved (uint16, 0) |
| 16 | PCM payload |

A browser discards audio whose `turn_id` does not match the turn it is currently playing, so
late chunks from a cancelled turn can never be heard.

## Barge-in, cancellation and turn isolation

Every user utterance gets a unique `turn_id` at VAD onset; the same id tags transcription,
generation, synthesis, browser events and audio chunks. A new speech onset, even while the
assistant is generating or playing, does all of the following without a button:

1. marks the active turn cancelled and sets its token;
2. sends `flush` to the browser, which stops scheduled audio sources and clears its queue;
3. cancels the awaiting turn task and lets the model threads observe the cancel token;
4. starts a new turn, keeping the pre-roll so the first word of the interruption is not lost.

Cancellation is bounded, not merely an asyncio cancel:

- **Language model:** generation runs in a dedicated worker thread and checks the cancel
  token between tokens. A barge-in therefore stops CPU generation after at most one token;
  no further tokens are scheduled. The cancelled turn's late text is dropped by `turn_id`.
- **Speech synthesis:** synthesis is serialised in one executor and one phrase is in flight
  at a time. A barge-in stops scheduling new phrases; the in-flight phrase finishes, is
  bounded by the chunk limit (160 characters), and its audio is discarded by the `turn_id`
  check. No unbounded synthesis continues after the interruption.
- **Recognition:** a stale transcription result is ignored when the turn was cancelled.
- **Browser:** queued audio is stopped and flushed immediately, and stale `turn_id` chunks
  are dropped on arrival.

Measured in the development-sandbox self-test: barge-in onset → `flush` dispatch 0.3 ms
(median), no stale audio chunks observed after a flush. Only server-side dispatch is
measured there; the audible stop latency on the target Mac is part of the browser acceptance
run (the UI reports the browser-side stop time too).

## Echo and headphones

The supported acoustic scenario is headphones. The browser requests
`echoCancellation: true`, `noiseSuppression: true` and `autoGainControl: true`, and the UI
shows the value reported by `MediaStreamTrack.getSettings().echoCancellation`. Browser echo
cancellation is best effort and does not guarantee that loudspeaker output is removed from
the microphone stream; therefore speaker playback is **not supported** in this MVP and the
page says so. With headphones, the assistant's own playback does not trigger a false
barge-in.

## Privacy

- Audio and transcripts are kept in memory only for the duration of a session; the service
  never writes them to `/data`, to the image or to logs.
- Application logs contain only operational data: model readiness, component failures
  (exception type), turn numbers and millisecond timings. Transcripts, model output and
  audio never appear. Uvicorn access logging is disabled.
- `/healthz` returns only process liveness; `/api/status` returns readiness, settings and
  content-free metrics.
- The service stays on the internal Compose network; only the loopback-bound Nginx proxy is
  published.

## Readiness, reconnects and errors

- `/healthz` answers as soon as the HTTP process is alive; it does **not** claim that the
  pipeline works.
- `/api/status` reports `pipeline_ready` and the status of each component (`vad`, `stt`,
  `llm`, `tts`), the active session state, and p50/p95 of the content-free timing metrics.
  The page keeps the Start button disabled until the pipeline is ready.
- Only one voice session is active at a time. A second browser gets a retryable `busy` error;
  a stale connection that is already closed is cleaned up and replaced.
- The browser reconnects the WebSocket with backoff without reloading the page and restarts
  the session; a stopped session releases the microphone, and a disconnect releases the
  server-side session, VAD state and any active turn.
- Empty recognition results and rejected noise detections return to listening without
  creating a turn. A turn failure is reported to the page and leaves the session usable.

## Run the checks

Syntax and static checks (development machine):

```sh
python -m compileall -q workspace
node --check workspace/static/app.js
node --check workspace/static/capture-worklet.js
```

End-to-end pipeline check inside the running container (synthesises its own Russian test
utterances with the configured `b_lez` voice, then drives the real WebSocket protocol):

```sh
docker compose exec voice python /workspace/tests/selftest.py --models /models
```

The check exercises one complete turn (VAD → Whisper → Gemma → Silero → tagged audio) and a
barge-in during playback, verifies that no stale audio is emitted after the flush, reads the
latency counters from `/api/status`, and confirms the process is still healthy. It writes no
audio or transcripts. A successful run prints `RESULT: PASS (12 checks)`.

## Measured baseline in the development sandbox

These numbers come from an ARM64 development sandbox with the downloaded models and the same
dependency versions, not from the target Mac/container and not from a browser. They are a
rough baseline only; the acceptance run must measure the real deployment.

| Measurement | Sandbox result |
| --- | --- |
| Streaming VAD CPU | 0.084 ms per 32 ms frame (~0.3 % of one core) |
| Whisper small final transcription | 2.5 s / 4.6 s for 7.1 s / 2.4 s utterances |
| LLM first audio (first sentence + first TTS phrase) | 2.1 s / 3.6 s |
| Speech end → first audio dispatch (`response_ms`) | p50 8.0 s, p95 10.1 s (2 turns, shared CPU) |
| Barge-in onset → flush dispatch | 0.3 ms median |
| End-to-end self-test | `RESULT: PASS (12 checks)` |

Do not turn these into acceptance targets. After the browser run, `/api/status` reports p50
and p95 for speech-end → first audible audio and for barge-in dispatch; agree targets from
that baseline, as required by SPEC-001.

## Known limitations

- Speaker playback is unsupported; browser echo cancellation is best effort, headphones are
  required.
- Cancellation bounds rather than pre-empts a single in-flight TTS phrase (at most the
  current speakable chunk, limited to 160 characters), because PyTorch CPU synthesis cannot
  be interrupted mid-call. The language model stops within one token.
- One active session at a time; no conversation history or persistent state. Each turn sends
  one final utterance to the model.
- Silero v4 Cyrillic does not place stress automatically and Russian pronunciation needs
  listening tests; Whisper small is used for accuracy/latency balance.
- The image built before this change contains the older `models.json` and download tools;
  rebuilding is only needed to download the VAD asset again, not to run the application.

## Verification status

Implemented and checked in development:

- [x] Streaming Silero VAD selected, pinned, documented; CPU cost measured.
- [x] Continuous browser capture, WebSocket protocol, start/stop and reconnect handling.
- [x] VAD onset/endpointing with pre-roll, noise rejection, final Whisper transcription.
- [x] Local Gemma streaming with incremental Silero `b_lez` synthesis/playback.
- [x] `turn_id` isolation, barge-in flush, bounded cancellation, stale-result rejection.
- [x] Echo policy, header/security hardening, readiness, content-free metrics.
- [x] End-to-end self-test passes in a development sandbox; HTTP smoke test passes.

Not yet observed (requires the real container and a browser on the Mac, as SPEC-001 requires):

- [ ] Complete conversation through `http://localhost:8765` with a real microphone.
- [ ] Barge-in observed in the browser during generation and during playback.
- [ ] No false self-interruption with headphones.
- [ ] p50/p95 latency measured on the target Mac; targets agreed from that baseline.
- [ ] Disconnect/reconnect and stop/restart scenarios in the browser.
- [ ] Storage and log inspection after a session proving no audio/transcript persistence.

## Publisher references

- https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF
- https://huggingface.co/Systran/faster-whisper-small
- https://github.com/snakers4/silero-models
- https://github.com/snakers4/silero-vad
- https://github.com/SYSTRAN/faster-whisper
- https://github.com/abetlen/llama-cpp-python
- https://pytorch.org/get-started/locally/
- https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints/echoCancellation

Silero v4 Cyrillic is covered by the publisher's CC BY-NC-SA 4.0 license, including its
noncommercial restriction. See the [publisher documentation](https://github.com/snakers4/silero-models#cyrillic-languages-v4)
and [license](https://github.com/snakers4/silero-models/blob/master/LICENSE). Silero VAD is
MIT-licensed. Gemma is covered by the Gemma Terms of Use.

Background references and locally saved source material are kept in `references/`.
See [`references/REFERENCES.md`](references/REFERENCES.md) for the article link.

## Continue development in the running container

```sh
docker compose exec voice sh
docker compose exec voice python /opt/el-voice/check_environment.py --skip-models
docker compose logs --tail=100 voice
docker compose restart voice
```

Edit `workspace/` on the Mac; changes appear immediately in `/workspace`. Python packages
are installed in the image's virtual environment, not in the source mount.

## Stop

```sh
docker compose stop voice
docker compose down
```

These commands preserve the bind-mounted model and data files. Do not delete `models/`
unless you intend to download the weights again.
