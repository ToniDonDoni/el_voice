# EL Voice

An independent voice assistant for Docker Desktop on Apple Silicon with two explicit launch
modes:

- **Local Voice** — fully offline. A browser on the Mac captures the microphone; streaming
  VAD, Whisper transcription and Silero `b_lez` synthesis run in the voice container, and
  Gemma runs in a separate internal LLM service. No cloud inference, API keys or runtime
  downloads are used.
- **Remote Voice** — the same local audio capture, STT and TTS, but generation goes through
  LangChain to the configured OpenCode Go model (`deepseek-v4.1-flash` by default) and a
  real DuckDuckGo `web_search` tool. Remote mode sends the bounded prior dialogue and
  search result titles, URLs and snippets back to the provider; microphone audio never
  leaves the Mac.

Both modes call the same internal LLM service over one streaming/cancellation contract, so
the voice application never loads Gemma and never contacts OpenCode Go directly. The
specification is [`tasks/SPEC-002-REMOTE-VOICE-LLM-SERVICE.md`](tasks/SPEC-002-REMOTE-VOICE-LLM-SERVICE.md);
the criterion-by-criterion evidence is in
[`tasks/SPEC-002-REMOTE-VOICE-LLM-SERVICE-IMPLEMENTATION-REPORT.md`](tasks/SPEC-002-REMOTE-VOICE-LLM-SERVICE-IMPLEMENTATION-REPORT.md).

The acceptance criteria in both specifications still require a run in the real container
with a real browser and microphone; live Remote Voice also requires the user's OpenCode Go
key. This development environment has no Docker daemon, browser, microphone or provider
key, so the "Verification status" section records exactly what has and has not been
observed. Live DuckDuckGo search and the deterministic service/agent contracts were
exercised here; the live provider and browser were not.

No source code, installer, infrastructure package, binary build, or model mirror from the
previously reviewed assistant author is used here.

## What the image installs

| Component | Source / purpose |
| --- | --- |
| Python 3.12 on Debian Bookworm | Official Docker image, pinned by digest |
| FFmpeg, OpenMP, libsndfile, curl, CA certificates | Debian repositories; audio decoding and CPU inference |
| FastAPI, Uvicorn, WebSockets | Browser/API layer and WebSocket audio transport |
| Faster-Whisper, CTranslate2, PyAV, ONNX Runtime | Local speech recognition, streaming VAD and audio processing |
| llama-cpp-python | Official upstream Linux ARM64 CPU wheel pinned by SHA-256; local LLM service only |
| PyTorch CPU | Official PyTorch Linux ARM64 wheel; no CUDA dependencies |
| LangChain Core, LangChain OpenAI, OpenAI SDK | Remote LangChain agent and OpenCode Go client |
| `ddgs` | DuckDuckGo search backend for the typed `web_search` tool |
| NumPy, Hugging Face Hub and resolved dependencies | Supporting libraries |

Direct inputs are in `requirements.in`. Python runtime dependencies and artifact hashes are
resolved in `requirements.lock` for CPython 3.12 / Linux ARM64. SPEC-002 adds direct pins
for `langchain-core==1.7.4`, `langchain-openai==1.7.3`, `openai==3.45.0` and `ddgs==9.16.1`;
the lock was regenerated with `sh tools/lock.sh` (uv 0.11.14). **These new packages require
one image rebuild** (`docker compose ... build voice`); after that, mounted `workspace/`,
`llm_service/` and proxy source edits only need a restart. Debian packages are not a complete
snapshot, so this is not a byte-reproducible image. Changing the lock requires `uv` on the
development machine and `sh tools/lock.sh`.

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
| `workspace/` | `/workspace` | Read-only application source; edit on the Mac |
| `llm_service/` | `/opt/llm-service/llm_service` | Read-only internal LLM service source; edit on the Mac |
| `models/` | `/models` | Read-only in voice and local LLM; writable in the download service |
| `data/` | `/data` | Writable cache and future optional application state |

`models/` is not mounted into the remote LLM service. The repository root, Git metadata,
home directory, Docker socket and `.env` are not mounted. Only the Dockerfile, `llm_service/`
and the small bootstrap tools enter the image build context. Models do not inflate image
layers and survive container recreation.

## Build and start

Give Docker Desktop approximately 12 GB memory and at least 15 GB free disk space. The
voice runtime service is limited to 10 GB and 6 CPUs. The local LLM service is limited to
8 GB; the remote LLM service needs much less. The first image build downloads the Python
dependencies; it no longer compiles llama.cpp. No AMD64 emulation or GPU access is used.
Nginx runs from a separate official unprivileged image; Compose pulls it on first start, so
adding the proxy does not rebuild `el-voice:dev`.

Run commands from the repository root. `compose.yaml` is the shared base; the mode files
add the internal `llm` service and point the voice service at it.

### Local Voice (offline, no keys)

```sh
sh tools/init.sh
docker compose config --quiet
docker compose -f compose.yaml -f compose.local.yaml build voice
docker compose -f compose.yaml -f compose.local.yaml up -d
docker compose -f compose.yaml -f compose.local.yaml ps
curl --fail http://localhost:8765/healthz
curl --fail http://localhost:8765/api/status
```

Local Voice mounts `workspace/`, `llm_service/` and `models/` into the private `llm`
service. Gemma loads there rather than in the voice container. The `private` network is
internal: neither the voice service nor the local LLM service has an outbound route.

### Remote Voice (OpenCode Go + DuckDuckGo)

Put the key only in the local, Git-ignored `.env` (never in Compose YAML or a commit):

```sh
cat >> .env <<'EOF'
OPENCODE_GO_API_KEY=
OPENCODE_GO_BASE_URL=https://opencode.ai/zen/go/v1
OPENCODE_GO_MODEL=deepseek-v4.1-flash
EOF
# Edit .env and fill the key, then:
docker compose -f compose.yaml -f compose.remote.yaml build voice
docker compose -f compose.yaml -f compose.remote.yaml up -d
docker compose -f compose.yaml -f compose.remote.yaml ps
curl --fail http://localhost:8765/healthz
curl --fail http://localhost:8765/api/status
```

Remote Voice does not mount `models/` into the LLM service and does not load Gemma. The
`llm` service joins `private` (to answer the voice service) and `provider` (outbound HTTPS
only); the voice, proxy and browser containers never receive `OPENCODE_GO_API_KEY`.
`OPENCODE_GO_BASE_URL` defaults to the requested `https://opencode.ai/zen/go/v1`. The
provider documentation also lists `https://opencode.ai/inference/go/openai/v1`; neither
endpoint has been verified with a real account here. Override the variable rather than
silently changing code. OpenCode Go also asks agent clients to send a stable
`x-opencode-session` header; the service sends one per process and allows
`OPENCODE_GO_SESSION_ID` to override it. This header requirement was confirmed against the
unauthenticated endpoint and provider documentation, not by a successful live completion.

`/api/status` reports `llm.mode` and readiness. A missing key leaves the service unready
with an actionable, redacted error and disables the Start button; there is no fallback to
Gemma or to another provider.

### Switch modes and stop

Always stop the previous graph before starting the other mode; the browser port is the same
and the two graphs must not run concurrently.

```sh
# Local -> Remote
docker compose -f compose.yaml -f compose.local.yaml down
docker compose -f compose.yaml -f compose.remote.yaml up -d

# Remote -> Local
docker compose -f compose.yaml -f compose.remote.yaml down
docker compose -f compose.yaml -f compose.local.yaml up -d

# Stop the active mode (keeps models/ and data/)
docker compose -f compose.yaml -f compose.local.yaml stop
docker compose -f compose.yaml -f compose.local.yaml down
# or the same commands with -f compose.remote.yaml
```

Open **http://localhost:8765** in a browser and press **Start conversation**. The page shows
a **Local mode** or **Remote mode** badge. Remote mode also shows the disclosure that
transcribed text, bounded prior dialogue and search result titles/URLs/snippets leave the
Mac in Remote mode. `/api/status` returns
`pipeline_ready: true` only when the local speech models and the LLM service are both ready.

Microphone capture is resampled to exactly 16 kHz inside the AudioWorklet with a
two-stage 4 kHz low-pass filter and fractional linear interpolation
(`workspace/static/resampler.js`), then emitted as 512-sample frames. This avoids the
44.1 kHz aliasing of a naive decimator and keeps 32/48/96 kHz devices working; verified by
`workspace/tests/resampler.test.mjs` and `capture-worklet.test.mjs`.

The only published mapping is `127.0.0.1:8765 -> proxy:8080`; Nginx forwards HTTP and
WebSocket requests to `voice:8000` over the internal network. The voice service, the LLM
service and their ports are never published. Nginx joins a separate bridge network for host
port publishing; that bridge permits potential outbound access from the Nginx container.
Nginx's fixed upstream config forwards requests only to `voice:8000`. The download service
uses egress to fetch models, while `voice` remains isolated on its internal network.

Application source changes are visible immediately in `/workspace` (and in the LLM service's
`/opt/llm-service/llm_service` mount) and only need a restart, not a rebuild. The exception
is a change to the model manifest, `tools/`, `requirements.*` or the Dockerfile: those are
baked into the image and need `docker compose ... build voice`.

## Models and downloaded assets

| Model | Publisher | Approximate size |
| --- | --- | --- |
| Gemma 3 1B IT Q4_K_M GGUF | ggml-org on Hugging Face | 806 MB |
| Whisper small, CTranslate2 format | Systran on Hugging Face | 464 MB including tokenizer/config |
| Silero v4 Cyrillic, selected voice `b_lez` (male Lezghian) | models.silero.ai | 34 MB |
| Silero VAD v5.1.2 streaming ONNX | snakers4/silero-vad (MIT) | 2.3 MB |

Local Voice needs all four models. Remote Voice needs Whisper, Silero TTS and VAD for local
speech processing, but **not Gemma**: the remote LLM service does not mount `models/`.

`models.json` pins Hugging Face commit revisions and publisher checksums, plus the VAD model
URL and SHA-256. `tools/download_models.py` verifies SHA-256 for large weights and Git blob
checksums for small configuration files, resumes partial downloads, and records local SHA-256
receipts. Re-run the download service to continue an interrupted download:

```sh
docker compose run --rm download
docker compose run --rm download python /opt/el-voice/download_models.py --verify
docker compose -f compose.yaml -f compose.local.yaml exec voice \
  python /opt/el-voice/check_environment.py
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
the VAD file has to be (re)downloaded, rebuild the image once with
`docker compose -f compose.yaml -f compose.local.yaml build voice` so the download service
uses the updated manifest. The voice service itself does not
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
- **Wrapper parity:** the adapter follows the official v5.1.2 `OnnxWrapper` streaming
  semantics: every 512-sample frame is prefixed with the previous 64 samples of context
  (576-sample ONNX input), and recurrent state and context are reset together. This is
  verified against a reference implementation in `workspace/tests/test_vad_parity.py`.
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
| `conversation.history_turns` | 4 | previous question/answer pairs kept in the prompt |
| `conversation.max_prompt_chars` | 4000 | character budget for the whole prompt |
| `audio.pre_roll_ms` | 480 | audio kept before the onset so the first phoneme is not lost |
| `audio.max_utterance_ms` | 20000 | hard limit that finalises an unbounded utterance |

Raise `end_silence_ms` if slow, hesitant speech is cut off; lower it for snappier turn
taking. `pre_roll_ms` is included in the audio sent to Whisper.

## Architecture

The voice container owns capture-facing audio work and the browser; the internal LLM
service owns prompt construction, conversation history, generation and (in Remote mode)
tool execution. Both modes use the same boundary.

```
Browser (Mac)                                   Container (Linux ARM64, CPU)
─────────────                                   ────────────────────────────
getUserMedia (echoCancellation: true)
  │
  ├─ AudioWorklet: 48 kHz → 16 kHz, 512-sample frames
  │        │ binary PCM over WebSocket (/ws) through Nginx :8765
  │        ▼
  │   voice service (private network, no provider key)
  │     ├── SpeechSegmenter ──► Silero VAD ONNX (per frame)
  │     │        │ onset: barge-in cancel of the active turn
  │     │        │ end:   finalised utterance
  │     ├── Whisper small (CTranslate2, CPU thread)
  │     ├── Silero b_lez (PyTorch CPU, serial executor)
  │     └── elvoice/llm_client.py
  │              │ WebSocket /generate (internal only)
  │              ▼
  │   LLM service:8100 (no host port)
  │     ├── server owns bounded ConversationHistory per connection
  │     ├── LocalBackend: Gemma 3 1B GGUF (llama-cpp-python, worker thread)
  │     └── RemoteBackend: LangChain ChatOpenAI → OpenCode Go
  │              └── web_search tool → ddgs (bounded, typed query only)
  │
  └─ Web Audio playback queue ◄──── binary audio frames over WebSocket
```

Local mode uses only the `private` network. Remote mode adds the `provider` egress network
to the LLM service only; the voice and browser path is unchanged, there is no Gemma mount
and no local remote-model weights.

Source layout (mounted at `/workspace`, server package at `/opt/llm-service`):

| File | Responsibility |
| --- | --- |
| `workspace/app.py` | FastAPI app, WebSocket endpoint, status/health, static files |
| `workspace/elvoice/settings.py` | settings load/validation and internal LLM service URL |
| `workspace/elvoice/models.py` | receipt/SHA-256 verification of publisher assets |
| `workspace/elvoice/vad.py` | Silero ONNX streaming wrapper and endpointing segmenter |
| `workspace/elvoice/engines.py` | VAD/STT/TTS lifecycle and inference (no LLM loading) |
| `workspace/elvoice/tts.py` | text sanitisation and speakable chunking |
| `workspace/elvoice/llm_client.py` | streaming/cancel WebSocket client for the LLM service |
| `workspace/elvoice/pipeline.py` | per-session turn state machine, barge-in, metrics |
| `llm_service/config.py` | mode/host/port/limits/provider env configuration |
| `llm_service/server.py` | `/healthz`, `/status`, `/generate` and per-connection history |
| `llm_service/backends/local.py` | Gemma streaming with per-token cancellation |
| `llm_service/backends/remote.py` | bounded LangChain tool loop and redacted errors |
| `llm_service/backends/search.py` | typed `web_search`, `ddgs`, URL/size bounds |
| `workspace/static/` | browser UI, capture worklet, WebSocket client, playback queue |
| `workspace/tests/selftest.py` | end-to-end local check with both services in-process |
| `tests/test_llm_service.py` | real HTTP/WS service contract with a fake backend |
| `tests/test_remote_backend.py` | deterministic remote agent/tool/failure checks |
| `tests/test_search_live.py` | mocked URL/bounds checks plus optional live `ddgs` |

Conversation context is owned by the internal LLM service, per WebSocket connection, and kept
in memory. `settings.json` `conversation.history_turns` (default 4) completed question/answer
pairs are prepended to every prompt and trimmed to `conversation.max_prompt_chars`; whole
pairs are dropped oldest first, cancelled turns are never added, and the history is cleared
when the connection stops or the client sends `reset`. The voice pipeline sends only the
current final utterance. `/api/status` exposes only the limits and the last prompt's
pair/character counts, never transcript content.

Models are loaded on a background thread so the HTTP process is alive immediately and can
report readiness while several hundred megabytes of weights load. Loading order is
PyTorch first, as documented in `tools/check_environment.py`, so its bundled OpenBLAS
registers before other native inference libraries.

## Internal LLM service contract

The voice application opens one WebSocket to `EL_VOICE_LLM_URL` (default
`ws://llm:8100/generate`) per browser session. One connection is one conversation and the
service allows one active generation on it.

Voice → LLM service:

| Message | Meaning |
| --- | --- |
| `{"type":"generate","turn_id":N,"text":"..."}` | start a turn; only the current final utterance is sent |
| `{"type":"cancel","turn_id":N}` | cancel that turn; the service sets its token, aborts the backend iterator and replies `cancelled` |
| `{"type":"reset"}` | cancel any active generation and clear this connection's history |

LLM service → voice:

| Event | Meaning |
| --- | --- |
| `{"type":"start","turn_id":N,"prompt":{"history_turns":n,"chars":c}}` | prompt built; content-free statistics |
| `{"type":"delta","turn_id":N,"text":"..."}` | answer text only; spoken by TTS |
| `{"type":"tool","turn_id":N,"name":"web_search","query":"..."}` | a typed tool call is about to run |
| `{"type":"sources","turn_id":N,"sources":[{"title","url","snippet"}]}` | bounded results for display only |
| `{"type":"done","turn_id":N,"chars":c}` | answer complete and added to this connection's history |
| `{"type":"error","turn_id":N,"code":"...","message":"...","retryable":bool}` | actionable, redacted failure; no fallback answer |
| `{"type":"cancelled","turn_id":N}` | generation stopped; no answer is added to history |

`/healthz` is process liveness. `/status` reports `mode`, `ready`, `error`, context limits,
search settings, provider base URL/model (never the key), active sessions and content-free
last-prompt statistics. Backend readiness means "configured/loaded"; for Remote mode it does
not prove the account or endpoint works until a real turn succeeds.

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
| `{"type":"assistant_text","turn_id":N,"delta":"..."}` | streamed answer text (display and TTS input) |
| `{"type":"assistant_tool","turn_id":N,"name":"...","query":"..."}` | tool/search indicator; never spoken |
| `{"type":"assistant_sources","turn_id":N,"sources":[...]}` | clickable source links; never spoken |
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

- **Language model:** the voice client sends `cancel(turn_id)` over the internal boundary.
  The service stops its generation loop, aborts the backend iterator and replies `cancelled`.
  In Local mode Gemma runs in a worker thread that checks the shared token between tokens, so
  CPU generation stops after at most one token. In Remote mode cancelling the agent loop
  closes the local HTTP stream and drops the turn; **provider-side cancellation is not
  verified** (no live key here) and is not claimed. A cancelled turn is never added to
  history.
- **Speech synthesis:** synthesis is serialised in one executor and one phrase is in flight
  at a time. A barge-in stops scheduling new phrases; the in-flight phrase finishes, is
  bounded by the chunk limit (160 characters), and its audio is discarded by the `turn_id`
  check. No unbounded synthesis continues after the interruption.
- **Recognition:** a stale transcription result is ignored when the turn was cancelled.
- **Browser:** queued audio is stopped and flushed immediately, and stale `turn_id` chunks
  are dropped on arrival.

Measured in the development-sandbox self-test: barge-in onset → `flush` dispatch 0.9 ms
(median), no stale audio chunks observed after a flush. Only server-side dispatch is
measured there; the audible stop latency on the target Mac is part of the browser acceptance
run (the UI reports the browser-side stop time too).

## Remote data disclosure and search

Local mode keeps the original offline privacy boundary: no keys and no egress. Remote mode is
an explicit, opt-in extension. When the badge says **Remote mode**, the page displays that:

- final transcribed text and bounded conversation context are sent to the configured
  OpenCode Go endpoint;
- when the model calls `web_search`, the search query is sent to DuckDuckGo;
- microphone audio is still processed locally and never sent to either service;
- search results are untrusted data. The tool accepts only a short query, never a URL or
  shell command, and the service does not fetch model-generated URLs. Results are bounded in
  count, snippet size and execution time; private, loopback, link-local and metadata URLs are
  removed before display;
- only the LLM service container receives `OPENCODE_GO_API_KEY`, and only it joins the
  `provider` egress network. The key is never returned by `/status`, written to logs or
  included in errors; provider error detail is truncated and redacted.

Search is invoked when the user asks for current facts or the model decides it needs them;
ordinary Dota conversation need not search. The configured backend is `ddgs`; replacing it
requires code and a new dependency pin. `EL_VOICE_SEARCH_PROVIDER` rejects unsupported
values rather than silently falling back.

## Interruption toggle

The page has an **Interrupt with voice** checkbox, enabled by default. Leave it on for
normal hands-free barge-in. Turn it off in noisy places: speech during generation or
playback is then captured and discarded, and the current answer is allowed to finish.
The preference is remembered in the browser (`localStorage`) and enforced by the server
through the WebSocket `config` message, so it applies to both Local and Remote modes.

## Echo and headphones

The supported acoustic scenario is headphones. The browser requests
`echoCancellation: true`, `noiseSuppression: true` and `autoGainControl: true`, and the UI
shows the value reported by `MediaStreamTrack.getSettings().echoCancellation`. Browser echo
cancellation is best effort and does not guarantee that loudspeaker output is removed from
the microphone stream; therefore speaker playback is **not supported** in this MVP and the
page says so. With headphones, the assistant's own playback does not trigger a false
barge-in.

## Privacy

- Audio and transcripts are kept in memory only for the duration of a session; the voice and
  LLM services never write them to `/data`, to the image or to logs.
- Local mode: no API keys, no outbound inference, no search. This remains the offline
  deliverable from SPEC-001.
- Remote mode: only text/search leaves the Mac, as disclosed in the UI and above. There is
  no persistent agent memory, database or external tracing. Provider keys live only in the
  remote `llm` container's environment.
- Application logs contain only operational data: model readiness, component failures
  (exception type), turn numbers and millisecond timings. Transcripts, model output, prompts,
  search queries and audio never appear. Uvicorn access logging is disabled. The LLM service
  logs error codes and exception types, not provider payloads.
- `/healthz` returns only process liveness; `/api/status` returns readiness, settings and
  content-free metrics (including the LLM mode and context limits).
- The voice service and the LLM service stay on the internal Compose network; only the
  loopback-bound Nginx proxy is published. The remote LLM service additionally joins the
  egress network; no other container does.

## Readiness, reconnects and errors

- `/healthz` answers as soon as the HTTP process is alive; it does **not** claim that the
  pipeline works.
- If the WebSocket closes unexpectedly, the browser stops every scheduled audio source,
  clears the current turn and reconnects with backoff; the server cancels the active turn
  and releases the session, so old speech never continues across a reconnect.
- `/api/status` reports `pipeline_ready` and the status of each component (`vad`, `stt`,
  `llm`, `tts`), the active session state, `llm.mode`/readiness and p50/p95 of the
  content-free timing metrics. The page keeps the Start button disabled until both the local
  speech models and the LLM service are ready.
- If the LLM service is unreachable or unready (for example, a Remote mode key is missing),
  the voice session start returns a retryable `llm_not_ready` error with the service's
  redacted message. No local or alternate-provider answer is substituted.
- Only one voice session is active at a time. A second browser gets a retryable `busy` error;
  a stale connection that is already closed is cleaned up and replaced.
- The browser reconnects the WebSocket with backoff without reloading the page and restarts
  the session; a stopped session releases the microphone, and a disconnect releases the
  server-side session, VAD state and any active turn.
- Empty recognition results and rejected noise detections return to listening without
  creating a turn. A turn failure is reported to the page and leaves the session usable.

## Run the checks

All checks are plain Python/node scripts (no `pytest` requirement). Run them from the
repository root with the development venv and the downloaded models.

```sh
python -m compileall -q workspace llm_service tests
node --check workspace/static/app.js
node --check workspace/static/capture-worklet.js
node --check workspace/static/playback.js
node --check workspace/static/resampler.js

EL_VOICE_MODELS=$PWD/models python workspace/tests/test_contracts.py
EL_VOICE_MODELS=$PWD/models python workspace/tests/test_pipeline_context.py
EL_VOICE_MODELS=$PWD/models python workspace/tests/test_vad_parity.py
EL_VOICE_MODELS=$PWD/models python tools/test_tts_integrity.py

EL_VOICE_MODELS=$PWD/models python tests/test_llm_service.py
EL_VOICE_MODELS=$PWD/models python tests/test_remote_backend.py
python tests/test_search_live.py            # mocked URL/bounds checks
python tests/test_search_live.py --live     # optional outbound DuckDuckGo check

for test in workspace/tests/*.test.mjs; do node "$test"; done
```

The service contract test starts the real FastAPI app on a random loopback port with a
deterministic fake backend; the remote test injects a LangChain-style fake chat model and a
fake search function. Neither needs a provider key or network. The live search check is
separate and skipped unless requested.

End-to-end local pipeline check inside the running container. It starts the real internal
LLM service (local Gemma backend) and the real voice app in one process, synthesises its own
Russian test utterances with the configured `b_lez` voice, and drives both real WebSocket
protocols:

```sh
docker compose -f compose.yaml -f compose.local.yaml exec voice \
  python /workspace/tests/selftest.py --models /models
```

The check exercises VAD → Whisper → the LLM-service boundary → Gemma → Silero → tagged audio,
a barge-in during playback, no stale audio after the flush, a follow-up turn, reconnect/turn
ids, content-free status, and local-mode reporting. It does not persist audio or transcripts
and prints only lengths and status. A
successful run in the development sandbox prints `RESULT: PASS (21 checks)`.

## Measured baseline in the development sandbox

These numbers come from an ARM64 development sandbox with the downloaded models and the same
dependency versions, not from the target Mac/container and not from a browser. They are a
rough baseline only; the acceptance run must measure the real deployment.

| Measurement | Sandbox result |
| --- | --- |
| Streaming VAD CPU | 0.084 ms per 32 ms frame (~0.3 % of one core) |
| Whisper small final transcription | 1.8–3.3 s for 4.2–4.4 s utterances in the two-service run |
| Local Gemma first token after STT | ~0.9 s |
| Speech end → first audio dispatch (`response_ms`) | p50 3.7 s, p95 11.3 s (4 turns, shared CPU) |
| Barge-in onset → flush dispatch | 0.9 ms median (1 measured interruption) |
| End-to-end local self-test | `RESULT: PASS (21 checks)` |
| Live DuckDuckGo search (`ddgs`) from the sandbox | 3 results in 4.8 s; one later run took 17.4 s during transient throttling |

Remote provider latency, remote streaming and remote first-audio timing are **unmeasured**:
there is no OpenCode Go key in this environment. Do not turn these numbers into acceptance
targets. After the browser run, `/api/status` reports p50 and p95 for speech-end → first
audible audio and for barge-in dispatch; agree targets from that baseline, as required by
SPEC-001.

## Known limitations

- Speaker playback is unsupported; browser echo cancellation is best effort, headphones are
  required.
- Cancellation bounds rather than pre-empts a single in-flight TTS phrase (at most the
  current speakable chunk, limited to 160 characters), because PyTorch CPU synthesis cannot
  be interrupted mid-call. Local Gemma stops within one token; remote provider-side
  cancellation is unverified.
- One active voice session at a time. Conversation history is bounded per LLM-service
  connection and is not persisted across sessions.
- **Live Remote Voice is unverified.** No OpenCode Go key exists in this environment, so the
  configured endpoint, model access, streaming, tool calling and account-specific privacy
  settings have not been exercised. Mocked checks do not complete live acceptance.
- The provider documentation lists `https://opencode.ai/inference/go/openai/v1` while the
  requested default is `https://opencode.ai/zen/go/v1`. Both are configurable; neither was
  verified with a real completion. OpenCode Go also requires/asks for an
  `x-opencode-session` header on agent traffic; the service sends one and exposes
  `OPENCODE_GO_SESSION_ID` for a stable override.
- Live DuckDuckGo search works from this sandbox, but search availability/rate limits from
  the target Mac/container network are not guaranteed. The service reports search failure
  and empty results honestly instead of fabricating an answer.
- Remote search result quality and the model's decision to search are not tuned against the
  live model; only deterministic fakes cover the tool loop.
- Browser acceptance (mode badge, clickable sources, remote disclosure, barge-in, latency)
  has not been run: this environment has no browser or microphone. Docker Compose
  configuration was validated statically; the Docker daemon and image build were unavailable.
- Silero v4 Cyrillic does not place stress automatically and Russian pronunciation needs
  listening tests; Whisper small is used for accuracy/latency balance.

## Verification status

Implemented and checked in development (SPEC-001 local regression):

- [x] Streaming Silero VAD selected, pinned, documented; CPU cost measured.
- [x] Continuous browser capture, WebSocket protocol, start/stop and reconnect handling.
- [x] VAD onset/endpointing with pre-roll, noise rejection, final Whisper transcription.
- [x] Local Gemma streaming through the internal LLM service with incremental Silero `b_lez`
  synthesis/playback.
- [x] `turn_id` isolation, barge-in flush, bounded cancellation, stale-result rejection.
- [x] Echo policy, header/security hardening, readiness, content-free metrics.
- [x] End-to-end local self-test passes in a development sandbox with both services;
  HTTP smoke test passes.

Implemented and checked in development (SPEC-002):

- [x] Separate internal LLM service with local and remote backends and one event contract.
- [x] Voice app no longer loads Gemma or contacts OpenCode Go; it sends current text only.
- [x] Server-owned bounded context, session isolation, reset, cancellation and error recovery
  proved at the real HTTP/WebSocket boundary with deterministic backends.
- [x] Remote LangChain tool loop, typed `web_search`, `ToolMessage` pairing, source events,
  iteration/result/snippet/time bounds and redacted actionable failures proved with
  deterministic fakes. Tool arguments and control fields are never emitted as delta text; a
  one-chunk lookahead is discarded when a tool call is detected. A provider that streams
  prose in the same assistant message before its tool arguments could still surface that
  prose, which live acceptance must check.
- [x] Live DuckDuckGo search returns bounded public results from the sandbox.
- [x] `compose.local.yaml` and `compose.remote.yaml` validate; local has no egress, remote
  gives only the LLM service provider egress and the key.
- [x] New dependencies pinned and `requirements.lock` regenerated; image rebuild documented.
- [x] Local mode reports `llm.mode=local`, readiness through `/api/status`, and search
  disabled; remote mode without a key reports an actionable error and no fallback.

Not yet observed (requires the real container, a browser and/or a provider key):

- [ ] Complete conversation through `http://localhost:8765` with a real microphone.
- [ ] Mode badge, remote disclosure and clickable sources rendered in the browser.
- [ ] Barge-in observed in the browser during generation and during playback.
- [ ] No false self-interruption with headphones.
- [ ] p50/p95 latency measured on the target Mac; targets agreed from that baseline.
- [ ] Disconnect/reconnect and stop/restart scenarios in the browser.
- [ ] Storage and log inspection after a session proving no audio/transcript persistence.
- [ ] Live OpenCode Go streaming/tool calling with the user's key, correct endpoint/model,
  and measured remote latency.
- [ ] Docker image build/run and both Compose graphs on the target Mac.

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

Use the same `-f compose.yaml -f compose.<mode>.yaml` pair as the active mode:

```sh
docker compose -f compose.yaml -f compose.local.yaml exec voice sh
docker compose -f compose.yaml -f compose.local.yaml exec voice \
  python /opt/el-voice/check_environment.py --skip-models
docker compose -f compose.yaml -f compose.local.yaml logs --tail=100 voice llm
docker compose -f compose.yaml -f compose.local.yaml restart voice llm
```

Edit `workspace/` or `llm_service/` on the Mac; changes appear immediately in `/workspace`
and `/opt/llm-service/llm_service`. Python packages are installed in the image's virtual
environment, not in the source mount, so dependency changes need a rebuild.

## Stop

```sh
# Active mode only (containers stay defined, files are preserved)
docker compose -f compose.yaml -f compose.local.yaml stop
docker compose -f compose.yaml -f compose.remote.yaml stop

# Remove the active graph; keeps bind-mounted models/ and data/
docker compose -f compose.yaml -f compose.local.yaml down
docker compose -f compose.yaml -f compose.remote.yaml down
```

These commands preserve the bind-mounted model and data files. Do not delete `models/`
unless you intend to download the weights again.
