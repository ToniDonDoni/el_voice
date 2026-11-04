# EL Voice

An independent Dota voice assistant for Docker Desktop on Apple Silicon. The browser captures microphone audio and plays responses; Linux ARM64 CPU containers run streaming VAD, Whisper transcription, LLM response generation, and Silero `b_lez` speech synthesis. Choose Local Voice for offline Gemma inference or Remote Voice for OpenCode Go and web search. No code or binary artifacts from the referenced assistant project are used.

## Choose a voice mode

Both modes use the same browser, local microphone capture, Whisper STT, and Silero TTS. Local mode runs Gemma in the internal LLM service and needs no provider key or internet access at runtime. Remote mode sends recognized text, bounded conversation context, and search queries/results to OpenCode Go and DuckDuckGo; microphone audio and speech synthesis stay local. The provider key is passed only to the Remote LLM service. Its health check confirms configuration, not successful live inference.

Prepare the required voice models for either mode. Run `sh tools/init.sh` first. Remote Voice deliberately downloads only Whisper and Silero, so this preparation does not need a provider key or the Gemma model. Add your real `OPENCODE_GO_API_KEY` to the ignored local `.env` file before starting Remote Voice; do not put it in a shell command or commit it.

```sh
sh tools/init.sh
docker compose -f compose.yaml -f compose.remote.yaml run --rm download
```

To use Local Voice instead, download all configured models:

```sh
docker compose -f compose.yaml -f compose.local.yaml run --rm download
```

Start exactly one mode at a time. The user-facing browser endpoint remains `http://localhost:8765`:

```sh
# Local Voice
docker compose -f compose.yaml -f compose.local.yaml up -d --build voice llm proxy

# Stop Local Voice before switching
docker compose -f compose.yaml -f compose.local.yaml down

# Remote Voice: put OPENCODE_GO_API_KEY in the ignored local .env file first
docker compose -f compose.yaml -f compose.remote.yaml up -d --build voice llm proxy

# Stop Remote Voice before switching back
docker compose -f compose.yaml -f compose.remote.yaml down
```

Use the same Compose file pair for all commands that inspect, restart, or stop the selected mode, for example `docker compose -f compose.yaml -f compose.remote.yaml ps` and `logs --follow voice llm proxy`. Do not run both mode graphs together because they share the browser port and service names. Remote defaults are `OPENCODE_GO_BASE_URL=https://opencode.ai/zen/go/v1`, `OPENCODE_GO_MODEL=deepseek-v4.1-flash`, and `WEB_SEARCH_BACKEND=duckduckgo`; if the supplied account requires the documentation endpoint, set `OPENCODE_GO_BASE_URL=https://opencode.ai/inference/go/openai/v1` in `.env`. Both HTTPS paths are explicitly validated; other hosts and paths are rejected. A search provider may impose its own access policy or credential requirements. `OPENCODE_GO_API_KEY` is not a search API key.

The remote model endpoint has not yet been verified with the user's account. The task specification records a difference between the requested base URL and the endpoint advertised in OpenCode Go documentation; confirm it with the supplied account before treating live Remote Voice as accepted. The implementation keeps the requested URL as its default, accepts only the documented same-provider correction above, and reports provider failures without exposing credentials. An empty key leaves the service unready; a nonempty but invalid key produces a redacted authentication error when a request is made. The Remote model download command does not require the key.

## Internal LLM service contract

The voice container calls `http://llm:8001` only on the private Compose network. It sends `POST /v1/sessions/{session_id}/turns/{turn_id}/stream` with a JSON `text` field. Both IDs are lowercase 32-character hexadecimal values; the LLM service owns the corresponding in-memory history, bounded to recent user/assistant pairs and the configured token budget. Audio never crosses this API.

The response is newline-delimited JSON. A successful searched turn may emit `tool_call` (`name`, `result_count`), then `sources` (up to five `title`/HTTPS `url`/bounded `snippet` records), followed by one or more `delta` (`text`) events and a terminal `done`. Only `delta.text` enters TTS. A failed or empty turn ends with `error` (`code`, redacted `message`) and no `done`; the voice client treats EOF without `done` as a truncated-stream error. Cancellation intentionally closes the stream without `done`.

Voice interrupts with `POST /v1/sessions/{session_id}/turns/{turn_id}/cancel`; ending a browser session uses `DELETE /v1/sessions/{session_id}`, which cancels its active turn and removes its in-memory history. Local Gemma cancellation asks llama.cpp to stop at the next token boundary. Remote HTTP requests are closed locally, and the owned search worker process is terminated. Whether OpenCode Go stops server-side inference after a disconnected request has not been verified.

## Components

| Component | Role |
| --- | --- |
| Silero VAD v6 ONNX bundled with the pinned Faster-Whisper package | Streaming speech onset and endpoint detection |
| Faster-Whisper `small` | Local Russian transcription |
| Gemma 3 1B IT Q4_K_M | Local Dota-focused response generation |
| Silero `v4_cyrillic.pt`, speaker `b_lez` | Local Russian speech synthesis |
| FastAPI WebSocket app and browser AudioWorklet | Continuous microphone streaming and incremental playback |
| Nginx unprivileged proxy | Loopback browser endpoint and WebSocket forwarding |

The VAD uses the dependency already present in the locked image; no new package or model download is required. Its thresholds and endpoint timings are configured in `workspace/settings.json`.

## Prepare and start

Run commands from the repository root:

```sh
sh tools/init.sh
docker compose -f compose.yaml -f compose.local.yaml config --quiet
docker compose -f compose.yaml -f compose.local.yaml run --rm download
docker compose -f compose.yaml -f compose.local.yaml run --rm download python /opt/el-voice/download_models.py --verify
docker compose -f compose.yaml -f compose.local.yaml up -d voice llm proxy
docker compose -f compose.yaml -f compose.local.yaml ps
curl --fail http://localhost:8765/healthz
```

Open **http://localhost:8765**, allow microphone access, and use headphones. The page requests browser echo cancellation, but browser support is best effort; speaker playback is not a supported setup. Start and stop controls open and close the microphone session.

The only host-published port is `127.0.0.1:8765` on Nginx. The voice service remains on the internal network and its port is not published. The browser does not require an audio-device mount into Docker.

## Debugging and logs

Follow the voice service logs while speaking:

```sh
docker compose -f compose.yaml -f compose.local.yaml logs --follow voice llm proxy
```

The local voice logs include structured events for recognized user text (`user_transcript`), assistant text (`assistant_reply` or `assistant_reply_interrupted`), model/session errors, browser playback errors, generated audio chunk sizes, and playback scheduling/latency. This content logging is enabled at the user's request for debugging. Docker may retain these logs, so treat them as conversation history. The application never logs audio payloads and does not save microphone audio or transcripts to `data/`.
The proxy service's Nginx errors are in the same combined output; normal access logs do not include WebSocket message bodies.

To inspect current readiness and recent diagnostics:

```sh
curl --fail http://localhost:8765/api/status
docker compose -f compose.yaml -f compose.local.yaml logs --since=10m --tail=200 voice llm proxy
```

`playback_started` means the browser scheduled a generated audio buffer in Web Audio; it cannot prove that sound reached the selected physical output. If the transcript and answer appear but you hear nothing, look for `tts_chunk_ready`, `playback_started`, and `browser_error`; then check the browser tab's output device, system volume, and mute state. A logged browser error or a missing `playback_started` narrows the failure to synthesis, WebSocket delivery, or browser scheduling.

## Development and verification

Application files under `workspace/` are bind-mounted read-only into the container, so edit them on the host and restart the service to apply changes. Routine source changes do not require an image rebuild:

```sh
docker compose -f compose.yaml -f compose.local.yaml restart voice llm
docker compose -f compose.yaml -f compose.local.yaml exec voice python /opt/el-voice/check_environment.py
```

Standalone voice synthesis can be checked with:

```sh
docker compose -f compose.yaml -f compose.local.yaml exec voice python /workspace/tts_smoke.py --text "Привет, это проверка локального голоса."
```

The smoke test writes `/data/tts-test.wav`; remove that optional sample after listening if you do not want to keep it. For a full conversation, use the browser at `http://localhost:8765`.

The Python lock targets CPython 3.12 / Linux ARM64. The image uses the official Python Bookworm base pinned by digest, Debian runtime libraries, and hash-locked Python wheels. `requirements.in` contains direct inputs; regenerate the lock with `uv` and `sh tools/lock.sh` when changing dependencies. Model files and receipts are ignored by Git and remain outside image layers.

## Model sources and licensing

| Model | Publisher |
| --- | --- |
| Gemma 3 1B IT Q4_K_M GGUF | [ggml-org on Hugging Face](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF) |
| Whisper small, CTranslate2 format | [Systran on Hugging Face](https://huggingface.co/Systran/faster-whisper-small) |
| Silero Cyrillic v4 | [Silero models](https://github.com/snakers4/silero-models) |
| Faster-Whisper | [SYSTRAN](https://github.com/SYSTRAN/faster-whisper) |
| llama-cpp-python | [abetlen](https://github.com/abetlen/llama-cpp-python) |
| PyTorch | [Official installation guide](https://pytorch.org/get-started/locally/) |

`models.json` pins model revisions and available checksums. The downloader verifies the publisher assets and records local SHA-256 receipts. Silero's `.pt` is executable PyTorch package content; only load the configured model from the publisher URL. Check model terms before redistribution or commercial use. Silero v4 Cyrillic is published under CC BY-NC-SA 4.0, including a noncommercial restriction; see the [license](https://github.com/snakers4/silero-models/blob/master/LICENSE).

Project background material is indexed in [`references/REFERENCES.md`](references/REFERENCES.md). Task specifications and evidence are in [`tasks/`](tasks/), including [SPEC-001](tasks/SPEC-001-LOCAL-VOICE-CONVERSATION.md) and [SPEC-002](tasks/SPEC-002-REMOTE-VOICE-LLM-SERVICE.md).
