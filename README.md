# EL Voice

An independent, local Dota voice assistant for Docker Desktop on Apple Silicon. The browser captures microphone audio and plays responses; the Linux ARM64 CPU container runs streaming VAD, Whisper transcription, Gemma response generation, and Silero `b_lez` speech synthesis. No code or binary artifacts from the referenced assistant project are used.

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
docker compose config --quiet
docker compose run --rm download
docker compose run --rm download python /opt/el-voice/download_models.py --verify
docker compose up -d voice proxy
docker compose ps
curl --fail http://localhost:8765/healthz
```

Open **http://localhost:8765**, allow microphone access, and use headphones. The page requests browser echo cancellation, but browser support is best effort; speaker playback is not a supported setup. Start and stop controls open and close the microphone session.

The only host-published port is `127.0.0.1:8765` on Nginx. The voice service remains on the internal network and its port is not published. The browser does not require an audio-device mount into Docker.

## Debugging and logs

Follow the voice service logs while speaking:

```sh
docker compose logs --follow voice proxy
```

The local voice logs include structured events for recognized user text (`user_transcript`), assistant text (`assistant_reply` or `assistant_reply_interrupted`), model/session errors, browser playback errors, generated audio chunk sizes, and playback scheduling/latency. This content logging is enabled at the user's request for debugging. Docker may retain these logs, so treat them as conversation history. The application never logs audio payloads and does not save microphone audio or transcripts to `data/`.
The proxy service's Nginx errors are in the same combined output; normal access logs do not include WebSocket message bodies.

To inspect current readiness and recent diagnostics:

```sh
curl --fail http://localhost:8765/api/status
docker compose logs --since=10m --tail=200 voice proxy
```

`playback_started` means the browser scheduled a generated audio buffer in Web Audio; it cannot prove that sound reached the selected physical output. If the transcript and answer appear but you hear nothing, look for `tts_chunk_ready`, `playback_started`, and `browser_error`; then check the browser tab's output device, system volume, and mute state. A logged browser error or a missing `playback_started` narrows the failure to synthesis, WebSocket delivery, or browser scheduling.

## Development and verification

Application files under `workspace/` are bind-mounted read-only into the container, so edit them on the host and restart the service to apply changes. Routine source changes do not require an image rebuild:

```sh
docker compose restart voice
docker compose exec voice python /opt/el-voice/check_environment.py
```

Standalone voice synthesis can be checked with:

```sh
docker compose exec voice python /workspace/tts_smoke.py --text "Привет, это проверка локального голоса."
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

Project background material is indexed in [`references/REFERENCES.md`](references/REFERENCES.md). The task specification and acceptance evidence are in [`tasks/SPEC-001-LOCAL-VOICE-CONVERSATION.md`](tasks/SPEC-001-LOCAL-VOICE-CONVERSATION.md).
