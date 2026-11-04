# EL Voice

An independent local voice assistant workspace for Docker Desktop on an Apple Silicon Mac.
This milestone prepares the container and model files. The browser conversation pipeline
will be implemented after the user builds and starts the environment.

No source code, installer, infrastructure package, binary build, or model mirror from the
previously reviewed assistant author is used here.

## What the image installs

| Component | Source / purpose |
| --- | --- |
| Python 3.12 on Debian Bookworm | Official Docker image, pinned by digest |
| FFmpeg, OpenMP, libsndfile, curl, CA certificates | Debian repositories; audio decoding and CPU inference |
| FastAPI, Uvicorn, WebSockets | Our future browser/API layer |
| Faster-Whisper, CTranslate2, PyAV, ONNX Runtime | Local speech recognition and audio processing |
| llama-cpp-python | Official upstream Linux ARM64 CPU wheel pinned by SHA-256 |
| PyTorch CPU | Official PyTorch Linux ARM64 wheel; no CUDA dependencies |
| NumPy, Hugging Face Hub and resolved dependencies | Supporting libraries |

Direct inputs are in `requirements.in`. Python runtime dependencies and artifact hashes are
resolved in `requirements.lock` for CPython 3.12 / Linux ARM64. Debian packages are not a
complete snapshot, so this is not a byte-reproducible image. Changing the lock requires `uv`
on the development machine and `sh tools/lock.sh`.

There is no host audio-device mount, GUI toolkit, Vosk requirement, CUDA toolkit, or Git
checkout of someone else's application. The first pipeline will use Whisper directly;
browser capture will handle microphone access and playback.

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

Open **http://localhost:8765** in a browser. The current page explicitly says that the
environment is ready and the voice pipeline is not yet implemented. A healthy container
is not evidence of working speech recognition, model inference, or a conversation.

The only published mapping is `127.0.0.1:8765 -> proxy:8080`; Nginx forwards HTTP and
WebSocket requests to `voice:8000` over the internal network. The voice service has no
published port and no normal outbound internet route. Nginx joins a separate bridge network
for host port publishing; that bridge permits potential outbound access from the Nginx
container. Disabling masquerading would not guarantee egress isolation. Nginx's fixed
upstream config forwards requests only to `voice:8000`. The download service uses egress to
fetch models, while `voice` remains isolated on its internal network.

The Dockerfile is used to build an image; Compose creates the container, publishes the
port and mounts the folders. Running the image without those mounts will not find app.py.

## Optional: download models now

This is separate from the image build. It uses the same image through a temporary service
with internet access. It does not publish ports or run model code.

```sh
docker compose run --rm download
docker compose run --rm download python /opt/el-voice/download_models.py --verify
docker compose exec voice python /opt/el-voice/check_environment.py
```

| Model | Publisher | Approximate download |
| --- | --- | --- |
| Gemma 3 1B IT Q4_K_M GGUF | ggml-org on Hugging Face | 806 MB |
| Whisper small, CTranslate2 format | Systran on Hugging Face | 486 MB including tokenizer/config |
| Silero v4 Cyrillic, selected voice b_lez (male Lezghian) | models.silero.ai | 35.4 MB |

`models.json` pins the Hugging Face commit revisions and publisher checksums. The downloader
verifies SHA-256 for large weights and Git blob checksums for small configuration files.
It resumes partial downloads and records local SHA-256 receipts. Re-run it to continue an
interrupted download. Models and receipts stay ignored by Git.

The Silero publisher URL has a versioned filename, but no independently verified publisher
SHA-256 was available when preparing this recipe. Its first download trusts the official
HTTPS source; the resulting local SHA-256 receipt detects subsequent changes. This is not
a publisher signature. Silero uses an executable PyTorch package/pickle: do not load
replacement `.pt` files from arbitrary sources. The preparation script only downloads bytes.

Gemma has its own terms and Silero model licensing must be checked with the publisher;
these model files are not automatically covered by our code or a Python package's license.

Publisher references:

- https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF
- https://huggingface.co/Systran/faster-whisper-small
- https://github.com/snakers4/silero-models
- https://github.com/SYSTRAN/faster-whisper
- https://github.com/abetlen/llama-cpp-python
- https://pytorch.org/get-started/locally/

## Check the selected voice

The configured TTS model is `v4_cyrillic.pt`, speaker `b_lez`, at 24 kHz on CPU.
This is a male Lezghian voice. Russian pronunciation and stress need listening tests;
the model does not provide automatic stress placement. The configuration lives in
`workspace/settings.json` and is mounted with our source code.

After downloading the models and starting the voice service:

```sh
docker compose exec voice python /workspace/tts_smoke.py --text "Hello. This is a local voice test."
```

For the actual listening check, replace the quoted text with a short Russian sentence,
including a number or a name. The test verifies the configured model's download receipt
before loading the official PyTorch package, checks that the speaker exists, runs CPU
synthesis, checks the waveform, and writes `data/tts-test.wav`. It prints loading time,
generation time and audio duration. Successful synthesis does not guarantee natural
pronunciation; listen to the sample.

Open **http://localhost:8765**, then use the voice-test audio player. Alternatively open
**http://localhost:8765/audio/tts-test.wav** directly. This verifies TTS playback only;
there is still no microphone input, STT-to-LLM conversation, or interruption handling.

If an older image was already built before changing the model manifest, run
`docker compose build voice` and `docker compose up -d --force-recreate voice` again.
Existing downloaded files are preserved; the old Russian voice is no longer selected.

Silero v4 Cyrillic is covered by the publisher's CC BY-NC-SA 4.0 license, including its
noncommercial restriction. See the [publisher documentation](https://github.com/snakers4/silero-models#cyrillic-languages-v4)
and [license](https://github.com/snakers4/silero-models/blob/master/LICENSE).

## Project references

Background references and locally saved source material are kept in `References/`.
See [`references/REFERENCES.md`](References/REFERENCES.md) for the article link.

## Continue development in the ready container

```sh
docker compose exec voice sh
docker compose exec voice python /opt/el-voice/check_environment.py --skip-models
docker compose logs --tail=100 voice
```

Edit `workspace/` on the Mac; changes appear immediately in `/workspace`. Python packages
are installed in the image's virtual environment, not in the source mount. Restart the
service after source edits with `docker compose restart voice`. Future FastAPI code can
replace `workspace/app.py` while keeping the existing image and port mapping.

For the next milestone: browser microphone input, local Whisper transcription, Gemma
response generation, Silero speech, interruption, and one complete browser audio test.

```sh
docker compose stop voice
docker compose down
```

These commands preserve the bind-mounted model and data files. Do not delete `models/`
unless you intend to download the weights again.
