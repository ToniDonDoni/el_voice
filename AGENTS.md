# EL Voice

## Scope and ownership

- This is an independent Dota voice assistant for Docker Desktop on Apple Silicon.
- Never copy code, installers, packages, model mirrors, or binary builds from the previously reviewed assistant author.
- Use upstream Python packages and model publishers. Local Voice is offline and requires no API keys.
- SPEC-002 explicitly adds opt-in Remote Voice: only the dedicated LLM service receives the provider key and outbound HTTPS; recognized text, bounded context, and search data may leave the Mac. Microphone audio and TTS remain local. Do not weaken this separation or pass the key to the voice or proxy services.
- Keep source code and project documentation in English.
- Work on a dedicated branch. Preserve unrelated edits.

## Layout

- `workspace/`: our application code, mounted at `/workspace` in the container.
- `tools/`: bootstrap and model preparation tools, copied into the image.
- `models/`: downloaded weights and receipts; never commit these files.
- `data/`: runtime cache and optional conversation data; never commit these files.
- `work/`: temporary checks, screenshots, and test artifacts; never commit these files.

## Task tracking

- `tasks/TODO.md` is the index of tasks that need to be completed. Each entry links to its task specification in `tasks/`.
- Read the referenced specification before starting a task and update its checklist and evidence when the work is complete.

## Current milestone

The browser conversation pipeline is implemented in the mounted `workspace/` source and runs
in the prepared image. It uses bundled Silero ONNX VAD, local Whisper, an internal LLM service
(Gemma for Local Voice or OpenCode Go for Remote Voice), and Silero `b_lez`.
Local stdout logs include recognized text, assistant replies, and diagnostic errors at the user's
request; audio payloads must never be logged or persisted. Continue verifying browser playback,
barge-in, and latency acceptance scenarios without rebuilding unless dependencies change.

## Container contract

- Python 3.12, Linux ARM64, CPU inference. Do not enable CUDA or emulate AMD64.
- The Nginx proxy publishes container port 8080 to host loopback port 8765 by default;
  the voice service's port 8000 is never published directly.
- Browser microphone capture and playback run on the Mac; no host audio-device mount.
- The voice service uses an internal network and has no direct host-published port. Nginx
  joins internal networks to publish the browser endpoint and has no outbound bridge.
- Local LLM joins only the internal network and mounts Gemma read-only. Remote LLM joins the
  internal network plus an outbound bridge and is the only runtime service with provider-key
  access. The download service uses egress only during model preparation.
- Local and Remote Voice are separate Compose override files. Always stop one mode before
  starting the other; both use the same host port and service names.
- Never mount the home directory, Docker socket, credentials, or unrelated projects.
- Model downloads must not load or execute model packages.
- Treat Silero packages as executable pickle content; load only the configured publisher asset.

## Verification

- Validate the Compose configuration before handing it over.
- Run Python syntax checks, Compose validation, `/healthz` and `/api/status` smoke checks when practical.
- A successful health check means only that the HTTP process is alive.
- Do not report a tested container build, working model, microphone, or conversation unless tested.
- At the next milestone, run `tools/check_environment.py` in the real container, verify the
  published host port, inspect the browser UI, and test one complete audio conversation.
