# EL Voice

## Scope and ownership

- This is an independent voice assistant for Docker Desktop on Apple Silicon with two
  explicitly selected modes: offline Local Voice and opt-in Remote Voice.
- Never copy code, installers, packages, model mirrors, or binary builds from the previously
  reviewed assistant author.
- Local Voice is the original offline deliverable: no cloud inference, API keys or runtime
  internet access. Preserve it.
- Remote Voice is the SPEC-002 extension: generation may go to the configured OpenCode Go
  provider and web search to DuckDuckGo, but only through the internal LLM service. Only
  final text, bounded context and search queries leave the Mac; microphone audio stays local.
- Keep source code and project documentation in English.
- Work on a dedicated branch. Preserve unrelated edits.

## Layout

- `workspace/`: browser app and local speech pipeline, mounted at `/workspace`.
- `llm_service/`: internal LLM service for both modes, mounted at
  `/opt/llm-service/llm_service`.
- `tools/`: bootstrap and model preparation tools, copied into the image.
- `models/`: downloaded weights and receipts; never commit these files. Not mounted into the
  remote LLM service.
- `data/`: runtime cache and optional conversation data; never commit these files.
- `work/`: temporary checks, screenshots, and test artifacts; never commit these files.
- `compose.local.yaml` / `compose.remote.yaml`: mode overrides for the shared `compose.yaml`.

## Task tracking

- `tasks/TODO.md` is the index of tasks that need to be completed. Each entry links to its
  task specification in `tasks/`.
- Read the referenced specification before starting a task and update its checklist and
  evidence when the work is complete.

## Current milestone

SPEC-001 local voice plus SPEC-002 remote voice/LLM service are implemented:

- local speech capture, streaming Silero VAD, Whisper, Silero `b_lez` and local Gemma through
  the internal LLM service;
- remote LangChain agent for OpenCode Go with typed `web_search` (DuckDuckGo), bounded
  context/tools, clickable sources and remote-mode disclosure;
- separate `llm` Compose service, local/remote override files, pinned dependencies and
  deterministic service/agent checks.

Remaining verification is live: a real Docker image build and both Compose graphs on the
target Mac, a real browser/microphone run, and a live Remote Voice run with the user's
`OPENCODE_GO_API_KEY` (streaming, tool calling, latency). This development environment has no
Docker daemon, browser/microphone or provider key. Do not report those as tested.

## Container contract

- Python 3.12, Linux ARM64, CPU inference. Do not enable CUDA or emulate AMD64.
- The Nginx proxy publishes container port 8080 to host loopback port 8765 by default; the
  voice service's port 8000 and the LLM service's port 8100 are never published directly.
- Browser microphone capture and playback run on the Mac; no host audio-device mount.
- Local mode: voice and LLM share the internal `private` network with no outbound route.
- Remote mode: only the `llm` service additionally joins the egress `provider` network. It
  alone receives `OPENCODE_GO_API_KEY`. The voice, proxy and browser never receive the key.
- `OPENCODE_GO_BASE_URL`, `OPENCODE_GO_MODEL` and `OPENCODE_GO_SESSION_ID` are configuration,
  not secrets. Keep the key in the ignored `.env`; never commit it or put it in Compose YAML.
- The download service uses egress for models. Never mount the home directory, Docker socket,
  credentials, or unrelated projects.
- Model downloads must not load or execute model packages.
- Treat Silero packages as executable pickle content; load only the configured publisher asset.
- Search results are untrusted. The tool accepts only a typed text query; do not add tools
  that execute shell commands or fetch model-generated URLs/private endpoints.

## Verification

- Validate both Compose configurations with
  `docker compose -f compose.yaml -f compose.local.yaml config --quiet` and the remote
  equivalent.
- Run Python syntax checks, node checks, the deterministic service/agent tests, existing
  regressions and the end-to-end local self-test when practical.
- A successful health check means only that the HTTP process is alive. `/api/status` reports
  pipeline and LLM-service readiness; provider readiness is configuration only until a live
  turn succeeds.
- Do not report a tested container build, working model, microphone, conversation, browser
  behavior or live provider response unless tested.
