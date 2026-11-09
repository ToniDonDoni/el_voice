# SPEC-007: Clipboard Images and Typed Questions

## 1. Problem and Current State

Questions currently arrive through local speech recognition only. The browser has
no text composer or screenshot attachment, and the remote provider receives plain
text. A misrecognized question cannot be corrected by typing, and the assistant
cannot see the user's game screenshot.

## 2. Required State

- Add a text composer with Send; Enter sends and Shift+Enter inserts a newline.
  Typed questions use the existing conversation, response streaming, and TTS.
- In Remote Voice, provide Paste image and support pasting an image directly into
  the composer. Accept images only from the clipboard, without a file picker.
- Preview one pending image and allow removal or replacement. Attach it once to
  the next submitted typed question or successfully recognized voice question.
  An image-only Send asks the model to describe it. Empty recognition keeps it.
- Keep typed text available without opening the microphone. Share the existing
  WebSocket session when voice is active. An explicit Send interrupts the current
  reply. Keep voice interruption preference semantics for microphone input.
- Send text and the inline image together through the internal LLM service to
  the configured OpenCode Go DeepSeek model using Chat Completions content parts.
- Convert clipboard images to PNG, fit within 2048 pixels per side, and enforce
  4 MiB per image in the browser and both services. Reject malformed input and
  reject images in Local Voice visibly. Do not fetch image URLs.
- Log typed/recognized text and correlated attachment metadata (ID, bytes,
  dimensions, SHA-256, provider request image count), never image bytes/base64.
- Hold images in memory only for the current question; do not add them to saved
  history, logs, files, or browser storage. Clear pending images on clear/end/
  disconnect. Preserve the existing provider-key and local-audio boundary.

## 3. Acceptance Criteria

- [x] Text-only Send works without microphone permission and during a voice session.
- [x] Clipboard button and composer paste show a removable preview; no file picker.
- [x] Image and typed question reach one provider user message, including all
  model calls in that agent run. Image-only Send uses a description question.
- [x] A queued image accompanies exactly one recognized voice question; empty
  recognition keeps it, and later questions do not reuse it.
- [x] Clear, disconnect, replacement, and removal do not leak stale attachments.
- [x] Local text input works; Local Voice cannot send images remotely.
- [x] Invalid/oversized images and text are rejected visibly, without logging data.
- [x] Logs establish browser acceptance, LLM receipt, and actual provider request
  attachment using the same checksum, alongside the submitted text.
- [x] Automated browser/protocol/provider tests pass, both Compose modes validate,
  and a live remote image question produces a correct answer and done event.
- [x] Restart Remote Voice and inspect the rendered composer for user testing.

## 4. Design

Use the existing authenticated, origin-checked voice WebSocket for typed turns
and one staged clipboard attachment. The browser can open it without microphone
capture. A staged image is consumed only when a nonempty question is submitted
to the LLM; the browser associates its preview with the acknowledged turn.
Large image controls have a separate bounded size allowance; ordinary controls
retain their small limit. The internal request validates the image again and
passes it as an image_url content block in the same HumanMessage as the text.
The image remains available through the planning/final-answer model calls, but
conversation history retains only question and answer text.

## 5. Verification Evidence

- Before implementation, a live synthetic PNG sent to the configured OpenCode
  endpoint returned the correct red-square/blue-circle description (HTTP 200).
- Python: 101 unittest cases in the existing Linux ARM64 image, including
  voice-image single-use behavior, empty-STT retention, typed interrupt, Local
  text path, Local image rejection, invalid input, payload-free metadata logs,
  internal HTTP propagation, and both LangChain model calls.
- Node: browser playback/input contracts and audio-capture suites passed.
- Live Chromium: real clipboard button, native Cmd+V in the composer, image
  preview/removal, typed Send and Enter, no microphone permission request,
  successful remote shape recognition, and a text-only follow-up passed.
- Live provider logs matched the same image SHA-256 at voice receipt, LLM receipt,
  and both outgoing model calls. The next question reported image_count=0.
- Desktop and 390-pixel browser layouts inspected; no horizontal overflow or
  uncaught browser errors. Synthetic screenshots are in ignored
  `work/clipboard-input/`.
- Both Compose modes validate; Python syntax and git diff whitespace checks pass.
  Remote services were recreated and report ready; local and existing public
  browser endpoints returned HTTP 200.
- Physical microphone capture and audible speaker output were not tested in this
  pass. The image-plus-voice association used deterministic recognition tests;
  real image processing used the browser text channel and the actual provider.
- A separate existing TTS limitation was observed: an answer consisting only of
  the digit "4" raised ValueError during synthesis and triggered the configured
  fallback. A normal sentence with the number written as a word succeeded.
  This change does not modify that existing fallback/synthesis policy.

## 6. API References

- [DeepSeek Vision](https://api-docs.deepseek.com/guides/vision/): text and inline
  image_url blocks in one user message.
- [OpenCode Go](https://opencode.ai/docs/go/): OpenAI-compatible inference and the
  configured DeepSeek model. Live acceptance used the already configured endpoint;
  no model, credentials, or endpoint were changed.
