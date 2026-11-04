- [ ] [Implement local Dota voice conversation](SPEC-001-LOCAL-VOICE-CONVERSATION.md).
  - [x] Implementation in `workspace/` (streaming VAD, Whisper, Gemma streaming through the
    internal LLM service, Silero `b_lez`, barge-in, reconnects).
  - [x] Development-sandbox end-to-end self-test passes (21 checks with both services).
  - [ ] Real container + Mac browser acceptance (P0/P1): start the Local graph, open
    http://localhost:8765, then record latency and privacy evidence.
- [ ] [Review follow-ups for the voice conversation](SPEC-001-LOCAL-VOICE-CONVERSATION-Review.md):
  R1-R5 implemented with local evidence; browser/container acceptance remains (see the
  review file's implementation status).
- [ ] [Remote Voice and LLM provider service](SPEC-002-REMOTE-VOICE-LLM-SERVICE.md): internal
  LLM service and both Compose modes implemented; deterministic service/agent/search checks
  and the local end-to-end self-test pass. Live OpenCode Go (no key), Docker image/graphs and
  browser acceptance remain unverified. Evidence:
  [implementation report](SPEC-002-REMOTE-VOICE-LLM-SERVICE-IMPLEMENTATION-REPORT.md).
