const TARGET_RATE = 16000;
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const clearButton = document.querySelector("#clear-button");
const interruptToggle = document.querySelector("#interrupt-toggle");
const statusDot = document.querySelector("#status-dot");
const statusLabel = document.querySelector("#status-label");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const liveLabel = document.querySelector("#live-label");
const vadIndicator = document.querySelector("#vad-indicator");
const diagnostics = document.querySelector("#diagnostics-content");

let mediaStream = null;
let audioContext = null;
let socket = null;
let captureNode = null;
let silentGain = null;
let sessionWanted = false;
let stopping = false;
let reconnectAttempt = 0;
let currentResponseTurn = null;
let pendingAudio = null;
let turnComplete = new Set();
let playbackStartedTurns = new Set();
let playbackFinishedTurns = new Set();
let cancelledTurns = new Set();
let messageNodes = new Map();
let playbackSources = new Map();
let playbackGeneration = new Map();
let nextPlaybackTime = new Map();
let audioSequence = new Map();
let playbackBufferedMs = new Map();
let outputGain = null;
let volumeMuted = false;
let echoCancellationSetting = "not checked";
let unavailableAudioTurns = new Set();
let fallbackTurns = new Set();

function setStatus(label, state = "") {
  statusLabel.textContent = label;
  statusDot.className = `status-dot ${state}`;
}

function scrollMessages() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, turnId, text = "") {
  welcome.hidden = true;
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.dataset.turnId = turnId || "";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "EL Voice";
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  article.append(label, body);
  messages.append(article);
  messageNodes.set(`${role}:${turnId}`, { article, body });
  clearButton.disabled = false;
  scrollMessages();
  return body;
}

function assistantBody(turnId) {
  const existing = messageNodes.get(`assistant:${turnId}`);
  return existing ? existing.body : addMessage("assistant", turnId);
}

function appendAssistantText(turnId, text) {
  assistantBody(turnId).textContent += text;
  scrollMessages();
}

function addMessageStatus(turnId, text) {
  const node = messageNodes.get(`assistant:${turnId}`)?.article;
  if (!node) return;
  let status = node.querySelector(".message-status");
  if (!status) {
    status = document.createElement("div");
    status.className = "message-status";
    node.append(status);
  }
  status.textContent = text;
}

function addSources(turnId, sources) {
  const article = messageNodes.get(`assistant:${turnId}`)?.article;
  if (!article || !Array.isArray(sources)) return;
  article.querySelector(".source-list")?.remove();
  const valid = sources.filter(source => {
    try { return new URL(source.url).protocol === "https:"; } catch (_) { return false; }
  }).slice(0, 5);
  if (!valid.length) return;
  const section = document.createElement("section");
  section.className = "source-list";
  const heading = document.createElement("div");
  heading.className = "source-heading";
  heading.textContent = "Web sources";
  section.append(heading);
  const list = document.createElement("ul");
  for (const source of valid) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = String(source.title || "Source").slice(0, 200);
    item.append(link);
    if (source.snippet) {
      const snippet = document.createElement("p");
      snippet.textContent = String(source.snippet).slice(0, 500);
      item.append(snippet);
    }
    list.append(item);
  }
  section.append(list);
  article.append(section);
  scrollMessages();
}

function stopPlayback(turnId = null) {
  const ids = turnId ? [turnId] : [...new Set([...playbackSources.keys(), ...nextPlaybackTime.keys()])];
  for (const id of ids) {
    playbackGeneration.set(id, (playbackGeneration.get(id) || 0) + 1);
    const sources = playbackSources.get(id);
    if (sources) {
      for (const source of sources) {
        try { source.stop(0); } catch (_) { /* Already ended. */ }
      }
    }
    playbackSources.delete(id);
    nextPlaybackTime.delete(id);
    playbackBufferedMs.delete(id);
    audioSequence.delete(id);
  }
  if (!turnId) {
    playbackBufferedMs.clear();
    audioSequence.clear();
  }
  if (!turnId || currentResponseTurn === turnId) currentResponseTurn = null;
}

function reportClientError(message) {
  const safeMessage = String(message || "Unknown browser playback error").slice(0, 500);
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "client_error", message: safeMessage }));
  } else {
    void fetch("/api/client-error", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: safeMessage }),
      keepalive: true,
    }).catch(() => {});
  }
  console.error("EL Voice:", safeMessage);
}

function finishPlayback(turnId) {
  if (cancelledTurns.has(turnId) || playbackFinishedTurns.has(turnId)) return;
  playbackFinishedTurns.add(turnId);
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "playback_finished", turn_id: turnId }));
  }
  playbackBufferedMs.delete(turnId);
  audioSequence.delete(turnId);
  if (currentResponseTurn === turnId) currentResponseTurn = null;
  liveLabel.textContent = sessionWanted ? "LISTENING" : "MIC OFF";
  vadIndicator.classList.remove("active");
  setStatus("Listening for your next question", "ready");
}

async function playPCM(buffer, metadata) {
  if (!metadata || !audioContext || cancelledTurns.has(metadata.turn_id)) return;
  if (audioContext.state !== "running") {
    try {
      await audioContext.resume();
    } catch (error) {
      reportClientError(`AudioContext resume failed: ${error.message || error}`);
      setStatus("Audio playback is blocked by the browser", "error");
      return;
    }
  }
  const turnId = metadata.turn_id;
  if (currentResponseTurn && currentResponseTurn !== turnId) stopPlayback(currentResponseTurn);
  currentResponseTurn = turnId;
  const generation = playbackGeneration.get(turnId) || 0;
  const samples = new Int16Array(buffer);
  const audioBuffer = audioContext.createBuffer(1, samples.length, metadata.sample_rate);
  const channel = audioBuffer.getChannelData(0);
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    channel[i] = samples[i] / 32768;
    peak = Math.max(peak, Math.abs(channel[i]));
  }
  if (peak < 0.0001) reportClientError(`Generated audio chunk is silent (${buffer.byteLength} bytes)`);
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  if (!outputGain) {
    outputGain = audioContext.createGain();
    outputGain.gain.value = volumeMuted ? 0 : Number(document.querySelector("#volume-control")?.value || 1);
    outputGain.connect(audioContext.destination);
  }
  source.connect(outputGain);
  const sources = playbackSources.get(turnId) || new Set();
  playbackSources.set(turnId, sources);
  sources.add(source);
  const scheduledAt = Math.max(audioContext.currentTime + 0.015, nextPlaybackTime.get(turnId) || 0);
  nextPlaybackTime.set(turnId, scheduledAt + audioBuffer.duration);
  const sequence = metadata.sequence;
  const bufferedMs = Math.max(0, (nextPlaybackTime.get(turnId) - audioContext.currentTime) * 1000);
  const previousBufferedMs = playbackBufferedMs.get(turnId) || 0;
  const consumedMs = Math.max(0, previousBufferedMs + audioBuffer.duration * 1000 - bufferedMs);
  playbackBufferedMs.set(turnId, bufferedMs);
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "playback_queue", turn_id: turnId, sequence,
      consumed_ms: consumedMs,
      buffered_ms: bufferedMs }));
  }
  source.onended = () => {
    sources.delete(source);
    if ((playbackGeneration.get(turnId) || 0) !== generation) return;
    if (!cancelledTurns.has(turnId) && socket?.readyState === WebSocket.OPEN) {
      const updatedBufferedMs = Math.max(0,
        ((nextPlaybackTime.get(turnId) || audioContext.currentTime) - audioContext.currentTime) * 1000);
      const previous = playbackBufferedMs.get(turnId) || 0;
      playbackBufferedMs.set(turnId, updatedBufferedMs);
      socket.send(JSON.stringify({ type: "playback_queue", turn_id: turnId,
        sequence: audioSequence.get(turnId) ?? 0,
        consumed_ms: Math.max(0, previous - updatedBufferedMs),
        buffered_ms: updatedBufferedMs }));
    }
    if (sources.size === 0) {
      playbackSources.delete(turnId);
      if (turnComplete.has(turnId)) finishPlayback(turnId);
    }
  };
  try {
    source.start(scheduledAt);
  } catch (error) {
    sources.delete(source);
    if (sources.size === 0) playbackSources.delete(turnId);
    nextPlaybackTime.delete(turnId);
    reportClientError(`Audio source start failed: ${error.message || error}`);
    setStatus("Could not start audio playback", "error");
    return;
  }
  if (!playbackStartedTurns.has(turnId)) {
    playbackStartedTurns.add(turnId);
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "playback_started", turn_id: turnId }));
    }
  }
  liveLabel.textContent = "SPEAKING · LISTENING";
  setStatus(interruptToggle.checked ? "Speaking · you can interrupt" : "Speaking · interruption is off", "busy");
}

async function handleServerMessage(data) {
  if (data instanceof ArrayBuffer) {
    try {
      if (data.byteLength < 30 || data.byteLength % 2 !== 0) throw new Error("Truncated audio packet");
      const view = new DataView(data);
      if (String.fromCharCode(...new Uint8Array(data, 0, 4)) !== "ELVA") throw new Error("Invalid audio packet marker");
      const turnId = [...new Uint8Array(data, 4, 16)].map(value => value.toString(16).padStart(2, "0")).join("");
      const sequence = view.getUint32(20, false);
      const sampleRate = view.getUint32(24, false);
      const previous = audioSequence.get(turnId);
      if (sampleRate !== 24000 || (previous !== undefined && sequence !== previous + 1)) {
        throw new Error("Invalid audio packet rate or sequence");
      }
      if (cancelledTurns.has(turnId)) return;
      const payload = data.slice(28);
      if (!payload.byteLength || payload.byteLength > 24000 * 2 * 2) throw new Error("Invalid audio packet length");
      audioSequence.set(turnId, sequence);
      const previousBuffered = Math.max(0, ((nextPlaybackTime.get(turnId) || 0) - (audioContext?.currentTime || 0)) * 1000);
      await playPCM(payload, { turn_id: turnId, sequence, sample_rate: sampleRate,
        previous_buffered_ms: previousBuffered });
    } catch (error) {
      reportClientError(`Rejected audio packet: ${error.message || error}`);
    }
    return;
  }
  let event;
  try { event = JSON.parse(data); } catch (_) { return; }
  switch (event.type) {
    case "session_started":
      reconnectAttempt = 0;
      if (typeof event.interrupt_enabled === "boolean") {
        interruptToggle.checked = event.interrupt_enabled;
      }
      setStatus("Listening for your next question", "ready");
      liveLabel.textContent = "LISTENING";
      break;
    case "interrupt_setting":
      if (typeof event.enabled === "boolean") interruptToggle.checked = event.enabled;
      break;
    case "not_ready":
      setStatus("Local models are still loading", "busy");
      break;
    case "busy":
      setStatus("Another browser session is active", "error");
      break;
    case "speech_started":
      vadIndicator.classList.add("active");
      liveLabel.textContent = "SPEECH DETECTED";
      setStatus("Listening to your question", "busy");
      break;
    case "speech_ended":
      vadIndicator.classList.remove("active");
      liveLabel.textContent = "PROCESSING";
      setStatus("Recognizing your speech…", "busy");
      break;
    case "speech_ignored":
      vadIndicator.classList.remove("active");
      liveLabel.textContent = "SPEAKING";
      setStatus("Reply continues; interruption is off", "busy");
      addMessageStatus(event.turn_id, "Speech ignored while the current reply finishes");
      break;
    case "transcript":
      addMessage("user", event.turn_id, event.text);
      break;
    case "generation_started":
      assistantBody(event.turn_id);
      setStatus("Thinking…", "busy");
      liveLabel.textContent = "THINKING";
      break;
    case "tool_call":
      addMessageStatus(event.turn_id, "Searching the web…");
      break;
    case "sources":
      addSources(event.turn_id, event.sources);
      addMessageStatus(event.turn_id, event.status === "error"
        ? (event.sources?.length
          ? "One search failed; using earlier sources"
          : "Search unavailable; answer may be unverified")
        : event.sources?.length
          ? `Grounded with ${event.sources.length} web source${event.sources.length === 1 ? "" : "s"}`
          : "Search returned no sources");
      break;
    case "text_delta":
      appendAssistantText(event.turn_id, event.text);
      break;
    case "response_replace": {
      const turnId = event.turn_id;
      turnComplete.delete(turnId);
      playbackFinishedTurns.delete(turnId);
      unavailableAudioTurns.delete(turnId);
      fallbackTurns.add(turnId);
      if (pendingAudio?.turn_id === turnId) pendingAudio = null;
      stopPlayback(turnId);
      const article = messageNodes.get(`assistant:${turnId}`)?.article;
      article?.querySelector(".source-list")?.remove();
      article?.querySelector(".message-status")?.remove();
      assistantBody(turnId).textContent = String(event.text || "Я так качался");
      scrollMessages();
      setStatus("Using a fallback response", "busy");
      break;
    }
    case "response_audio_unavailable":
      unavailableAudioTurns.add(event.turn_id);
      addMessageStatus(event.turn_id, "Fallback response shown; voice synthesis failed");
      break;
    case "turn_complete":
      turnComplete.add(event.turn_id);
      addMessageStatus(event.turn_id, unavailableAudioTurns.has(event.turn_id)
        ? "Fallback response shown; voice synthesis failed"
        : fallbackTurns.has(event.turn_id) ? "Fallback response" : "Response generated");
      if (!playbackSources.get(event.turn_id)?.size) finishPlayback(event.turn_id);
      break;
    case "turn_failed":
      turnComplete.add(event.turn_id);
      if (!playbackSources.get(event.turn_id)?.size) finishPlayback(event.turn_id);
      break;
    case "interrupt": {
      const oldId = event.turn_id;
      cancelledTurns.add(oldId);
      turnComplete.delete(oldId);
      stopPlayback(oldId);
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "playback_stopped", turn_id: oldId }));
      }
      const oldMessage = messageNodes.get(`assistant:${oldId}`)?.article;
      if (oldMessage) {
        oldMessage.classList.add("interrupted");
        addMessageStatus(oldId, "Interrupted");
      }
      if (pendingAudio?.turn_id === oldId) pendingAudio = null;
      setStatus("I heard you — stopping the reply", "busy");
      break;
    }
    case "empty_turn":
      liveLabel.textContent = "LISTENING";
      setStatus("I did not catch that. Try again.", "ready");
      if (event.turn_id && socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "playback_finished", turn_id: event.turn_id }));
      }
      break;
    case "listening":
      setStatus("Listening for your next question", "ready");
      liveLabel.textContent = "LISTENING";
      break;
    case "error":
      setStatus(event.message || "A local voice component failed", "error");
      if (event.turn_id) addMessageStatus(event.turn_id, "Could not complete this turn");
      reportClientError(`Server reported ${event.code || "error"}: ${event.message || "unknown failure"}`);
      break;
  }
}

function socketURL() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}/ws/voice`;
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(socketURL());
    ws.binaryType = "arraybuffer";
    socket = ws;
    let opened = false;
    ws.onopen = () => {
      opened = true;
      const track = mediaStream?.getAudioTracks()[0];
      ws.send(JSON.stringify({
        type: "start",
        sample_rate: TARGET_RATE,
        interrupt_enabled: interruptToggle.checked,
        echo_cancellation: track?.getSettings?.().echoCancellation ?? null,
      }));
      resolve();
    };
    ws.onmessage = event => {
      if (socket === ws) void handleServerMessage(event.data);
    };
    ws.onerror = () => {
      if (!opened) reject(new Error("Could not connect to the local voice service"));
    };
    ws.onclose = () => {
      if (socket !== ws) return;
      if (socket === ws) socket = null;
      pendingAudio = null;
      stopPlayback();
      if (sessionWanted && !stopping) void reconnectSocket();
    };
  });
}

async function reconnectSocket() {
  if (!sessionWanted || stopping) return;
  if (reconnectAttempt >= 5) {
    setStatus("Connection lost. End the session and try again.", "error");
    await stopSession(false);
    return;
  }
  const delay = Math.min(500 * (2 ** reconnectAttempt), 8000);
  reconnectAttempt += 1;
  setStatus("Connection lost — reconnecting…", "busy");
  await new Promise(resolve => setTimeout(resolve, delay));
  if (!sessionWanted || stopping) return;
  try {
    await connectSocket();
  } catch (_) {
    if (sessionWanted) void reconnectSocket();
  }
}

async function startSession() {
  if (sessionWanted) return;
  startButton.disabled = true;
  setStatus("Requesting microphone access…", "busy");
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: { ideal: 1 },
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
    const inputTrack = mediaStream.getAudioTracks()[0];
    const inputSettings = inputTrack?.getSettings?.() || {};
    echoCancellationSetting = inputSettings.echoCancellation === undefined
      ? "unsupported or unknown"
      : String(inputSettings.echoCancellation);
    audioContext = new AudioContext();
    await audioContext.resume();
    await audioContext.audioWorklet.addModule("/static/audio-capture.js");
    captureNode = new AudioWorkletNode(audioContext, "pcm16-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { targetRate: TARGET_RATE },
    });
    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;
    captureNode.port.onmessage = event => {
      if (sessionWanted && socket?.readyState === WebSocket.OPEN) socket.send(event.data);
    };
    const source = audioContext.createMediaStreamSource(mediaStream);
    source.connect(captureNode);
    captureNode.connect(silentGain);
    silentGain.connect(audioContext.destination);
    sessionWanted = true;
    stopping = false;
    stopButton.hidden = false;
    startButton.hidden = true;
    liveLabel.textContent = "CONNECTING";
    await connectSocket();
  } catch (error) {
    sessionWanted = false;
    startButton.disabled = false;
    setStatus(error.name === "NotAllowedError" ? "Microphone permission was denied" : "Could not start the voice session", "error");
    reportClientError(`Voice session start failed (${error.name || "Error"}): ${error.message || error}`);
    await releaseAudio();
  }
}

async function releaseAudio() {
  if (captureNode) {
    captureNode.port.onmessage = null;
    try { captureNode.disconnect(); } catch (_) { /* Already disconnected. */ }
    captureNode = null;
  }
  if (silentGain) {
    try { silentGain.disconnect(); } catch (_) { /* Already disconnected. */ }
    silentGain = null;
  }
  if (mediaStream) {
    for (const track of mediaStream.getTracks()) track.stop();
    mediaStream = null;
  }
  if (audioContext) {
    await audioContext.close().catch(() => {});
    audioContext = null;
  }
}

async function stopSession(userInitiated = true) {
  stopping = true;
  sessionWanted = false;
  pendingAudio = null;
  stopPlayback();
  if (socket && socket.readyState < WebSocket.CLOSING) {
    socket.close(1000, "Session stopped");
  }
  socket = null;
  await releaseAudio();
  outputGain = null;
  stopButton.hidden = true;
  startButton.hidden = false;
  startButton.disabled = false;
  liveLabel.textContent = "MIC OFF";
  vadIndicator.classList.remove("active");
  if (userInitiated) setStatus("Session ended", "");
  stopping = false;
}

async function clearChat() {
  for (const id of [...playbackSources.keys()]) stopPlayback(id);
  messages.replaceChildren();
  welcome.hidden = false;
  messageNodes.clear();
  turnComplete.clear();
  playbackStartedTurns.clear();
  playbackFinishedTurns.clear();
  cancelledTurns.clear();
  clearButton.disabled = true;
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "clear_history" }));
}

function renderDiagnostics(status) {
  diagnostics.replaceChildren();
  const rows = [
    ["Voice mode", status.mode === "remote" ? "Remote — text sent to provider" : "Local — offline inference"],
    ["Platform", `${status.system || "—"} ${status.architecture || ""}`],
    ["Pipeline", status.stage || "unknown"],
    ["VAD", status.models?.vad || "unknown"],
    ["Speech recognition", status.models?.stt || "unknown"],
    ["Language model", status.models?.llm || "unknown"],
    ["Voice", status.models?.tts || "unknown"],
    ["Echo cancellation", echoCancellationSetting],
    ["Response latency p50 / p95", formatMetric(status.metrics?.speech_end_to_first_audio_ms)],
    ["Interrupt latency p50 / p95", formatMetric(status.metrics?.speech_onset_to_playback_stopped_ms)],
    ["Browser audio queue limit", `${status.audio?.max_buffered_audio_ms || 2000} ms`],
  ];
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.textContent = `${label}: ${value}`;
    row.className = value === "ready" || value.includes(" ms") ? "good" : "";
    diagnostics.append(row);
  }
}

document.querySelector("#mute-button").addEventListener("click", event => {
  volumeMuted = !volumeMuted;
  if (outputGain) outputGain.gain.value = volumeMuted ? 0 : Number(document.querySelector("#volume-control").value);
  const button = event?.currentTarget || document.querySelector("#mute-button");
  button.setAttribute("aria-pressed", String(volumeMuted));
  button.textContent = volumeMuted ? "Unmute" : "Mute";
});
document.querySelector("#volume-control").addEventListener("input", event => {
  if (outputGain && !volumeMuted) outputGain.gain.value = Number(event.currentTarget.value);
});
interruptToggle.addEventListener("change", event => {
  if ([...playbackSources.values()].some(sources => sources.size)) {
    setStatus(event.currentTarget.checked
      ? "Speaking · you can interrupt"
      : "Speaking · interruption is off", "busy");
  }
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      type: "interrupt_setting", enabled: event.currentTarget.checked,
    }));
  }
});

function formatMetric(metric) {
  if (!metric || metric.p50 == null) return "not measured";
  return `${metric.p50} / ${metric.p95} ms`;
}

function renderBuildRevision(revision) {
  const fullRevision = typeof revision === "string" && revision.trim()
    ? revision.trim().slice(0, 64)
    : "unknown";
  const isDirty = fullRevision.endsWith("-dirty");
  const commit = isDirty ? fullRevision.slice(0, -6) : fullRevision;
  const shortCommit = commit === "unknown" ? "unversioned" : commit.slice(0, 12);
  const suffix = isDirty ? "-dirty" : "";
  const element = document.querySelector("#build-revision");
  element.textContent = `COMMIT ${shortCommit}${suffix}`;
  element.title = commit === "unknown"
    ? "No Git revision was supplied to Docker Compose"
    : `Git commit ${commit}${isDirty ? " (working tree contains uncommitted changes)" : ""}`;
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    renderBuildRevision(status.revision);
    renderDiagnostics(status);
    const remote = status.mode === "remote";
    const badge = document.querySelector("#mode-badge");
    badge.replaceChildren(Object.assign(document.createElement("span"), { className: "local-dot" }));
    badge.append(document.createTextNode(remote ? "REMOTE · TEXT SHARED" : "LOCAL · PRIVATE"));
    const note = document.querySelector("#privacy-note p");
    if (remote) {
      note.replaceChildren();
      const strong = document.createElement("strong");
      strong.textContent = "Remote Voice sends text to OpenCode Go.";
      note.append(strong, document.createTextNode(
        " Microphone audio stays local. Recognized text and bounded history go to OpenCode Go; search queries and result snippets may be sent to OpenCode Go and DuckDuckGo. Use headphones; echo cancellation is best effort."
      ));
      document.querySelector("#footer-mode").textContent = "Remote text inference and web search · Speech audio stays local";
    } else {
      document.querySelector("#footer-mode").textContent = "Runs on your Mac · CPU inference in Docker · No remote text or audio";
    }
    if (!sessionWanted) {
      if (status.ready) {
        startButton.disabled = false;
        setStatus(remote ? "Remote Voice ready" : "Local Voice ready", "ready");
      } else if (status.stage === "error") {
        startButton.disabled = true;
        setStatus(`${remote ? "Remote" : "Local"} voice pipeline failed to initialize (${status.error || "unknown error"})`, "error");
      } else {
        startButton.disabled = true;
        setStatus(remote ? "Connecting remote language service…" : "Loading local voice models…", "busy");
      }
    }
  } catch (_) {
    if (!sessionWanted) setStatus("Local voice service is unavailable", "error");
  }
}

startButton.addEventListener("click", () => { void startSession(); });
stopButton.addEventListener("click", () => { void stopSession(); });
clearButton.addEventListener("click", () => { void clearChat(); });
window.addEventListener("pagehide", () => { void stopSession(false); });
window.addEventListener("error", event => reportClientError(`Browser error: ${event.message}`));
window.addEventListener("unhandledrejection", event => {
  reportClientError(`Browser promise rejected: ${event.reason?.message || event.reason || "unknown reason"}`);
});

void pollStatus();
setInterval(() => { void pollStatus(); }, 2500);
