const TARGET_RATE = 16000;
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const clearButton = document.querySelector("#clear-button");
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
let nextPlaybackTime = new Map();
let echoCancellationSetting = "not checked";

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

function stopPlayback(turnId = null) {
  const ids = turnId ? [turnId] : [...playbackSources.keys()];
  for (const id of ids) {
    const sources = playbackSources.get(id);
    if (sources) {
      for (const source of sources) {
        try { source.stop(0); } catch (_) { /* Already ended. */ }
      }
    }
    playbackSources.delete(id);
    nextPlaybackTime.delete(id);
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
  source.connect(audioContext.destination);
  const sources = playbackSources.get(turnId) || new Set();
  playbackSources.set(turnId, sources);
  sources.add(source);
  const scheduledAt = Math.max(audioContext.currentTime + 0.015, nextPlaybackTime.get(turnId) || 0);
  nextPlaybackTime.set(turnId, scheduledAt + audioBuffer.duration);
  source.onended = () => {
    sources.delete(source);
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
  setStatus("Speaking · you can interrupt", "busy");
}

async function handleServerMessage(data) {
  if (data instanceof ArrayBuffer) {
    const metadata = pendingAudio;
    pendingAudio = null;
    if (metadata?.turn_id === currentResponseTurn || !currentResponseTurn) await playPCM(data, metadata);
    return;
  }
  let event;
  try { event = JSON.parse(data); } catch (_) { return; }
  switch (event.type) {
    case "session_started":
      reconnectAttempt = 0;
      setStatus("Listening for your next question", "ready");
      liveLabel.textContent = "LISTENING";
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
    case "transcript":
      addMessage("user", event.turn_id, event.text);
      break;
    case "generation_started":
      assistantBody(event.turn_id);
      setStatus("Thinking…", "busy");
      liveLabel.textContent = "THINKING";
      break;
    case "text_delta":
      appendAssistantText(event.turn_id, event.text);
      break;
    case "audio_chunk":
      pendingAudio = event;
      break;
    case "turn_complete":
      turnComplete.add(event.turn_id);
      addMessageStatus(event.turn_id, "Response generated");
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
        echo_cancellation: track?.getSettings?.().echoCancellation ?? null,
      }));
      resolve();
    };
    ws.onmessage = event => { void handleServerMessage(event.data); };
    ws.onerror = () => {
      if (!opened) reject(new Error("Could not connect to the local voice service"));
    };
    ws.onclose = () => {
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
    ["Platform", `${status.system || "—"} ${status.architecture || ""}`],
    ["Pipeline", status.stage || "unknown"],
    ["VAD", status.models?.vad || "unknown"],
    ["Speech recognition", status.models?.stt || "unknown"],
    ["Language model", status.models?.llm || "unknown"],
    ["Voice", status.models?.tts || "unknown"],
    ["Echo cancellation", echoCancellationSetting],
    ["Response latency p50 / p95", formatMetric(status.metrics?.speech_end_to_first_audio_ms)],
    ["Interrupt latency p50 / p95", formatMetric(status.metrics?.speech_onset_to_playback_stopped_ms)],
  ];
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.textContent = `${label}: ${value}`;
    row.className = value === "ready" || value.includes(" ms") ? "good" : "";
    diagnostics.append(row);
  }
}

function formatMetric(metric) {
  if (!metric || metric.p50 == null) return "not measured";
  return `${metric.p50} / ${metric.p95} ms`;
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    renderDiagnostics(status);
    if (!sessionWanted) {
      if (status.ready) {
        startButton.disabled = false;
        setStatus("Local models ready", "ready");
      } else if (status.stage === "error") {
        startButton.disabled = true;
        setStatus(`Local pipeline failed to initialize (${status.error || "unknown error"})`, "error");
      } else {
        startButton.disabled = true;
        setStatus("Loading local models…", "busy");
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
