const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor() {
    this.listeners = new Map();
    this.children = [];
    this.classList = { add() {}, remove() {}, contains() { return false; } };
    this.dataset = {};
    this.value = "1";
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  querySelector() { return new Element(); }
}

class FakeAudioContext {
  constructor() {
    this.state = "running";
    this.currentTime = 0;
    this.destination = {};
    this.sources = [];
    this.gains = [];
  }
  async close() { this.state = "closed"; }
  createBuffer(_channels, length, sampleRate) {
    return { duration: length / sampleRate, getChannelData: () => new Float32Array(length) };
  }
  createBufferSource() {
    const source = {
      connect(target) { this.target = target; },
      start(when) { this.when = when; },
      stop() { this.stopped = true; },
    };
    this.sources.push(source);
    return source;
  }
  finishNext() {
    const source = this.sources.find(item => item.when !== undefined && !item.ended && !item.stopped);
    assert.ok(source, "a scheduled source should be available to finish");
    this.currentTime = Math.max(this.currentTime + 1, source.when + 1);
    source.ended = true;
    source.onended?.();
  }
  createGain() {
    const gain = { gain: { value: 1 }, connect() {} };
    this.gains.push(gain);
    return gain;
  }
}

class FakeWebSocket {
  constructor() {
    this.readyState = FakeWebSocket.OPEN;
    this.sent = [];
    FakeWebSocket.instances.push(this);
    setTimeout(() => this.onopen?.(), 0);
  }
  send(message) { this.sent.push(message); }
  close() { this.readyState = FakeWebSocket.CLOSING; this.onclose?.(); }
}
FakeWebSocket.OPEN = 1;
FakeWebSocket.CLOSING = 2;
FakeWebSocket.instances = [];

const appSource = fs.readFileSync(new URL("../static/app.js", `file://${__filename}`), "utf8");
const html = fs.readFileSync(new URL("../static/index.html", `file://${__filename}`), "utf8");
const failures = [];

function harness() {
  const elements = new Map();
  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    },
    createElement() { return new Element(); },
  };
  const window = { events: new Map(), addEventListener(name, callback) { this.events.set(name, callback); } };
  const source = appSource + `
window.__elVoiceTest = {
  setAudio(context, ws) { audioContext = context; socket = ws; },
  handleServerMessage,
  connectSocket,
  stopSession,
  reportClientError,
  renderDiagnostics,
  setSessionWanted(value) { sessionWanted = value; },
  getQueueEnd(turnId) { return nextPlaybackTime.get(turnId) || 0; },
  getAssistantText(turnId) { return messageNodes.get("assistant:" + turnId)?.body.textContent || ""; },
};`;
  vm.runInNewContext(source, {
    document, window, WebSocket: FakeWebSocket, AudioContext: FakeAudioContext,
    location: { protocol: "http:", host: "localhost" },
    fetch: async () => ({ json: async () => ({ ready: true, stage: "ready", models: {}, metrics: {},
      audio: { max_buffered_audio_ms: 2000 } }) }),
    setInterval() {}, setTimeout, console, ArrayBuffer, Int16Array, Map, Set, Math, Promise,
  });
  return { api: window.__elVoiceTest, elements, window };
}

async function check(name, callback) {
  try { await callback(); }
  catch (error) { failures.push(`${name}: ${error.message}`); }
}

function makePacket(turnHex, sequence, sampleRate, validMagic = true) {
  const packet = Buffer.alloc(28 + 48000);
  packet.write(validMagic ? "ELVA" : "BAD!", 0, "ascii");
  Buffer.from(turnHex, "hex").copy(packet, 4);
  packet.writeUInt32BE(sequence, 20);
  packet.writeUInt32BE(sampleRate, 24);
  return packet.buffer.slice(packet.byteOffset, packet.byteOffset + packet.byteLength);
}

async function run() {
  await check("browser audio queue honors the API limit", async () => {
    const { api } = harness();
    const audio = new FakeAudioContext();
    const socket = new FakeWebSocket();
    api.setAudio(audio, socket);
    const turn = "00112233445566778899aabbccddeeff";
    for (let sequence = 0; sequence < 8; sequence += 1) {
      let bufferedMs = (api.getQueueEnd(turn) - audio.currentTime) * 1000 - 15;
      if (bufferedMs >= 1900) audio.finishNext();
      await api.handleServerMessage(makePacket(turn, sequence, 24000));
      bufferedMs = (api.getQueueEnd(turn) - audio.currentTime) * 1000 - 15;
      assert.ok(bufferedMs <= 2000, `configured browser buffer limit exceeded: ${bufferedMs} ms`);
      if (bufferedMs >= 1900) audio.finishNext();
    }
    while (audio.sources.some(source => source.when !== undefined && !source.ended && !source.stopped)) {
      audio.finishNext();
    }
    const scheduled = audio.sources.filter(source => source.when !== undefined).length;
    const bufferedMs = (api.getQueueEnd(turn) - audio.currentTime) * 1000;
    assert.equal(scheduled, 8, "all audio must play eventually as browser capacity is released");
    assert.ok(bufferedMs <= 2000, `configured browser buffer limit exceeded: ${bufferedMs} ms`);
    assert.ok(socket.sent.some(message => {
      try { return JSON.parse(message).type === "playback_queue"; } catch (_) { return false; }
    }), "the browser must report queue duration so the server can apply backpressure");
  });

  await check("malformed, invalid-rate, and out-of-order packets are rejected", async () => {
    const { api } = harness();
    const audio = new FakeAudioContext();
    api.setAudio(audio, new FakeWebSocket());
    const turn = "00112233445566778899aabbccddeeff";
    await api.handleServerMessage(makePacket(turn, 0, 24000));
    const accepted = audio.sources.filter(source => source.when !== undefined).length;
    assert.equal(accepted, 1, "the valid packet must establish that playback is wired");
    await api.handleServerMessage(makePacket(turn, 2, 24000));
    await api.handleServerMessage(makePacket("10112233445566778899aabbccddeeff", 0, 8000));
    await api.handleServerMessage(new ArrayBuffer(8));
    assert.equal(audio.sources.filter(source => source.when !== undefined).length, accepted,
      "invalid packets must not reach AudioContext");
  });

  await check("interruption discards delayed audio from the cancelled turn", async () => {
    const { api } = harness();
    const audio = new FakeAudioContext();
    api.setAudio(audio, new FakeWebSocket());
    const turn = "00112233445566778899aabbccddeeff";
    await api.handleServerMessage(makePacket(turn, 0, 24000));
    const accepted = audio.sources.filter(source => source.when !== undefined).length;
    assert.equal(accepted, 1, "initial valid audio must be scheduled");
    await api.handleServerMessage(JSON.stringify({ type: "interrupt", turn_id: turn }));
    await api.handleServerMessage(makePacket(turn, 1, 24000));
    assert.equal(audio.sources.filter(source => source.when !== undefined).length, accepted,
      "a delayed cancelled-turn packet must not resume playback");
    assert.ok(audio.sources.every(source => source.stopped), "interruption must stop scheduled sources");
  });

  await check("WebSocket closure immediately stops scheduled playback", async () => {
    const { api } = harness();
    const audio = new FakeAudioContext();
    await api.connectSocket();
    const ws = FakeWebSocket.instances.at(-1);
    api.setAudio(audio, ws);
    await api.handleServerMessage(JSON.stringify({ type: "audio_chunk", turn_id: "close-turn", sample_rate: 24000 }));
    await api.handleServerMessage(new ArrayBuffer(4800));
    assert.ok(audio.sources.some(source => source.when !== undefined), "audio should be scheduled first");
    ws.onclose();
    assert.ok(audio.sources.every(source => source.stopped), "socket close must stop active and scheduled sources");
  });

  await check("stopping a browser session clears playback and closes audio resources", async () => {
    const { api, elements } = harness();
    const audio = new FakeAudioContext();
    await api.connectSocket();
    const ws = FakeWebSocket.instances.at(-1);
    api.setAudio(audio, ws);
    await api.handleServerMessage(makePacket("00112233445566778899aabbccddeeff", 0, 24000));
    assert.ok(audio.sources.some(source => source.when !== undefined), "the test must start playback first");
    await api.stopSession(false);
    assert.ok(audio.sources.every(source => source.stopped), "stop must cancel active and scheduled sources");
    assert.equal(audio.state, "closed", "stop must close the browser AudioContext");
    assert.equal(elements.get("#stop-button").hidden, true);
    assert.equal(elements.get("#start-button").hidden, false);
  });

  await check("messages from an old socket cannot append after reconnect", async () => {
    const { api } = harness();
    await api.connectSocket();
    const oldSocket = FakeWebSocket.instances.at(-1);
    const turn = "old-session-turn";
    oldSocket.onmessage({ data: JSON.stringify({ type: "text_delta", turn_id: turn, text: "old start" }) });
    oldSocket.onclose();
    await api.connectSocket();
    oldSocket.onmessage({ data: JSON.stringify({ type: "text_delta", turn_id: turn, text: " stale" }) });
    assert.equal(api.getAssistantText(turn), "old start", "stale old-session text must be ignored");
  });

  await check("visible mute and volume controls change the playback gain", async () => {
    const muteMarkup = html.match(/<button[^>]*id="mute-button"[^>]*>/)?.[0];
    const volumeMarkup = html.match(/<input[^>]*id="volume-control"[^>]*>/)?.[0];
    assert.ok(muteMarkup && !/\shidden(?:\s|>)/.test(muteMarkup), "mute must be visible in the page markup");
    assert.ok(volumeMarkup && !/\shidden(?:\s|>)/.test(volumeMarkup), "volume must be visible in the page markup");
    const { api, elements } = harness();
    const audio = new FakeAudioContext();
    api.setAudio(audio, new FakeWebSocket());
    const mute = elements.get("#mute-button");
    const volume = elements.get("#volume-control");
    assert.ok(mute.listeners.has("click"), "mute must have an active click handler");
    assert.ok(volume.listeners.has("input"), "volume must have an active input handler");
    await api.handleServerMessage(makePacket("00112233445566778899aabbccddeeff", 0, 24000));
    const source = audio.sources.find(item => item.when !== undefined);
    assert.ok(source, "audio should be scheduled through the active player");
    assert.equal(source.target, audio.gains.at(-1), "scheduled output must route through the user-controlled gain");
    await mute.listeners.get("click")();
    assert.equal(audio.gains.at(-1).gain.value, 0);
    volume.value = "0.4";
    await volume.listeners.get("input")();
    assert.equal(audio.gains.at(-1).gain.value, 0.4);
  });

  await check("audio buffering diagnostics and browser error reporting are visible", async () => {
    const { api, elements } = harness();
    const socket = new FakeWebSocket();
    api.setAudio(new FakeAudioContext(), socket);
    api.renderDiagnostics({
      system: "Linux", architecture: "aarch64", stage: "ready", models: { vad: "ready", stt: "ready", llm: "ready", tts: "ready" },
      metrics: {}, audio: { max_buffered_audio_ms: 2000, current_buffered_audio_ms: 500 },
    });
    const diagnostics = elements.get("#diagnostics-content").children.map(row => row.textContent).join(" ");
    assert.match(diagnostics, /Audio buffering/i, "the user should see the active audio queue limit");
    api.reportClientError("synthetic browser playback error");
    const report = JSON.parse(socket.sent.at(-1));
    assert.equal(report.type, "client_error");
    assert.equal(report.message, "synthetic browser playback error");
  });

  if (failures.length) {
    console.error(failures.join("\n"));
    process.exitCode = 1;
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
