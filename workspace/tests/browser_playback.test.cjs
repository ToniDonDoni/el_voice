const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor() {
    this.listeners = new Map();
    this.children = [];
    this.classList = { add() {}, remove() {}, contains() { return false; } };
    this.dataset = {};
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  querySelector() { return new Element(); }
  scrollTo() {}
}

class FakeAudioContext {
  constructor() {
    this.state = "running";
    this.currentTime = 0;
    this.destination = {};
    this.sources = [];
  }
  createBuffer(_channels, length, sampleRate) {
    return { duration: length / sampleRate, getChannelData: () => new Float32Array(length) };
  }
  createBufferSource() {
    const source = {
      connect() {},
      start(when) { this.when = when; },
      stop() { this.stopped = true; },
    };
    this.sources.push(source);
    return source;
  }
}

class FakeWebSocket {
  constructor() { this.readyState = FakeWebSocket.OPEN; this.sent = []; }
  send(message) { this.sent.push(message); }
}
FakeWebSocket.OPEN = 1;
FakeWebSocket.CLOSING = 2;

const elements = new Map();
const document = {
  querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  },
  createElement() { return new Element(); },
};
const window = { addEventListener() {} };
const source = fs.readFileSync(new URL("../static/app.js", `file://${__filename}`), "utf8") + `
window.__elVoiceTest = {
  setAudio(context, ws) { audioContext = context; socket = ws; },
  handleServerMessage,
  getQueueEnd(turnId) { return nextPlaybackTime.get(turnId) || 0; },
};`;
vm.runInNewContext(source, {
  document, window, WebSocket: FakeWebSocket, AudioContext: FakeAudioContext,
  location: { protocol: "http:", host: "localhost" },
  fetch: async () => ({ json: async () => ({ ready: true, stage: "ready", models: {}, metrics: {} }) }),
  setInterval() {}, setTimeout, console, ArrayBuffer, Int16Array, Map, Set, Math, Promise,
});

async function main() {
  const api = window.__elVoiceTest;
  const audio = new FakeAudioContext();
  const socket = new FakeWebSocket();
  api.setAudio(audio, socket);
  const turnId = "turn-buffer-test";
  const pcm = new ArrayBuffer(24000 * 2); // one second at 24 kHz
  for (let index = 0; index < 8; index += 1) {
    await api.handleServerMessage(JSON.stringify({ type: "audio_chunk", turn_id: turnId, sample_rate: 24000 }));
    await api.handleServerMessage(pcm);
  }
  const bufferedMs = (api.getQueueEnd(turnId) - audio.currentTime) * 1000;
  assert.ok(bufferedMs <= 2000, `browser playback queue exceeded 2000 ms: ${bufferedMs}`);

  await api.handleServerMessage(JSON.stringify({ type: "interrupt", turn_id: turnId }));
  const startsAfterInterrupt = audio.sources.filter(source => source.when !== undefined).length;
  await api.handleServerMessage(JSON.stringify({ type: "audio_chunk", turn_id: turnId, sample_rate: 24000 }));
  await api.handleServerMessage(pcm);
  assert.equal(audio.sources.filter(source => source.when !== undefined).length, startsAfterInterrupt,
    "delayed audio from an interrupted turn must not be scheduled");

  const packetAudio = new FakeAudioContext();
  api.setAudio(packetAudio, socket);
  const packetTurn = "00112233445566778899aabbccddeeff";
  await api.handleServerMessage(makePacket(packetTurn, 0, 24000, true));
  assert.equal(packetAudio.sources.filter(source => source.when !== undefined).length, 1,
    "a valid self-describing audio packet should be played");
  await api.handleServerMessage(makePacket(packetTurn, 2, 24000, true));
  assert.equal(packetAudio.sources.filter(source => source.when !== undefined).length, 1,
    "out-of-sequence audio should be rejected");
  await api.handleServerMessage(JSON.stringify({ type: "interrupt", turn_id: packetTurn }));
  await api.handleServerMessage(makePacket(packetTurn, 1, 24000, true));
  assert.equal(packetAudio.sources.filter(source => source.when !== undefined).length, 1,
    "late audio from an interrupted turn should be rejected");
  await api.handleServerMessage(makePacket("10112233445566778899aabbccddeeff", 0, 24000, false));
  assert.equal(packetAudio.sources.filter(source => source.when !== undefined).length, 1,
    "malformed binary framing should be rejected");
}

function makePacket(turnHex, sequence, sampleRate, validMagic) {
  const packet = Buffer.alloc(28 + 960);
  packet.write(validMagic ? "ELVA" : "BAD!", 0, "ascii");
  Buffer.from(turnHex, "hex").copy(packet, 4);
  packet.writeUInt32BE(sequence, 20);
  packet.writeUInt32BE(sampleRate, 24);
  return packet.buffer.slice(packet.byteOffset, packet.byteOffset + packet.byteLength);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
