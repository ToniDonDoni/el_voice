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
    this.checked = false;
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  setAttribute(name, value) { this.attributes ||= new Map(); this.attributes.set(name, value); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  querySelector() { return new Element(); }
  remove() {}
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
      if (!elements.has(selector)) {
        const element = new Element();
        if (selector === "#interrupt-toggle") {
          const markup = html.match(/<input[^>]*id="interrupt-toggle"[^>]*>/)?.[0] || "";
          element.checked = /\bchecked\b/.test(markup);
        }
        elements.set(selector, element);
      }
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
  renderBuildRevision,
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
    const acknowledgements = socket.sent.map(message => {
      try { return JSON.parse(message); } catch (_) { return null; }
    }).filter(message => message?.type === "playback_queue");
    assert.ok(acknowledgements.length > 0, "the browser must report queue duration so the server can apply backpressure");
    assert.ok(acknowledgements.every(message => message.turn_id === turn
      && Number.isInteger(message.sequence) && message.buffered_ms >= 0 && message.consumed_ms >= 0),
    "each audio acknowledgement must identify its turn, packet sequence, and buffered/consumed duration");
    assert.ok(acknowledgements.some(message => message.consumed_ms > 0),
      "source completion must return playback credit to the server");
    assert.ok(acknowledgements.every((message, index) => index === 0
      || message.sequence >= acknowledgements[index - 1].sequence),
    "completion acknowledgements must never move behind the latest scheduled packet");
    assert.ok(acknowledgements.some(message => message.sequence === 1 && message.consumed_ms > 0),
      "completion of packet zero after packet one is scheduled must acknowledge the latest received sequence");
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

  await check("fallback replacement clears partial text and stops queued playback", async () => {
    const { api } = harness();
    const audio = new FakeAudioContext();
    const socket = new FakeWebSocket();
    api.setAudio(audio, socket);
    const turn = "00112233445566778899aabbccddeeff";
    await api.handleServerMessage(JSON.stringify({ type: "text_delta", turn_id: turn, text: "Partial reply." }));
    await api.handleServerMessage(makePacket(turn, 0, 24000));
    const partialSource = audio.sources.find(source => source.when !== undefined);
    assert.ok(partialSource, "the partial response should already be scheduled");
    await api.handleServerMessage(JSON.stringify({ type: "response_replace", turn_id: turn, text: "Я так качался" }));
    assert.equal(api.getAssistantText(turn), "Я так качался", "fallback text must replace partial output");
    assert.equal(partialSource.stopped, true, "already queued partial audio must stop");
    await api.handleServerMessage(makePacket(turn, 0, 24000));
    const fallbackSource = audio.sources.filter(source => source.when !== undefined && !source.stopped);
    assert.equal(fallbackSource.length, 1, "fallback audio must be allowed to play after replacement");
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

  await check("ignored speech does not stop queued assistant audio and both sources finish", async () => {
    const { api } = harness();
    const audio = new FakeAudioContext();
    const socket = new FakeWebSocket();
    api.setAudio(audio, socket);
    const turn = "00112233445566778899aabbccddeeff";
    await api.handleServerMessage(makePacket(turn, 0, 24000));
    const first = audio.sources.find(source => source.when !== undefined);
    assert.ok(first, "the first assistant audio source must start");
    await api.handleServerMessage(makePacket(turn, 1, 24000));
    const second = audio.sources.filter(source => source.when !== undefined)[1];
    assert.ok(second, "the response tail must be queued before the ignored speech event");
    await api.handleServerMessage(JSON.stringify({ type: "speech_ignored", turn_id: turn }));
    assert.equal(first.stopped, undefined, "ignored speech must not stop current assistant audio");
    assert.equal(second.stopped, undefined, "ignored speech must not stop queued assistant audio");

    await api.handleServerMessage(JSON.stringify({ type: "turn_complete", turn_id: turn }));
    const hasPlaybackFinished = () => socket.sent.map(message => {
      try { return JSON.parse(message); } catch (_) { return null; }
    }).some(message => message?.type === "playback_finished" && message.turn_id === turn);
    assert.equal(hasPlaybackFinished(), false,
      "the browser must wait for both queued sources to finish before acknowledging playback");

    audio.finishNext();
    assert.ok(first.ended, "the first source must reach its natural completion callback");
    assert.equal(hasPlaybackFinished(), false,
      "one completed source must not finish the turn while the response tail is playing");
    audio.finishNext();
    assert.ok(second.ended, "the response tail must reach its natural completion callback");
    assert.equal(hasPlaybackFinished(), true,
      "the browser must report playback completion after both sources finish");
  });

  await check("failed turns release the server playback slot", async () => {
    const { api } = harness();
    const audio = new FakeAudioContext();
    const socket = new FakeWebSocket();
    api.setAudio(audio, socket);
    const turn = "00112233445566778899aabbccddeeff";
    await api.handleServerMessage(makePacket(turn, 0, 24000));
    const source = audio.sources.find(item => item.when !== undefined);
    assert.ok(source, "partial audio must remain scheduled after a later generation failure");
    await api.handleServerMessage(JSON.stringify({
      type: "error", turn_id: turn, message: "The assistant could not complete this turn.",
    }));
    await api.handleServerMessage(JSON.stringify({ type: "turn_failed", turn_id: turn }));
    const finishedAcknowledgement = () => socket.sent.some(message => {
      try {
        const event = JSON.parse(message);
        return event.type === "playback_finished" && event.turn_id === turn;
      } catch (_) {
        return false;
      }
    });
    assert.equal(finishedAcknowledgement(), false,
      "the browser must let already scheduled audio finish before releasing the turn");
    audio.finishNext();
    assert.equal(finishedAcknowledgement(), true,
      "the browser must acknowledge after queued partial audio finishes");
  });

  await check("WebSocket closure immediately stops scheduled playback", async () => {
    const { api } = harness();
    const audio = new FakeAudioContext();
    await api.connectSocket();
    const ws = FakeWebSocket.instances.at(-1);
    api.setAudio(audio, ws);
    await api.handleServerMessage(makePacket("00112233445566778899aabbccddeeff", 0, 24000));
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
    const audio = new FakeAudioContext();
    await api.connectSocket();
    const oldSocket = FakeWebSocket.instances.at(-1);
    api.setAudio(audio, oldSocket);
    const turn = "00112233445566778899aabbccddeeff";
    oldSocket.onmessage({ data: JSON.stringify({ type: "text_delta", turn_id: turn, text: "old start" }) });
    oldSocket.onclose();
    await api.connectSocket();
    api.setAudio(audio, FakeWebSocket.instances.at(-1));
    oldSocket.onmessage({ data: JSON.stringify({ type: "text_delta", turn_id: turn, text: " stale" }) });
    oldSocket.onmessage({ data: makePacket(turn, 0, 24000) });
    assert.equal(api.getAssistantText(turn), "old start", "stale old-session text must be ignored");
    assert.equal(audio.sources.filter(source => source.when !== undefined).length, 0,
      "stale old-session audio must not be scheduled after reconnect");
  });

  await check("visible mute and volume controls change the playback gain", async () => {
    assert.match(html, /id="stop-button"[^>]*hidden/, "end session must be hidden until a session starts");
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
    await mute.listeners.get("click")();
    volume.value = "0.4";
    await volume.listeners.get("input")({ currentTarget: volume });
    assert.equal(audio.gains.at(-1).gain.value, 0.4);
  });

  await check("visible interrupt preference defaults on and updates the active socket", async () => {
    const markup = html.match(/<input[^>]*id="interrupt-toggle"[^>]*>/)?.[0];
    assert.ok(markup && /\btype="checkbox"/.test(markup) && /\bchecked\b/.test(markup),
      "the interrupt checkbox must be visible and checked by default");
    assert.match(html, /<label[^>]*for="interrupt-toggle"[^>]*>[\s\S]*?Interrupt assistant when I speak[\s\S]*?<\/label>/,
      "the preference must have an accessible visible label");
    assert.match(html, /When off, speech during an active reply is ignored/,
      "the control must explain that speech during an active reply is ignored when disabled");

    const playbackHarness = harness();
    playbackHarness.api.setAudio(new FakeAudioContext(), new FakeWebSocket());
    await playbackHarness.api.handleServerMessage(makePacket("00112233445566778899aabbccddeeff", 0, 24000));
    const playbackToggle = playbackHarness.elements.get("#interrupt-toggle");
    playbackToggle.checked = false;
    playbackToggle.listeners.get("change")({ currentTarget: playbackToggle });
    assert.equal(playbackHarness.elements.get("#status-label").textContent,
      "Speaking · interruption is off", "the playback status must describe the disabled setting");
    playbackToggle.checked = true;
    playbackToggle.listeners.get("change")({ currentTarget: playbackToggle });
    assert.equal(playbackHarness.elements.get("#status-label").textContent,
      "Speaking · you can interrupt", "the playback status must update when interruption is re-enabled");

    const { api, elements } = harness();
    await api.connectSocket();
    let ws = FakeWebSocket.instances.at(-1);
    let start = JSON.parse(ws.sent[0]);
    assert.equal(start.interrupt_enabled, true,
      "new sessions must send the default-on interrupt preference");
    const toggle = elements.get("#interrupt-toggle");
    assert.ok(toggle.listeners.has("change"), "the visible checkbox must have a change handler");

    await api.stopSession(false);
    toggle.checked = false;
    await api.connectSocket();
    ws = FakeWebSocket.instances.at(-1);
    start = JSON.parse(ws.sent[0]);
    assert.equal(start.interrupt_enabled, false,
      "a disabled preference selected before session start must be sent to the service");

    toggle.listeners.get("change")({ currentTarget: toggle });
    assert.deepEqual(JSON.parse(ws.sent.at(-1)), { type: "interrupt_setting", enabled: false });
    toggle.checked = true;
    toggle.listeners.get("change")({ currentTarget: toggle });
    assert.deepEqual(JSON.parse(ws.sent.at(-1)), { type: "interrupt_setting", enabled: true },
      "turning interruption on must update the active socket");

    toggle.checked = false;
    toggle.listeners.get("change")({ currentTarget: toggle });
    api.setSessionWanted(true);
    const disconnectedSocket = ws;
    disconnectedSocket.close();
    const reconnect = await new Promise((resolve, reject) => {
      const deadline = Date.now() + 1500;
      const poll = () => {
        const candidate = FakeWebSocket.instances.at(-1);
        if (candidate !== disconnectedSocket && candidate?.sent.length) return resolve(candidate);
        if (Date.now() >= deadline) return reject(new Error("socket did not reconnect"));
        setTimeout(poll, 10);
      };
      poll();
    });
    assert.equal(JSON.parse(reconnect.sent[0]).interrupt_enabled, false,
      "a reconnect must preserve the current unchecked preference");
    await api.stopSession(false);
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
    assert.match(diagnostics, /Browser audio queue limit/i, "the user should see the active audio queue limit");
    api.reportClientError("synthetic browser playback error");
    const report = JSON.parse(socket.sent.at(-1));
    assert.equal(report.type, "client_error");
    assert.equal(report.message, "synthetic browser playback error");
  });

  await check("top-left build badge shows the source commit", async () => {
    assert.match(html, /id="build-revision"[^>]*>COMMIT —<\/span>/,
      "the top bar must reserve a visible source revision badge");
    const { api, elements } = harness();
    api.renderBuildRevision("0123456789abcdef0123456789abcdef01234567");
    const badge = elements.get("#build-revision");
    assert.equal(badge.textContent, "COMMIT 0123456789ab");
    assert.equal(badge.title, "Git commit 0123456789abcdef0123456789abcdef01234567");
    api.renderBuildRevision("0123456789abcdef0123456789abcdef01234567-dirty");
    assert.equal(badge.textContent, "COMMIT 0123456789ab-dirty");
    assert.match(badge.title, /uncommitted changes/);
    api.renderBuildRevision(undefined);
    assert.equal(badge.textContent, "COMMIT unversioned");
  });

  if (failures.length) {
    console.error(failures.join("\n"));
    process.exitCode = 1;
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
