/**
 * EL Voice browser client.
 *
 * Responsibilities:
 *  - capture the microphone and stream 16 kHz PCM frames to the local service;
 *  - play returned 24 kHz PCM chunks incrementally and tag them by turn;
 *  - stop playback immediately when the server reports barge-in or stop;
 *  - reconnect the WebSocket without reloading the page;
 *  - show pipeline readiness, error and timing state.
 *
 * Audio and transcripts stay in memory only.
 */

'use strict';

import { PlaybackQueue } from './playback.js';

const INPUT_SAMPLE_RATE = 16000;
const INPUT_FRAME_SAMPLES = 512;
const OUTPUT_SAMPLE_RATE = 24000;
const AUDIO_MAGIC = 'EVA1';
const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
const MAX_RECONNECT_ATTEMPTS = 6;

const dom = {
  pill: document.getElementById('state-pill'),
  modeBadge: document.getElementById('mode-badge'),
  remoteDisclosure: document.getElementById('remote-disclosure'),
  start: document.getElementById('start'),
  stop: document.getElementById('stop'),
  volume: document.getElementById('volume'),
  volumeValue: document.getElementById('volume-value'),
  mute: document.getElementById('mute'),
  level: document.getElementById('level'),
  messages: document.getElementById('messages'),
  components: document.getElementById('components'),
  echo: document.getElementById('echo'),
  latency: document.getElementById('latency'),
  barge: document.getElementById('barge'),
  playbackStop: document.getElementById('playback-stop'),
  audioState: document.getElementById('audio-state'),
  error: document.getElementById('error'),
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const STATE_LABELS = {
  idle: 'idle',
  starting: 'starting',
  listening: 'listening',
  processing: 'processing',
  speaking: 'speaking',
  reconnecting: 'reconnecting',
  error: 'error',
};

const ui = {
  setState(state) {
    const label = STATE_LABELS[state] || state;
    dom.pill.textContent = label;
    dom.pill.className = 'pill ' + (STATE_LABELS[state] ? state : '');
  },
  error(message) {
    dom.error.textContent = message;
    dom.error.hidden = false;
  },
  clearError() {
    dom.error.hidden = true;
    dom.error.textContent = '';
  },
  setEcho(state) {
    dom.echo.textContent = state === 'on' ? 'on (browser best effort)'
      : state === 'off' ? 'off — headphones required'
        : state === 'unsupported' ? 'unknown — headphones required' : '—';
  },
  setLevel(rms) {
    dom.level.style.width = Math.min(100, Math.round(rms * 480)) + '%';
  },
  meetComponents(status) {
    const components = (status && status.components && status.components.components) || {};
    const names = ['vad', 'stt', 'llm', 'tts'];
    const llmMode = (status && status.llm && status.llm.mode) || 'unknown';
    const labels = {
      vad: 'Streaming VAD',
      stt: 'Whisper STT',
      llm: `LLM service (${llmMode})`,
      tts: 'Silero TTS',
    };
    dom.components.replaceChildren(...names.map((name) => {
      const row = document.createElement('div');
      const term = document.createElement('dt');
      term.textContent = labels[name];
      const value = document.createElement('dd');
      const state = (components[name] || {}).status || 'pending';
      value.textContent = state;
      value.className = state === 'ready' ? 'ok' : state === 'error' ? 'error' : 'pending';
      row.append(term, value);
      return row;
    }));
  },
  renderStatus(status) {
    ui.meetComponents(status);
    const llm = (status && status.llm) || {};
    const mode = llm.mode || 'unknown';
    if (dom.modeBadge) {
      dom.modeBadge.textContent = mode === 'local' ? 'Local mode'
        : mode === 'remote' ? 'Remote mode'
          : 'Mode unknown';
      dom.modeBadge.className = 'mode-badge ' + mode;
    }
    if (dom.remoteDisclosure) {
      dom.remoteDisclosure.hidden = mode !== 'remote';
    }
    const metrics = status.metrics || {};
    const response = metrics.response_latency_ms || { count: 0 };
    dom.latency.textContent = response.count
      ? `${response.p50} ms / ${response.p95} ms (n=${response.count})`
      : 'no measurements yet';
    const barge = metrics.barge_in_dispatch_ms || { count: 0 };
    dom.barge.textContent = barge.count
      ? `${barge.p50} ms / ${barge.p95} ms (n=${barge.count})`
      : 'no measurements yet';
    const stop = metrics.client_playback_stop_ms || { count: 0 };
    dom.playbackStop.textContent = stop.count
      ? `${stop.p50} ms median`
      : 'no measurements yet';
    if (status.settings_error) {
      ui.error('Configuration error: ' + status.settings_error);
    } else if (llm.error) {
      ui.error('LLM service: ' + llm.error);
    }
  },
};

function addMessage(role, text) {
  const node = document.createElement('div');
  node.className = 'message ' + role;
  const who = document.createElement('span');
  who.className = 'who';
  who.textContent = role === 'user' ? 'You' : 'EL Voice';
  const body = document.createElement('span');
  body.textContent = text;
  node.append(who, body);
  const hint = dom.messages.querySelector('.hint');
  if (hint) hint.remove();
  dom.messages.append(node);
  dom.messages.scrollTop = dom.messages.scrollHeight;
  return body;
}

class Downsampler {
  constructor(inputRate, targetRate = INPUT_SAMPLE_RATE, frameSamples = INPUT_FRAME_SAMPLES) {
    this.ratio = inputRate / targetRate;
    this.frameSamples = frameSamples;
    this.frame = new Int16Array(frameSamples);
    this.length = 0;
    this.accumulator = 0;
    this.count = 0;
  }

  push(channel) {
    for (let index = 0; index < channel.length; index += 1) {
      this.accumulator += channel[index];
      this.count += 1;
      if (this.count >= this.ratio) {
        const value = Math.max(-1, Math.min(1, this.accumulator / this.count));
        this.frame[this.length] = value < 0 ? value * 0x8000 : value * 0x7fff;
        this.length += 1;
        this.accumulator = 0;
        this.count = 0;
        if (this.length >= this.frameSamples) {
          const ready = this.frame;
          this.frame = new Int16Array(this.frameSamples);
          this.length = 0;
          return ready;
        }
      }
    }
    return null;
  }
}

class VoiceClient {
  constructor() {
    this.ws = null;
    this.wantSession = false;
    this.starting = false;
    this.reconnectAttempts = 0;
    this.reconnectTimer = null;
    this.readyWaiter = null;
    this.latestTurnId = null;
    this.turnEnded = null;
    this.playbackDoneSent = false;
    this.audioChunks = 0;
    this.capture = null;
    this.playback = null;
    this.statusTimer = null;
    this.turns = new Map();
    this.echoState = 'unknown';
    this.audioChunks = 0;
    this.volume = 1;
    this.muted = false;
    this._loadVolumePreferences();
  }

  // -- public controls ----------------------------------------------------
  async start() {
    if (this.wantSession) return;
    this.wantSession = true;
    this.starting = true;
    this.reconnectAttempts = 0;
    ui.clearError();
    ui.setState('starting');
    dom.start.disabled = true;
    dom.stop.disabled = false;
    try {
      await this._ensurePlayback();
      await this._ensureCapture();
      await this._connectAndStart();
      ui.setState('listening');
      dom.start.disabled = true;
    } catch (error) {
      this.wantSession = false;
      ui.error('Could not start the session: ' + (error && error.message ? error.message : error));
      ui.setState('error');
      this._stopCapture();
      this._closeSocket();
      dom.start.disabled = false;
      dom.stop.disabled = true;
    } finally {
      this.starting = false;
      this._refreshStatus();
    }
  }

  async stop() {
    this.wantSession = false;
    this.reconnectAttempts = 0;
    window.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this._stopCapture();
    this._flushAll();
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: 'stop' }));
      } catch (error) { /* socket may already be closing */ }
    }
    this._closeSocket();
    this.latestTurnId = null;
    ui.setState('idle');
    dom.start.disabled = false;
    dom.stop.disabled = true;
    ui.setLevel(0);
    this._refreshStatus();
  }

  // -- WebSocket ----------------------------------------------------------
  _openSocket() {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(WS_URL);
      socket.binaryType = 'arraybuffer';
      const timer = window.setTimeout(() => {
        socket.close();
        reject(new Error('WebSocket connection timed out'));
      }, 8000);
      socket.onopen = () => {
        window.clearTimeout(timer);
        this.ws = socket;
        resolve();
      };
      socket.onmessage = (event) => this._onMessage(event);
      socket.onclose = () => {
        window.clearTimeout(timer);
        if (this.ws === socket) {
          this.ws = null;
        }
        this.latestTurnId = null;
        this.turnEnded = null;
        this.playbackDoneSent = false;
        this._flushAll();
        if (this.readyWaiter) {
          this.readyWaiter.reject(new Error('connection closed'));
          this.readyWaiter = null;
        }
        if (this.wantSession && !this.starting) {
          this._scheduleReconnect();
        }
      };
      socket.onerror = () => { /* onclose handles recovery */ };
    });
  }

  _closeSocket() {
    const socket = this.ws;
    this.ws = null;
    if (socket) {
      socket.onclose = null;
      socket.onerror = null;
      socket.onmessage = null;
      try {
        socket.close();
      } catch (error) { /* already closed */ }
    }
  }

  async _connectAndStart() {
    let lastError = '';
    for (let attempt = 0; attempt < 12; attempt += 1) {
      if (!this.wantSession) throw new Error('session cancelled');
      if (!this.ws || this.ws.readyState === WebSocket.CLOSED || this.ws.readyState === WebSocket.CLOSING) {
        await this._openSocket();
      }
      const message = await this._requestStart();
      if (message.type === 'ready') {
        this.reconnectAttempts = 0;
        return message;
      }
      lastError = message.message || message.code || 'not ready';
      this._closeSocket();
      if (!message.retryable) break;
      await sleep(1500);
    }
    throw new Error(lastError);
  }

  _requestStart() {
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this.readyWaiter = null;
        reject(new Error('the local service did not answer'));
      }, 25000);
      this.readyWaiter = {
        resolve: (message) => {
          window.clearTimeout(timer);
          this.readyWaiter = null;
          resolve(message);
        },
        reject: (error) => {
          window.clearTimeout(timer);
          this.readyWaiter = null;
          reject(error);
        },
      };
      this.ws.send(JSON.stringify({ type: 'start' }));
    });
  }

  _scheduleReconnect() {
    if (this.reconnectTimer) return;
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      this.wantSession = false;
      ui.error('Connection to the local service was lost. Press Start to try again.');
      ui.setState('error');
      this._stopCapture();
      return;
    }
    this.reconnectAttempts += 1;
    ui.setState('reconnecting');
    const delay = Math.min(5000, 400 * 2 ** (this.reconnectAttempts - 1));
    this.reconnectTimer = window.setTimeout(async () => {
      this.reconnectTimer = null;
      if (!this.wantSession) return;
      this.starting = true;
      try {
        await this._ensurCaptureAlive();
        await this._connectAndStart();
        ui.setState('listening');
      } catch (error) {
        this._scheduleReconnect();
      } finally {
        this.starting = false;
      }
    }, delay);
  }

  async _ensurCaptureAlive() {
    const track = this.capture && this.capture.stream.getAudioTracks()[0];
    if (!track || track.readyState !== 'live') {
      this._stopCapture();
      await this._ensureCapture();
    }
  }

  _onMessage(event) {
    if (typeof event.data === 'string') {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (error) {
        return;
      }
      this._handleJson(message);
      return;
    }
    this._scheduleAudio(event.data);
  }

  _handleJson(message) {
    switch (message.type) {
      case 'ready':
        this.turns.clear();
        this._flushAll();
        this.latestTurnId = null;
        this.turnEnded = null;
        this.playbackDoneSent = false;
        this._updateAudioState();
        if (this.readyWaiter) this.readyWaiter.resolve(message);
        break;
      case 'state':
        if (message.state) ui.setState(message.state);
        break;
      case 'user_speech_start':
        this.latestTurnId = message.turn_id;
        this.turnEnded = null;
        this.playbackDoneSent = false;
        break;
      case 'user_text':
        this._userText(message.turn_id, message.text);
        break;
      case 'assistant_text':
        this._assistantText(message.turn_id, message.delta);
        break;
      case 'assistant_tool':
        this._assistantTool(message.turn_id, message.name, message.query);
        break;
      case 'assistant_sources':
        this._assistantSources(message.turn_id, message.sources);
        break;
      case 'flush':
        this._flushTurn(message.turn_id);
        break;
      case 'turn_end':
        this.turnEnded = message.turn_id;
        this._maybePlaybackDone(message.turn_id);
        break;
      case 'empty_turn':
        break;
      case 'error':
        if (this.readyWaiter && (message.code === 'not_ready' || message.code === 'busy')) {
          this.readyWaiter.resolve(message);
        } else {
          ui.error(message.message || 'The local pipeline reported an error.');
        }
        break;
      default:
        break;
    }
  }

  // -- microphone capture -------------------------------------------------
  async _ensureCapture() {
    if (this.capture) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('this browser does not support microphone capture');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const track = stream.getAudioTracks()[0];
    const settings = track && track.getSettings ? track.getSettings() : {};
    if (settings.echoCancellation === undefined) {
      this.echoState = 'unsupported';
    } else {
      this.echoState = settings.echoCancellation ? 'on' : 'off';
    }
    ui.setEcho(this.echoState);

    const context = new AudioContext({ latencyHint: 'interactive' });
    const source = context.createMediaStreamSource(stream);
    let node = null;
    let sink = null;
    if (context.audioWorklet) {
      await context.audioWorklet.addModule('/static/capture-worklet.js');
      node = new AudioWorkletNode(context, 'pcm-capture', {
        processorOptions: {
          inputSampleRate: context.sampleRate,
          targetSampleRate: INPUT_SAMPLE_RATE,
          frameSamples: INPUT_FRAME_SAMPLES,
        },
      });
      sink = context.createGain();
      sink.gain.value = 0;
      node.connect(sink).connect(context.destination);
      node.port.onmessage = (event) => this._sendAudioFrame(new Int16Array(event.data));
    } else {
      const processor = context.createScriptProcessor(2048, 1, 1);
      const downsampler = new Downsampler(context.sampleRate);
      processor.onaudioprocess = (event) => {
        const frame = downsampler.push(event.inputBuffer.getChannelData(0));
        if (frame) this._sendAudioFrame(frame);
      };
      processor.connect(context.destination);
      node = processor;
    }
    source.connect(node);
    await context.resume();
    this.capture = { stream, source, node, sink, context };
  }

  _sendAudioFrame(frame) {
    this._updateLevel(frame);
    if (!this.wantSession) return;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(frame.buffer);
    } catch (error) {
      /* the reconnect path will restore the socket */
    }
  }

  _updateLevel(frame) {
    let sum = 0;
    for (let index = 0; index < frame.length; index += 1) {
      const value = frame[index] / 32768;
      sum += value * value;
    }
    ui.setLevel(Math.sqrt(sum / Math.max(1, frame.length)));
  }

  _stopCapture() {
    if (!this.capture) return;
    const { stream, context, node } = this.capture;
    try {
      stream.getTracks().forEach((track) => track.stop());
    } catch (error) { /* already stopped */ }
    try {
      if (node && node.port) node.port.onmessage = null;
      if (node && node.onaudioprocess !== undefined) node.onaudioprocess = null;
      if (node) node.disconnect();
    } catch (error) { /* already disconnected */ }
    try {
      context.close();
    } catch (error) { /* already closed */ }
    this.capture = null;
  }

  // -- audio playback -----------------------------------------------------
  async _ensurePlayback() {
    if (this.playback) return;
    let context;
    try {
      context = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE, latencyHint: 'interactive' });
    } catch (error) {
      // Some browsers refuse a non-device sample rate; buffers keep their own
      // rate and the audio graph resamples them.
      context = new AudioContext({ latencyHint: 'interactive' });
    }
    await context.resume();
    this.playback = new PlaybackQueue(context, this.muted ? 0 : this.volume);
    this._updateAudioState();
  }

  _loadVolumePreferences() {
    try {
      const storedVolume = window.localStorage.getItem('elvoice-volume');
      if (storedVolume !== null) {
        this.volume = Math.max(0, Math.min(1, Number(storedVolume) / 100));
      }
      this.muted = window.localStorage.getItem('elvoice-muted') === '1';
    } catch (error) {
      /* storage may be unavailable; defaults are fine */
    }
    if (dom.volume) {
      dom.volume.value = String(Math.round(this.volume * 100));
    }
    this._renderVolume();
  }

  _renderVolume() {
    if (dom.volumeValue) {
      dom.volumeValue.textContent = `${Math.round(this.volume * 100)}%`;
    }
    if (dom.mute) {
      dom.mute.textContent = this.muted ? 'Unmute' : 'Mute';
      dom.mute.setAttribute('aria-pressed', this.muted ? 'true' : 'false');
    }
    if (this.playback) {
      this.playback.setVolume(this.muted ? 0 : this.volume);
    }
  }

  _setVolume(value) {
    this.volume = Math.max(0, Math.min(1, value));
    if (this.volume > 0) this.muted = false;
    try {
      window.localStorage.setItem('elvoice-volume', String(Math.round(this.volume * 100)));
      window.localStorage.setItem('elvoice-muted', this.muted ? '1' : '0');
    } catch (error) {
      /* storage may be unavailable */
    }
    this._renderVolume();
  }

  _toggleMute() {
    this.muted = !this.muted;
    try {
      window.localStorage.setItem('elvoice-muted', this.muted ? '1' : '0');
    } catch (error) {
      /* storage may be unavailable */
    }
    this._renderVolume();
  }

  _updateAudioState() {
    if (!dom.audioState) return;
    const context = this.playback && this.playback.context;
    dom.audioState.textContent = `${this.audioChunks} chunks · ctx ${context ? context.state : 'n/a'}`;
  }

  _scheduleAudio(buffer) {
    if (!this.playback) return;
    if (!buffer || buffer.byteLength <= 16) return;
    const view = new DataView(buffer);
    const magic = String.fromCharCode(
      view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3),
    );
    if (magic !== AUDIO_MAGIC) return;
    const turnId = view.getUint32(4, true);
    const sampleRate = view.getUint16(12, true) || OUTPUT_SAMPLE_RATE;
    if (this.latestTurnId !== turnId) return;
    if ((buffer.byteLength - 16) % 2 !== 0) return;
    const pcm = new Int16Array(buffer, 16);
    if (pcm.length === 0) return;
    this.playback.schedule(turnId, pcm, sampleRate, (id) => this._maybePlaybackDone(id));
    this.audioChunks += 1;
    this._updateAudioState();
    if (this.playback.context.state !== 'running') {
      this.playback.context.resume().then(() => this._updateAudioState()).catch(() => {});
    }
  }

  _maybePlaybackDone(turnId) {
    if (!this.playback) return;
    if (this.turnEnded !== turnId) return;
    if (this.playbackDoneSent) return;
    if (this.playback.activeCount() > 0) return;
    this.playbackDoneSent = true;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'playback_done', turn_id: turnId }));
    }
  }

  _flushTurn(turnId) {
    if (!this.playback) return;
    const started = performance.now();
    const stopped = this.playback.flush(turnId);
    const elapsed = performance.now() - started;
    if (stopped && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'playback_stopped',
        turn_id: turnId,
        elapsed_ms: Math.round(elapsed * 100) / 100,
      }));
    }
    if (this.latestTurnId === turnId) {
      this.latestTurnId = null;
      this.turnEnded = null;
    }
  }

  _flushAll() {
    if (this.playback) {
      this.playback.stopAll();
    }
  }

  // -- transcript ---------------------------------------------------------
  _turn(turnId) {
    let turn = this.turns.get(turnId);
    if (!turn) {
      turn = { user: null, assistant: null, searching: null, sources: null };
      this.turns.set(turnId, turn);
    }
    return turn;
  }

  _userText(turnId, text) {
    const turn = this._turn(turnId);
    if (!turn.user) {
      turn.user = addMessage('user', text || '…');
    } else {
      turn.user.textContent = text || '…';
    }
    dom.messages.scrollTop = dom.messages.scrollHeight;
  }

  _assistantText(turnId, delta) {
    const turn = this._turn(turnId);
    if (!turn.assistant) {
      turn.assistant = addMessage('assistant', '');
    }
    turn.assistant.textContent += delta;
    dom.messages.scrollTop = dom.messages.scrollHeight;
  }

  _assistantTool(turnId, name, query) {
    const turn = this._turn(turnId);
    if (!turn.assistant) {
      turn.assistant = addMessage('assistant', '');
    }
    if (!turn.searching) {
      const note = document.createElement('span');
      note.className = 'tool-note';
      turn.assistant.parentElement.append(note);
      turn.searching = note;
    }
    const label = name === 'web_search' ? 'Searching the web' : `Calling ${name}`;
    turn.searching.textContent = `${label} for “${query}”…`;
    dom.messages.scrollTop = dom.messages.scrollHeight;
  }

  _assistantSources(turnId, sources) {
    if (!Array.isArray(sources) || sources.length === 0) return;
    const turn = this._turn(turnId);
    if (!turn.assistant) {
      turn.assistant = addMessage('assistant', '');
    }
    if (turn.searching) {
      turn.searching.remove();
      turn.searching = null;
    }
    if (turn.sources) turn.sources.remove();
    const box = document.createElement('div');
    box.className = 'sources';
    const title = document.createElement('span');
    title.className = 'sources-title';
    title.textContent = 'Sources';
    box.append(title);
    let links = 0;
    for (const source of sources) {
      const url = source && typeof source.url === 'string' ? source.url : '';
      if (!/^https?:\/\//i.test(url)) continue;
      const link = document.createElement('a');
      link.href = url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = (source && source.title) || url;
      box.append(link);
      links += 1;
    }
    if (links === 0) return;
    turn.assistant.parentElement.append(box);
    turn.sources = box;
    dom.messages.scrollTop = dom.messages.scrollHeight;
  }

  // -- status polling -----------------------------------------------------
  startPolling() {
    if (this.statusTimer) return;
    this._refreshStatus();
    this.statusTimer = window.setInterval(() => this._refreshStatus(), 2500);
  }

  async _refreshStatus() {
    try {
      const response = await fetch('/api/status', { cache: 'no-store' });
      if (!response.ok) return;
      const status = await response.json();
      ui.renderStatus(status);
      if (!this.wantSession) {
        dom.start.disabled = !status.pipeline_ready;
      }
    } catch (error) {
      /* the service may be restarting; the next poll recovers */
    }
  }
}

const client = new VoiceClient();
dom.start.addEventListener('click', () => client.start());
dom.stop.addEventListener('click', () => client.stop());
dom.volume.addEventListener('input', (event) => client._setVolume(Number(event.target.value) / 100));
dom.mute.addEventListener('click', () => client._toggleMute());
window.addEventListener('beforeunload', () => {
  if (client.wantSession) {
    try {
      client.stop();
    } catch (error) { /* page is unloading */ }
  }
});
client.startPolling();
