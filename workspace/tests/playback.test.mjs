// SDDTDD SPEC: tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-Review.md (R3, R5).
//
// The browser playback queue must tag scheduled audio with a turn id, stop only
// the interrupted turn on a barge-in flush, and stop every scheduled source when
// the WebSocket connection is lost.
import { PlaybackQueue } from '../static/playback.js';

const failures = [];
function check(name, condition, detail = '') {
  console.log(`${condition ? 'PASS' : 'FAIL'}: ${name}${detail ? ` — ${detail}` : ''}`);
  if (!condition) failures.push(name);
}

class FakeSource {
  constructor() {
    this.started = false;
    this.stopped = false;
    this.onended = null;
  }
  connect() { return this; }
  start() { this.started = true; }
  stop() { this.stopped = true; if (this.onended) this.onended(); }
}

class FakeContext {
  constructor() {
    this.currentTime = 0;
    this.destination = {};
    this.created = [];
  }
  createGain() {
    return { gain: { value: 1 }, connect: () => this.destination };
  }
  createBuffer(channels, length, rate) {
    return {
      duration: length / rate,
      getChannelData: () => new Float32Array(length),
    };
  }
  createBufferSource() {
    const source = new FakeSource();
    this.created.push(source);
    return source;
  }
}

function pcm(seconds, rate = 24000) {
  return new Int16Array(Math.round(seconds * rate));
}

const context = new FakeContext();
const queue = new PlaybackQueue(context);

queue.schedule(1, pcm(0.4), 24000);
queue.schedule(1, pcm(0.3), 24000);
queue.schedule(2, pcm(0.5), 24000);

check('all scheduled chunks were started',
      context.created.length === 3 && context.created.every((source) => source.started),
      `${context.created.length} sources`);

queue.flush(1);
const turnOneSources = context.created.slice(0, 2);
const turnTwoSources = context.created.slice(2);
check('flush stops only the interrupted turn',
      turnOneSources.every((source) => source.stopped) &&
      turnTwoSources.every((source) => !source.stopped),
      `${turnOneSources.map((s) => s.stopped)} / ${turnTwoSources.map((s) => s.stopped)}`);
check('flush leaves the other turn queued',
      queue.activeCount() === 1, `${queue.activeCount()}`);

queue.stopAll();
check('connection loss stops every scheduled source',
      context.created.every((source) => source.stopped), 'sources stopped');
check('connection loss leaves an empty queue', queue.activeCount() === 0);

queue.schedule(3, pcm(0.2), 24000);
check('the queue accepts a new turn after a disconnect',
      queue.activeCount() === 1 && context.created.at(-1).started);

console.log();
if (failures.length) {
  console.log(`RESULT: FAIL (${failures.length} checks)`);
  process.exit(1);
}
console.log('RESULT: PASS');
