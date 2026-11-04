/**
 * Turn-tagged browser playback queue.
 *
 * Audio chunks are scheduled as Web Audio buffer sources. Every source carries
 * the turn id of the answer it belongs to so a barge-in flush stops only the
 * interrupted turn, while a lost WebSocket stops everything that is scheduled
 * (otherwise old speech can keep playing across a reconnect).
 */
export class PlaybackQueue {
  constructor(context, volume = 1) {
    this.context = context;
    this.gain = context.createGain();
    this.gain.gain.value = volume;
    this.gain.connect(context.destination);
    this.sources = new Set();
    this.playHead = context.currentTime;
  }

  setVolume(value) {
    this.gain.gain.value = Math.max(0, Math.min(1, value));
  }

  schedule(turnId, pcm, sampleRate, onEnded) {
    const { context } = this;
    const buffer = context.createBuffer(1, pcm.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let index = 0; index < pcm.length; index += 1) {
      channel[index] = pcm[index] / 32768;
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.gain);
    const startAt = Math.max(context.currentTime + 0.04, this.playHead);
    this.playHead = startAt + buffer.duration;
    const entry = { source, turnId };
    this.sources.add(entry);
    source.onended = () => {
      this.sources.delete(entry);
      if (typeof onEnded === 'function') onEnded(turnId);
    };
    source.start(startAt);
    return source;
  }

  activeCount() {
    return this.sources.size;
  }

  activeForTurn(turnId) {
    let count = 0;
    for (const entry of this.sources) {
      if (entry.turnId === turnId) count += 1;
    }
    return count;
  }

  flush(turnId) {
    let stopped = false;
    for (const entry of Array.from(this.sources)) {
      if (entry.turnId !== turnId) continue;
      try {
        entry.source.stop();
      } catch (error) { /* already finished */ }
      this.sources.delete(entry);
      stopped = true;
    }
    if (stopped) {
      this.playHead = this.context.currentTime + 0.05;
    }
    return stopped;
  }

  stopAll() {
    let stopped = false;
    for (const entry of Array.from(this.sources)) {
      try {
        entry.source.stop();
      } catch (error) { /* already finished */ }
      this.sources.delete(entry);
      stopped = true;
    }
    this.playHead = this.context.currentTime + 0.05;
    return stopped;
  }
}
