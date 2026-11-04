/**
 * Streaming microphone resampler for the capture worklet and its fallback.
 *
 * The browser microphone usually runs at 48 kHz or 44.1 kHz; the streaming
 * Silero VAD and Whisper need 16 kHz. A naive decimator aliases at 44.1 kHz
 * because the ratio is not an integer (2.75625). This resampler applies a
 * one-pole low-pass filter at the input rate and then linearly interpolates at
 * fractional output positions, so 44.1/48/32/96 kHz streams become exact
 * 16 kHz frames while preserving low frequencies.
 */

const FILTER_CUTOFF_HZ = 4000;

export class Resampler {
  constructor(inputRate, outputRate = 16000, frameSamples = 512) {
    if (!(inputRate > 0) || !(outputRate > 0)) {
      throw new Error('invalid sample rates');
    }
    if (inputRate < outputRate) {
      throw new Error('upsampling is not supported by the capture resampler');
    }
    this.step = inputRate / outputRate;
    this.frameSamples = frameSamples;
    this.frame = new Int16Array(frameSamples);
    this.length = 0;
    // Two cascaded one-pole low-pass stages suppress aliasing before decimation.
    this.alpha = Math.min(1, (2 * Math.PI * FILTER_CUTOFF_HZ) / inputRate);
    this.filterStage1 = 0;
    this.filterStage2 = 0;
    // Absolute input index of buffer[0]; nextOutputIndex is fractional.
    this.inputIndex = 0;
    this.buffer = new Float32Array(0);
    this.nextOutputIndex = 0;
  }

  push(input) {
    const filtered = new Float32Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
      this.filterStage1 += this.alpha * (input[i] - this.filterStage1);
      this.filterStage2 += this.alpha * (this.filterStage1 - this.filterStage2);
      filtered[i] = this.filterStage2;
    }
    if (filtered.length) {
      const merged = new Float32Array(this.buffer.length + filtered.length);
      merged.set(this.buffer, 0);
      merged.set(filtered, this.buffer.length);
      this.buffer = merged;
    }
    return this._drain();
  }

  _drain() {
    const frames = [];
    for (;;) {
      const position = this.nextOutputIndex;
      const base = Math.floor(position);
      const offset = base - this.inputIndex;
      if (offset + 1 >= this.buffer.length) break;
      const fraction = position - base;
      const value = this.buffer[offset] * (1 - fraction) + this.buffer[offset + 1] * fraction;
      const clamped = Math.max(-1, Math.min(1, value));
      this.frame[this.length] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      this.length += 1;
      if (this.length >= this.frameSamples) {
        frames.push(this.frame);
        this.frame = new Int16Array(this.frameSamples);
        this.length = 0;
      }
      this.nextOutputIndex += this.step;
    }
    // Drop samples that can no longer be interpolated (keep one sample back).
    const consumed = Math.max(0, Math.floor(this.nextOutputIndex) - 1 - this.inputIndex);
    if (consumed > 0 && consumed <= this.buffer.length) {
      this.buffer = this.buffer.slice(consumed);
      this.inputIndex += consumed;
    }
    return frames;
  }

  flush() {
    if (this.length === 0) return null;
    const tail = this.frame.slice(0, this.length);
    this.frame = new Int16Array(this.frameSamples);
    this.length = 0;
    return tail;
  }
}
