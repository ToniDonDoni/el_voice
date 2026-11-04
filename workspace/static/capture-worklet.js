/**
 * Capture worklet: microphone audio in, 16 kHz mono signed 16-bit frames out.
 *
 * The AudioContext usually runs at 48 kHz or 44.1 kHz. The shared Resampler
 * low-pass filters and fracionally interpolates to exactly 16 kHz, then emits
 * fixed 512-sample frames (32 ms), the frame size the streaming Silero VAD
 * expects.
 */
import { Resampler } from './resampler.js';

class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.resampler = new Resampler(
      opts.inputSampleRate || sampleRate,
      opts.targetSampleRate || 16000,
      opts.frameSamples || 512,
    );
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) {
      return true;
    }
    for (const frame of this.resampler.push(channel)) {
      this.port.postMessage(frame.buffer, [frame.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCapture);
