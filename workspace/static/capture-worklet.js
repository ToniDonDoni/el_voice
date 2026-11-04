/**
 * Capture worklet: microphone audio in, 16 kHz mono signed 16-bit frames out.
 *
 * The AudioContext runs at the device sample rate (usually 48 kHz). This node
 * averages consecutive input samples down to 16 kHz and emits fixed 512-sample
 * frames (32 ms), which is the frame size the streaming Silero VAD expects.
 */
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.ratio = (opts.inputSampleRate || sampleRate) / (opts.targetSampleRate || 16000);
    this.frameSamples = opts.frameSamples || 512;
    this.frame = new Int16Array(this.frameSamples);
    this.length = 0;
    this.accumulator = 0;
    this.count = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) {
      return true;
    }
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
          this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
          this.frame = new Int16Array(this.frameSamples);
          this.length = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCapture);
