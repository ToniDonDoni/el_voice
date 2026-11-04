class PCM16CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions?.targetRate || 16000;
    this.step = sampleRate / this.targetRate;
    this.inputIndex = 0;
    this.previousIndex = -1;
    this.previousSample = 0;
    this.nextPosition = 0;
    this.output = [];
    this.frameSize = 512;
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (output) output.fill(0);
    if (!input) return true;

    for (let offset = 0; offset < input.length; offset += 1) {
      const currentIndex = this.inputIndex + offset;
      const currentSample = input[offset];
      if (this.previousIndex < 0) {
        this.previousIndex = currentIndex;
        this.previousSample = currentSample;
        this.nextPosition = currentIndex;
      }
      while (this.nextPosition <= currentIndex) {
        const width = currentIndex - this.previousIndex;
        const blend = width > 0 ? (this.nextPosition - this.previousIndex) / width : 1;
        const value = this.previousSample + (currentSample - this.previousSample) * blend;
        this.output.push(Math.max(-1, Math.min(1, value)));
        this.nextPosition += this.step;
        if (this.output.length === this.frameSize) this.flushFrame();
      }
      this.previousIndex = currentIndex;
      this.previousSample = currentSample;
    }
    this.inputIndex += input.length;
    return true;
  }

  flushFrame() {
    const pcm = new Int16Array(this.frameSize);
    for (let i = 0; i < this.frameSize; i += 1) {
      const sample = this.output[i];
      pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    this.output.length = 0;
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
  }
}

registerProcessor("pcm16-capture", PCM16CaptureProcessor);
