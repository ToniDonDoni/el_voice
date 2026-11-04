const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync(new URL("../static/audio-capture.js", `file://${__filename}`), "utf8");

for (const rate of [44100, 48000]) {
  let Processor;
  const packets = [];
  class AudioWorkletProcessor {
    constructor() {
      this.port = { postMessage(buffer) { packets.push(new Int16Array(buffer)); } };
    }
  }
  vm.runInNewContext(source, {
    AudioWorkletProcessor,
    sampleRate: rate,
    registerProcessor(_name, processor) { Processor = processor; },
  });
  const processor = new Processor({ processorOptions: { targetRate: 16000 } });
  const input = new Float32Array(rate).fill(0.25);
  let offset = 0;
  const callbackSizes = [127, 256, 511, 1024, 173, 333];
  let callback = 0;
  while (offset < input.length) {
    const size = Math.min(callbackSizes[callback++ % callbackSizes.length], input.length - offset);
    processor.process([[input.subarray(offset, offset + size)]], []);
    offset += size;
  }
  const emitted = packets.reduce((total, packet) => total + packet.length, 0);
  assert.equal(emitted + processor.output.length, 16000, `${rate} Hz must retain all resampled samples`);
  assert.ok(processor.output.length < 512, "only one partial VAD frame may remain buffered");
}
