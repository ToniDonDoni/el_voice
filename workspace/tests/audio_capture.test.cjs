const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync(new URL("../static/audio-capture.js", `file://${__filename}`), "utf8");

function capture(rate, callbackSizes) {
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
  const input = Float32Array.from({ length: rate }, (_, index) => Math.sin(index / 91) * 0.7);
  let offset = 0;
  let callback = 0;
  while (offset < input.length) {
    const size = Math.min(callbackSizes[callback++ % callbackSizes.length], input.length - offset);
    processor.process([[input.subarray(offset, offset + size)]], []);
    offset += size;
  }
  const emitted = packets.reduce((total, packet) => total + packet.length, 0);
  assert.equal(emitted + processor.output.length, 16000, `${rate} Hz must retain all resampled samples`);
  assert.ok(processor.output.length < 512, "only one partial VAD frame may remain buffered");
  return [...packets.flatMap(packet => [...packet]), ...processor.output];
}

for (const rate of [44100, 48000]) {
  const oneBuffer = capture(rate, [rate]);
  const irregularCallbacks = capture(rate, [127, 256, 511, 1024, 173, 333]);
  assert.deepEqual(irregularCallbacks, oneBuffer, `${rate} Hz callback boundaries must preserve sample continuity`);
}
