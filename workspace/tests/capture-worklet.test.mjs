// SDDTDD SPEC: tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-Review.md (R2).
//
// The AudioWorklet that captures the microphone must emit exact 512-sample
// 16 kHz frames for 44.1 and 48 kHz inputs, including through the shared
// resampler.
const frames = [];
let Processor = null;

globalThis.sampleRate = 48000;
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = { postMessage: (buffer) => frames.push(new Int16Array(buffer)) };
  }
};
globalThis.registerProcessor = (name, implementation) => { Processor = implementation; };

await import('../static/capture-worklet.js');

const failures = [];
function check(name, condition, detail = '') {
  console.log(`${condition ? 'PASS' : 'FAIL'}: ${name}${detail ? ` — ${detail}` : ''}`);
  if (!condition) failures.push(name);
}

for (const inputRate of [44100, 48000]) {
  frames.length = 0;
  const node = new Processor({
    processorOptions: { inputSampleRate: inputRate, targetSampleRate: 16000, frameSamples: 512 },
  });
  const block = 128;
  const totalSamples = inputRate; // 1 second
  for (let offset = 0; offset < totalSamples; offset += block) {
    const channel = new Float32Array(block);
    for (let i = 0; i < block; i += 1) {
      channel[i] = 0.25 * Math.sin((2 * Math.PI * 10000 * (offset + i)) / inputRate);
    }
    node.process([[channel]], [[channel]]);
  }
  const samples = frames.reduce((sum, frame) => sum + frame.length, 0);
  check(`${inputRate} Hz worklet emits exact 512-sample frames`,
        frames.length > 0 && frames.every((frame) => frame.length === 512),
        `${frames.length} frames`);
  check(`${inputRate} Hz worklet emits 16000 samples per second minus at most one frame`,
        samples >= 16000 - 512 && samples <= 16000, `${samples} samples`);
  const rms = Math.sqrt(frames.reduce((sum, frame) => {
    let total = 0;
    for (const value of frame) total += (value / 32768) ** 2;
    return sum + total;
  }, 0) / Math.max(1, samples));
  check(`${inputRate} Hz worklet suppresses aliased 10 kHz content`, rms < 0.15, `rms ${rms.toFixed(3)}`);
}
frames.length = 0;

console.log();
if (failures.length) {
  console.log(`RESULT: FAIL (${failures.length} checks)`);
  process.exit(1);
}
console.log('RESULT: PASS');
