// SDDTDD SPEC: tasks/SPEC-001-LOCAL-VOICE-CONVERSATION-Review.md (R2).
//
// Streaming resampler contract for the browser capture path. The worklet must
// turn a 44.1/48 kHz microphone stream into exact 16 kHz, 512-sample frames
// without losing samples or mangling frequencies.
//
// Run:  node tests/resampler.test.mjs
import { Resampler } from '../static/resampler.js';

const failures = [];
function check(name, condition, detail = '') {
  console.log(`${condition ? 'PASS' : 'FAIL'}: ${name}${detail ? ` — ${detail}` : ''}`);
  if (!condition) failures.push(name);
}

function run(inputRate, samples, chunk = samples.length, frameSamples = 512) {
  const resampler = new Resampler(inputRate, 16000, frameSamples);
  const out = [];
  const frames = [];
  for (let offset = 0; offset < samples.length; offset += chunk) {
    for (const frame of resampler.push(samples.subarray(offset, offset + chunk))) {
      frames.push(frame);
      out.push(...frame);
    }
  }
  const tail = resampler.flush();
  if (tail) out.push(...tail);
  return { out, frames, tail };
}

function sine(rate, hz, seconds) {
  const samples = new Float32Array(Math.round(rate * seconds));
  for (let i = 0; i < samples.length; i += 1) samples[i] = 0.5 * Math.sin((2 * Math.PI * hz * i) / rate);
  return samples;
}

function frequencyOf(samples, rate) {
  let crossings = 0;
  for (let i = 1; i < samples.length; i += 1) {
    if (samples[i - 1] <= 0 && samples[i] > 0) crossings += 1;
  }
  return (crossings * rate) / samples.length;
}

for (const inputRate of [48000, 44100, 32000, 96000]) {
  const seconds = 2;
  const { out, frames, tail } = run(inputRate, sine(inputRate, 1000, seconds));
  const expected = 16000 * seconds;
  check(`${inputRate} Hz input produces 16 kHz sample count`,
        Math.abs(out.length - expected) <= 2, `${out.length} vs ${expected}`);
  check(`${inputRate} Hz output is delivered in 512-sample frames`,
        frames.every((frame) => frame.length === 512), `${frames.length} frames`);
  check(`${inputRate} Hz flush returns only the partial tail`,
        (tail === null || tail.length < 512),
        tail ? `${tail.length} tail samples` : 'no tail');
  const frequency = frequencyOf(out, 16000);
  check(`${inputRate} Hz input keeps a 1 kHz tone near 1 kHz`,
        Math.abs(frequency - 1000) < 20, `${frequency.toFixed(1)} Hz`);
  const peak = out.reduce((max, value) => Math.max(max, Math.abs(value)), 0) / 32768;
  check(`${inputRate} Hz input keeps the signal amplitude`,
        peak > 0.3 && peak <= 0.6, `peak ${peak.toFixed(3)}`);
}

// A 60 s 44.1 kHz capture must produce 960,000 samples at 16 kHz.
{
  const { out } = run(44100, sine(44100, 440, 60), 44100);
  check('60 s of 44.1 kHz capture yields 960000 samples',
        Math.abs(out.length - 960000) <= 2, `${out.length}`);
}

// The anti-alias filter must suppress content above the 8 kHz Nyquist limit.
{
  const { out } = run(44100, sine(44100, 10000, 1), 1000);
  const rms = Math.sqrt(out.reduce((sum, value) => sum + (value / 32768) ** 2, 0) / out.length);
  check('10 kHz content is suppressed before 16 kHz decimation',
        rms < 0.15, `rms ${rms.toFixed(3)}`);
}

// Constant signal must survive (no low-frequency loss).
{
  const dc = new Float32Array(48000).fill(0.25);
  const { out } = run(48000, dc);
  const mean = out.reduce((sum, value) => sum + value, 0) / out.length / 32768;
  check('DC signal survives resampling', Math.abs(mean - 0.25) < 0.01, `mean ${mean.toFixed(3)}`);
}

// Chunking must not change the produced audio stream.
{
  const samples = sine(44100, 440, 1);
  const whole = run(44100, samples, samples.length).out;
  const chunked = run(44100, samples, 100).out;
  const same = whole.length === chunked.length && whole.every((value, i) => value === chunked[i]);
  check('output is independent of input chunking', same, `${whole.length} vs ${chunked.length}`);
}

// Frames are delivered in exact 512-sample blocks and the tail is flushed.
{
  const { frames, tail } = run(48000, sine(48000, 500, 0.7), 1234);
  check('chunked input still delivers whole 512-sample frames',
        frames.every((frame) => frame.length === 512), `${frames.length} frames`);
  check('tail flush returns the remaining samples',
        tail === null || tail.length < 512, tail ? `${tail.length}` : 'no tail');
}

console.log();
if (failures.length) {
  console.log(`RESULT: FAIL (${failures.length} checks)`);
  process.exit(1);
}
console.log('RESULT: PASS');
