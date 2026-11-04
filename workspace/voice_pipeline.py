"""Local model loading, streaming VAD, and inference helpers for EL Voice."""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

MODEL_ROOT = Path(os.environ.get("EL_VOICE_MODELS", "/models")).resolve()
MODEL_TOOL = Path("/opt/el-voice/download_models.py")
WORKSPACE = Path(__file__).resolve().parent
SETTINGS_PATH = WORKSPACE / "settings.json"
SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 512
VAD_FRAME_SECONDS = VAD_FRAME_SAMPLES / SAMPLE_RATE


def _load_settings() -> dict[str, Any]:
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def _load_manifest() -> dict[str, Any]:
    return json.loads(Path("/opt/el-voice/models.json").read_text(encoding="utf-8"))


def verify_models() -> None:
    """Verify publisher assets and local receipts without loading model code."""
    if not MODEL_TOOL.is_file():
        raise RuntimeError("The model verification tool is unavailable")
    result = subprocess.run(
        [sys.executable, str(MODEL_TOOL), "--verify"],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError("One or more configured model assets failed verification")


class StreamingSileroVAD:
    """Run the bundled Silero ONNX VAD sequentially on 512-sample frames."""

    def __init__(self, config: dict[str, Any]):
        import onnxruntime as ort

        asset = importlib.resources.files("faster_whisper").joinpath(
            "assets", "silero_vad_v6.onnx"
        )
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 4
        self.session = ort.InferenceSession(
            str(asset), providers=["CPUExecutionProvider"], sess_options=options
        )
        self.threshold = float(config.get("threshold", 0.55))
        self.negative_threshold = float(config.get("negative_threshold", 0.35))
        self.start_frames = int(config.get("start_frames", 3))
        self.end_silence_ms = int(config.get("end_silence_ms", 640))
        self.min_speech_ms = int(config.get("min_speech_ms", 240))
        self.pre_roll_ms = int(config.get("pre_roll_ms", 320))
        self.max_utterance_seconds = int(config.get("max_utterance_seconds", 45))
        self.pre_roll = deque(maxlen=max(1, math.ceil(self.pre_roll_ms / (1000 * VAD_FRAME_SECONDS))))
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(64, dtype=np.float32)
        self.candidate_frames: list[np.ndarray] = []
        self.utterance_frames: list[np.ndarray] = []
        self.speech_frames = 0
        self.silence_frames = 0
        self.active = False

    def probability(self, frame: np.ndarray) -> float:
        if frame.shape != (VAD_FRAME_SAMPLES,):
            raise ValueError("Audio frame must contain exactly 512 mono samples")
        model_input = np.concatenate((self.context, frame)).astype(np.float32, copy=False)
        outputs = self.session.run(
            None,
            {
                "input": model_input.reshape(1, -1),
                "h": self.h,
                "c": self.c,
            },
        )
        probability, self.h, self.c = outputs
        self.context = frame[-64:].copy()
        return float(np.asarray(probability).reshape(-1)[0])

    def accept(self, frame: np.ndarray) -> tuple[str | None, np.ndarray | None, float]:
        """Return onset/end events while retaining a pre-roll and endpoint tail."""
        frame = np.asarray(frame, dtype=np.float32).copy()
        score = self.probability(frame)
        if not self.active:
            self.pre_roll.append(frame)
            if score >= self.threshold:
                self.candidate_frames.append(frame)
                if len(self.candidate_frames) >= self.start_frames:
                    self.active = True
                    self.utterance_frames = list(self.pre_roll)
                    self.speech_frames = len(self.candidate_frames)
                    self.candidate_frames.clear()
                    self.silence_frames = 0
                    return "start", None, score
            else:
                self.candidate_frames.clear()
            return None, None, score

        self.utterance_frames.append(frame)
        if score < self.negative_threshold:
            self.silence_frames += 1
        else:
            self.silence_frames = 0
            self.speech_frames += 1

        max_frames = max(1, int(self.max_utterance_seconds / VAD_FRAME_SECONDS))
        end_frames = max(1, math.ceil(self.end_silence_ms / (1000 * VAD_FRAME_SECONDS)))
        if len(self.utterance_frames) >= max_frames or self.silence_frames >= end_frames:
            frames = self.utterance_frames
            enough_speech = self.speech_frames * VAD_FRAME_SECONDS >= self.min_speech_ms / 1000
            if self.silence_frames:
                # Keep a short tail after the last speech frame for Whisper's decoder.
                keep = max(1, math.ceil(160 / (1000 * VAD_FRAME_SECONDS)))
                frames = frames[: max(1, len(frames) - self.silence_frames + keep)]
            audio = np.concatenate(frames).astype(np.float32, copy=False)
            min_samples = int(self.min_speech_ms * SAMPLE_RATE / 1000)
            self.active = False
            self.utterance_frames = []
            self.candidate_frames.clear()
            self.pre_roll.clear()
            self.silence_frames = 0
            self.speech_frames = 0
            if enough_speech and audio.size >= min_samples:
                return "end", audio, score
            return "empty", None, score
        return None, None, score


class CancelOnToken:
    """Make llama.cpp emit EOS as soon as cancellation is observed."""

    def __init__(self, cancel_event: threading.Event, eos_token: int):
        self.cancel_event = cancel_event
        self.eos_token = eos_token

    def __call__(self, _input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if self.cancel_event.is_set():
            scores.fill(-np.inf)
            scores[self.eos_token] = 0.0
        return scores


@dataclass
class Turn:
    turn_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    speech_onset_at: float = field(default_factory=time.monotonic)
    speech_end_at: float | None = None
    playback_started: bool = False


class VoiceRuntime:
    def __init__(self):
        self.config = _load_settings()
        self.status: dict[str, Any] = {
            "stage": "loading",
            "ready": False,
            "models": {"vad": "loading", "stt": "loading", "llm": "loading", "tts": "loading"},
            "error": None,
        }
        self.whisper = None
        self.llm = None
        self.tts = None
        self.torch = None
        self.llm_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.metrics: deque[dict[str, float]] = deque(maxlen=100)
        self.vad_config = self.config.get("vad", {})

    async def initialize(self) -> None:
        try:
            await asyncio.to_thread(self._load_models)
            self.status.update(stage="ready", ready=True, error=None)
        except Exception as exc:
            self.status.update(stage="error", ready=False, error=type(exc).__name__)
            logging.getLogger("el_voice").exception(
                "Local model initialization failed (%s): %s", type(exc).__name__, str(exc)[:500]
            )

    def _load_models(self) -> None:
        verify_models()
        # Keep the import order used by the environment check to avoid native OpenBLAS conflicts.
        import torch

        self.torch = torch
        tts_config = self.config["tts"]
        torch.set_num_threads(int(tts_config.get("cpu_threads", 4)))
        torch.set_num_interop_threads(1)

        self.status["models"]["vad"] = "loading"
        self.vad_probe = StreamingSileroVAD(self.vad_config)
        self.status["models"]["vad"] = "ready"

        self.status["models"]["stt"] = "loading"
        from faster_whisper import WhisperModel

        stt_path = MODEL_ROOT / self.config["stt"]["model"]
        self.whisper = WhisperModel(
            str(stt_path),
            device="cpu",
            compute_type="int8",
            cpu_threads=int(self.config.get("stt", {}).get("cpu_threads", 4)),
            num_workers=1,
            local_files_only=True,
        )
        self.status["models"]["stt"] = "ready"

        self.status["models"]["llm"] = "loading"
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=str(MODEL_ROOT / self.config["llm"]["model"]),
            n_ctx=int(self.config.get("llm", {}).get("context_tokens", 3072)),
            n_threads=int(self.config.get("llm", {}).get("cpu_threads", 4)),
            n_batch=int(self.config["llm"].get("batch_tokens", 256)),
            verbose=False,
        )
        self.status["models"]["llm"] = "ready"

        self.status["models"]["tts"] = "loading"
        tts_path = MODEL_ROOT / tts_config["model"]
        self.tts = torch.package.PackageImporter(str(tts_path)).load_pickle("tts_models", "model")
        self.tts.to(torch.device("cpu"))
        if tts_config["speaker"] not in self.tts.speakers:
            raise RuntimeError("Configured TTS speaker is unavailable")
        self.status["models"]["tts"] = "ready"

    def new_vad(self) -> StreamingSileroVAD:
        return StreamingSileroVAD(self.vad_config)

    def transcribe(self, audio: np.ndarray) -> str:
        if self.whisper is None:
            raise RuntimeError("STT is not ready")

        def run() -> str:
            segments, _info = self.whisper.transcribe(
                audio,
                language=self.config["stt"]["language"],
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                without_timestamps=True,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

        return run()

    def llm_stream(
        self,
        user_text: str,
        cancel_event: threading.Event,
        history: list[dict[str, str]],
    ):
        if self.llm is None:
            raise RuntimeError("LLM is not ready")
        system_prompt = (
            "You are a concise Russian-speaking Dota 2 companion. Answer in Russian unless "
            "the user asks otherwise. Focus on Dota 2 strategy, heroes, items, and game concepts. "
            "Do not claim live match data or current patch facts you cannot verify. Keep replies "
            "natural and short enough to speak aloud; avoid markdown and lists unless useful."
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_text})
        processor = CancelOnToken(cancel_event, self.llm.token_eos())
        with self.llm_lock:
            stream = self.llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=0.55,
                top_p=0.9,
                max_tokens=int(self.config["llm"].get("max_response_tokens", 192)),
                logits_processor=[processor],
            )
            try:
                for item in stream:
                    if cancel_event.is_set():
                        break
                    content = item.get("choices", [{}])[0].get("delta", {}).get("content")
                    if content:
                        yield content
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()

    def synthesize(self, text: str) -> tuple[bytes, int]:
        if self.tts is None:
            raise RuntimeError("TTS is not ready")
        with self.tts_lock, self.torch.inference_mode():
            audio = self.tts.apply_tts(
                text=text,
                speaker=self.config["tts"]["speaker"],
                sample_rate=int(self.config["tts"]["sample_rate"]),
            )
        audio = audio.detach().cpu().float().flatten()
        if audio.numel() == 0 or not bool(self.torch.isfinite(audio).all()):
            raise RuntimeError("TTS returned invalid audio")
        pcm = (audio.clamp(-1, 1) * 32767).to(self.torch.int16).numpy().astype("<i2", copy=False)
        return pcm.tobytes(), int(self.config["tts"]["sample_rate"])

    def status_snapshot(self) -> dict[str, Any]:
        timings = list(self.metrics)
        response = [entry["response_ms"] for entry in timings if "response_ms" in entry]
        interrupt = [entry["interrupt_ms"] for entry in timings if "interrupt_ms" in entry]

        def percentiles(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"p50": None, "p95": None}
            return {
                "p50": round(float(np.percentile(values, 50)), 1),
                "p95": round(float(np.percentile(values, 95)), 1),
            }

        return {
            **self.status,
            "system": "Linux",
            "architecture": "aarch64",
            "model_file_count": len(_load_manifest()["assets"]) if self.status["ready"] else None,
            "vad": {"type": "Silero ONNX bundled with faster-whisper", **self.vad_config},
            "metrics": {
                "response_samples": len(response),
                "interrupt_samples": len(interrupt),
                "speech_end_to_first_audio_ms": percentiles(response),
                "speech_onset_to_playback_stopped_ms": percentiles(interrupt),
            },
        }


@dataclass
class Outbound:
    event: dict[str, Any] | None = None
    audio: bytes | None = None
    turn_id: str | None = None
