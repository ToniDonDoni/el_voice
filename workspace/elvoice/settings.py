"""Load and validate the settings file mounted with the application source."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"
MODEL_ROOT_ENV = "EL_VOICE_MODELS"
DATA_ROOT_ENV = "EL_VOICE_DATA"
LLM_SERVICE_ENV = "EL_VOICE_LLM_URL"
DEFAULT_LLM_SERVICE_URL = "ws://llm:8100"


def model_root() -> Path:
    """Directory with the downloaded publisher assets (read-only in the container)."""
    return Path(os.environ.get(MODEL_ROOT_ENV, "/models"))


def data_root() -> Path:
    """Writable runtime directory (cache only, no audio or transcripts)."""
    return Path(os.environ.get(DATA_ROOT_ENV, "/data"))


def llm_service_url() -> str:
    """Internal WebSocket URL of the LLM service (never a provider URL)."""
    return os.environ.get(LLM_SERVICE_ENV, DEFAULT_LLM_SERVICE_URL).rstrip("/")


def _number(block: dict, key: str, where: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    value = block.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"settings: {where}.{key} must be a number")
    if minimum is not None and value < minimum:
        raise ValueError(f"settings: {where}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"settings: {where}.{key} must be <= {maximum}")
    return value


def _integer(block: dict, key: str, where: str, minimum: int, maximum: int) -> int:
    value = block.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"settings: {where}.{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"settings: {where}.{key} must be in [{minimum}, {maximum}]")
    return value


def _text(block: dict, key: str, where: str) -> str:
    value = block.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"settings: {where}.{key} must be a non-empty string")
    return value


def _block(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"settings: missing object '{key}'")
    return value


def validate_settings(raw: dict) -> dict:
    """Validate cross-component constraints and return the configuration."""
    audio = _block(raw, "audio")
    conversation = _block(raw, "conversation")
    vad = _block(raw, "vad")
    stt = _block(raw, "stt")
    llm = _block(raw, "llm")
    tts = _block(raw, "tts")

    input_rate = _integer(audio, "input_sample_rate", "audio", 8000, 48000)
    output_rate = _integer(audio, "output_sample_rate", "audio", 8000, 48000)
    frame_samples = _integer(audio, "vad_frame_samples", "audio", 256, 4096)
    pre_roll_ms = _number(audio, "pre_roll_ms", "audio", 0, 2000)
    max_utterance_ms = _number(audio, "max_utterance_ms", "audio", 1000, 120000)

    if input_rate != 16000:
        raise ValueError("settings: the streaming Silero VAD requires 16000 Hz input audio")
    if frame_samples != 512:
        raise ValueError("settings: the streaming Silero VAD requires 512-sample frames")
    if tts.get("sample_rate") != output_rate:
        raise ValueError("settings: audio.output_sample_rate must equal tts.sample_rate")
    if pre_roll_ms < 0 or pre_roll_ms > max_utterance_ms:
        raise ValueError("settings: audio.pre_roll_ms must not exceed the utterance limit")

    _integer(conversation, "history_turns", "conversation", 0, 20)
    _integer(conversation, "max_prompt_chars", "conversation", 500, 20000)

    threshold = _number(vad, "threshold", "vad", 0.01, 0.99)
    neg_threshold = _number(vad, "neg_threshold", "vad", 0.0, 0.99)
    if neg_threshold >= threshold:
        raise ValueError("settings: vad.neg_threshold must be lower than vad.threshold")
    min_speech_ms = _number(vad, "min_speech_ms", "vad", 0, max_utterance_ms)
    end_silence_ms = _number(vad, "end_silence_ms", "vad", 100, 10000)
    if min_speech_ms < frame_samples / input_rate * 1000:
        raise ValueError("settings: vad.min_speech_ms must cover at least one frame")
    _text(vad, "model", "vad")

    _integer(stt, "beam_size", "stt", 1, 5)
    _integer(stt, "cpu_threads", "stt", 1, 32)
    _text(stt, "language", "stt")
    _text(stt, "model_dir", "stt")

    _integer(llm, "n_ctx", "llm", 512, 131072)
    _integer(llm, "n_batch", "llm", 32, 4096)
    _integer(llm, "n_threads", "llm", 1, 32)
    _integer(llm, "max_tokens", "llm", 1, 2048)
    _number(llm, "temperature", "llm", 0.0, 2.0)
    _number(llm, "top_p", "llm", 0.0, 1.0)
    _number(llm, "repeat_penalty", "llm", 0.0, 4.0)
    _text(llm, "system_prompt", "llm")
    _text(llm, "model", "llm")

    _integer(tts, "cpu_threads", "tts", 1, 32)
    min_chunk = _integer(tts, "min_chunk_chars", "tts", 1, 200)
    soft_chunk = _integer(tts, "soft_chunk_chars", "tts", min_chunk, 400)
    max_chunk = _integer(tts, "max_chunk_chars", "tts", soft_chunk, 400)
    _text(tts, "speaker", "tts")
    _text(tts, "model", "tts")

    return raw


def load_settings(path: Path | str | None = None) -> dict:
    """Read and validate the settings file; configuration errors abort startup."""
    source = Path(path) if path else DEFAULT_SETTINGS_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"settings: cannot read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"settings: invalid JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("settings: the root value must be an object")
    return validate_settings(raw)
