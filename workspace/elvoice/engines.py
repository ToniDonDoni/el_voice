"""Model lifecycle, readiness reporting and local CPU inference.

Everything runs on CPU inside the container. Model files are verified against
their download receipts before an engine is allowed to open them. Initialization
runs on a background thread so the HTTP process is live (and can report
readiness) before the several-hundred-megabyte models finish loading.

Import order matters: PyTorch is imported first, as documented in
``tools/check_environment.py``, so that its bundled OpenBLAS registers before
other native inference libraries load a library with the same SONAME.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from pathlib import Path

from . import models

logger = logging.getLogger("elvoice.engines")

COMPONENTS = ("vad", "stt", "llm", "tts")


class EngineError(RuntimeError):
    """A model is unavailable or inference failed."""


class EngineManager:
    """Owns the loaded models and serialises the CPU-heavy inference calls."""

    def __init__(self, cfg: dict, root: Path):
        self._cfg = cfg
        self._root = Path(root)
        self._components: dict[str, dict] = {
            name: {"status": "pending", "detail": ""} for name in COMPONENTS
        }
        self._init_error: str | None = None
        self._initialized = False
        self._ready: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._vad = None
        self._stt = None
        self._llm = None
        self._tts_model = None
        self._torch = None
        self._np = None
        self._llm_lock = threading.Lock()
        self._stt_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._tts_executor: concurrent.futures.ThreadPoolExecutor | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Start loading models in the background; call with a running event loop."""
        self._loop = asyncio.get_running_loop()
        self._ready = asyncio.Event()
        self._thread = threading.Thread(target=self._initialize, name="engine-init", daemon=True)
        self._thread.start()

    @property
    def is_ready(self) -> bool:
        return (
            self._initialized
            and self._init_error is None
            and all(component["status"] == "ready" for component in self._components.values())
        )

    @property
    def vad(self):
        """The loaded streaming VAD; its recurrent state is reset per session."""
        if self._vad is None:
            raise EngineError("voice activity detection is not ready")
        return self._vad

    def status(self) -> dict:
        return {
            "ready": self.is_ready,
            "error": self._init_error,
            "components": {name: dict(value) for name, value in self._components.items()},
        }

    def _mark(self, name: str, status: str, detail: str = "") -> None:
        self._components[name] = {"status": status, "detail": detail}

    def _initialize(self) -> None:
        logger.info("engine initialization started")
        try:
            self._load_all()
        except Exception as exc:  # noqa: BLE001 - readiness must report the failure
            self._init_error = f"{type(exc).__name__}: {exc}"
            for name, component in self._components.items():
                if component["status"] != "ready":
                    self._mark(name, "error", self._init_error)
        finally:
            if self._init_error:
                logger.error("engine initialization failed: %s", self._init_error)
            else:
                logger.info("engine initialization complete: all components ready")
            try:
                if self._loop is not None and self._ready is not None:
                    self._loop.call_soon_threadsafe(self._ready.set)
            except RuntimeError:  # event loop already closed during shutdown
                pass

    async def wait_ready(self, timeout: float = 180.0) -> None:
        if self._ready is None:
            raise EngineError("model initialization has not started")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise EngineError("timed out waiting for model initialization") from exc
        if self._init_error:
            raise EngineError(self._init_error)
        if not self.is_ready:
            raise EngineError("model initialization did not complete")

    # -- model loading -----------------------------------------------------
    def _load_all(self) -> None:
        # Torch must be imported before the other native inference packages.
        import torch  # noqa: F401 - imported first on purpose

        import numpy as np

        self._torch = torch
        self._np = np
        torch.set_num_threads(int(self._cfg["tts"]["cpu_threads"]))

        self._load_vad()
        self._load_tts()
        self._load_stt()
        self._load_llm()

        self._stt_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="elvoice-stt"
        )
        self._tts_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="elvoice-tts"
        )
        self._warm_up()
        self._initialized = True

    def _load_vad(self) -> None:
        from .vad import SileroVad

        started = time.perf_counter()
        path = models.verify_asset(self._root, self._cfg["vad"]["model"])
        self._vad = SileroVad(
            str(path),
            sample_rate=int(self._cfg["audio"]["input_sample_rate"]),
            threads=1,
        )
        self._mark("vad", "ready")
        logger.info("model vad ready in %.1f s", time.perf_counter() - started)

    def _load_tts(self) -> None:
        torch = self._torch
        started = time.perf_counter()
        path = models.verify_asset(self._root, self._cfg["tts"]["model"])
        package = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        package.to(torch.device("cpu"))
        speaker = self._cfg["tts"]["speaker"]
        if speaker not in package.speakers:
            raise EngineError(f"configured Silero speaker is absent from the model: {speaker}")
        self._tts_model = package
        self._mark("tts", "ready")
        logger.info("model tts ready in %.1f s", time.perf_counter() - started)

    def _load_stt(self) -> None:
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        models.verify_whisper_files(self._root)
        stt_cfg = self._cfg["stt"]
        self._stt = WhisperModel(
            str(self._root / stt_cfg["model_dir"]),
            device="cpu",
            compute_type="int8",
            cpu_threads=int(stt_cfg["cpu_threads"]),
        )
        self._mark("stt", "ready")
        logger.info("model stt ready in %.1f s", time.perf_counter() - started)

    def _load_llm(self) -> None:
        from llama_cpp import Llama

        llm_cfg = self._cfg["llm"]
        started = time.perf_counter()
        path = models.verify_asset(self._root, llm_cfg["model"])
        self._llm = Llama(
            model_path=str(path),
            n_ctx=int(llm_cfg["n_ctx"]),
            n_batch=int(llm_cfg["n_batch"]),
            n_threads=int(llm_cfg["n_threads"]),
            verbose=False,
        )
        self._mark("llm", "ready")
        logger.info("model llm ready in %.1f s", time.perf_counter() - started)

    def _warm_up(self) -> None:
        """Run each engine once so lazy init is not paid on the first real turn."""
        started = time.perf_counter()
        try:
            # Long enough to warm the TTS path for a full first phrase.
            self._synthesize_sync("Привет! Это голосовой помощник по Dota 2.")
            self._stt.transcribe(
                self._np.zeros(8000, dtype=self._np.float32),
                language=self._cfg["stt"]["language"],
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
            )
            # Same system prompt as a real turn to warm prompt evaluation.
            self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self._cfg["llm"]["system_prompt"]},
                    {"role": "user", "content": "Привет"},
                ],
                max_tokens=1,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - warm-up must not block readiness
            logger.warning("engine warm-up skipped: %s", type(exc).__name__)
        logger.info("engine warm-up finished in %.1f s", time.perf_counter() - started)

    # -- inference ---------------------------------------------------------
    async def transcribe(self, pcm: bytes) -> str:
        """Final Russian transcription of one complete utterance (VAD-filter free)."""
        if self._stt is None or self._stt_executor is None:
            raise EngineError("speech recognition is not ready")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._stt_executor, self._transcribe_sync, pcm)

    def _transcribe_sync(self, pcm: bytes) -> str:
        np = self._np
        cfg = self._cfg["stt"]
        started = time.perf_counter()
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _info = self._stt.transcribe(
            audio,
            language=cfg["language"],
            beam_size=int(cfg["beam_size"]),
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        logger.info(
            "stt: %.0f ms inference for %.2f s audio",
            (time.perf_counter() - started) * 1000.0, len(audio) / 16000.0,
        )
        return " ".join(text.split())

    async def synthesize(self, text: str) -> bytes:
        """Synthesise one speakable phrase to 16-bit little-endian PCM."""
        if self._tts_model is None or self._tts_executor is None:
            raise EngineError("speech synthesis is not ready")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tts_executor, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        torch = self._torch
        np = self._np
        started = time.perf_counter()
        tts_cfg = self._cfg["tts"]
        with torch.inference_mode():
            audio = self._tts_model.apply_tts(
                text=text,
                speaker=tts_cfg["speaker"],
                sample_rate=int(tts_cfg["sample_rate"]),
            )
        array = audio.detach().to(dtype=torch.float32).clamp(-1.0, 1.0).numpy()
        logger.info(
            "tts: %.0f ms synthesis for %.2f s audio",
            (time.perf_counter() - started) * 1000.0, len(array) / 24000.0,
        )
        return (array * 32767.0).astype(np.int16).tobytes()

    async def stream_chat(self, messages: list[dict], cancel: threading.Event):
        """Yield response text deltas from the local model.

        Generation runs on its own thread because llama.cpp generation is
        blocking CPU work. The cancellation check happens between tokens, so a
        barge-in stops inference after at most one token; the underlying engine
        call for that token is never scheduled again. This bounds the active
        inference rather than only cancelling an awaiting coroutine.
        """
        if self._llm is None:
            raise EngineError("language model is not ready")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        llm_cfg = self._cfg["llm"]

        def _deliver(item) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass

        def _worker() -> None:
            try:
                with self._llm_lock:
                    if cancel.is_set():
                        return
                    generation = self._llm.create_chat_completion(
                        messages=messages,
                        stream=True,
                        max_tokens=int(llm_cfg["max_tokens"]),
                        temperature=float(llm_cfg["temperature"]),
                        top_p=float(llm_cfg["top_p"]),
                        repeat_penalty=float(llm_cfg["repeat_penalty"]),
                    )
                    first_token = True
                    try:
                        for chunk in generation:
                            if cancel.is_set():
                                break
                            choice = chunk["choices"][0]
                            delta = choice.get("delta") or {}
                            text = delta.get("content") or ""
                            if text:
                                if first_token:
                                    first_token = False
                                    logger.info("llm: first token delivered")
                                _deliver(("token", text))
                    finally:
                        close = getattr(generation, "close", None)
                        if callable(close):
                            close()
            except Exception as exc:  # noqa: BLE001 - surfaced to the turn
                _deliver(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                _deliver(("end", None))

        threading.Thread(target=_worker, name="elvoice-llm", daemon=True).start()
        while True:
            kind, value = await queue.get()
            if kind == "end":
                return
            if kind == "error":
                raise EngineError(value)
            yield value
