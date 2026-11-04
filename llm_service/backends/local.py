"""Local Gemma backend using llama-cpp-python.

The model is the same pinned GGUF used by SPEC-001, verified against its
download receipt before loading. Generation runs in a dedicated worker thread
because llama.cpp is blocking CPU code; the server-facing async iterator is fed
through a queue. The shared ``threading.Event`` is checked between tokens, so a
barge-in stops CPU generation after at most one token instead of only cancelling
an awaiting coroutine.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from ..config import ServiceConfig
from .base import BackendError

logger = logging.getLogger("llm_service.local")


class LocalBackend:
    mode = "local"

    def __init__(self, config: ServiceConfig):
        self._config = config
        self._llm = None
        self._load_error: str | None = None
        self._model_lock = threading.Lock()
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        self._started = True
        try:
            await asyncio.to_thread(self._load)
        except Exception as exc:  # noqa: BLE001 - readiness reports the failure
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error("local model load failed: %s", type(exc).__name__)
            raise

    def _load(self) -> None:
        from elvoice import models

        llm_cfg = self._config.llm_settings
        started = time.perf_counter()
        path = models.verify_asset(self._config.models_root, llm_cfg["model"])
        from llama_cpp import Llama  # noqa: PLC0415 - only local mode needs llama.cpp

        self._llm = Llama(
            model_path=str(path),
            n_ctx=int(llm_cfg["n_ctx"]),
            n_batch=int(llm_cfg["n_batch"]),
            n_threads=int(llm_cfg["n_threads"]),
            verbose=False,
        )
        self._load_error = None
        # A one-token warm-up pays prompt evaluation before the first real turn.
        try:
            self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self._config.system_prompt},
                    {"role": "user", "content": "Привет"},
                ],
                max_tokens=1,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - warm-up must not block readiness
            logger.warning("local model warm-up skipped: %s", type(exc).__name__)
        logger.info(
            "local model ready in %.1f s (%s)",
            time.perf_counter() - started,
            llm_cfg["model"],
        )

    async def close(self) -> None:
        self._llm = None

    def status(self) -> dict:
        return {
            "mode": "local",
            "ready": self._llm is not None,
            "error": self._load_error,
            "model": self._config.llm_settings["model"],
        }

    # -- generation --------------------------------------------------------
    async def stream(self, messages: list[dict], cancel: threading.Event):
        if self._llm is None:
            raise BackendError(
                "local_model_not_ready",
                "The local Gemma model is not loaded yet.",
                retryable=True,
            )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        llm_cfg = self._config.llm_settings

        def _deliver(item) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # loop closed during shutdown
                pass

        def _worker() -> None:
            try:
                with self._model_lock:
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
                    try:
                        for chunk in generation:
                            if cancel.is_set():
                                break
                            choice = chunk["choices"][0]
                            delta = choice.get("delta") or {}
                            text = delta.get("content") or ""
                            if text:
                                _deliver(("delta", text))
                    finally:
                        close = getattr(generation, "close", None)
                        if callable(close):
                            close()
            except Exception as exc:  # noqa: BLE001 - surfaced as a turn error
                logger.error("local generation failed: %s", type(exc).__name__)
                _deliver(("error", exc))
            finally:
                _deliver(("end", None))

        threading.Thread(target=_worker, name="llm-local", daemon=True).start()
        while True:
            kind, value = await queue.get()
            if kind == "end":
                return
            if kind == "error":
                raise BackendError(
                    "local_generation_failed",
                    f"The local Gemma model failed to generate a response ({type(value).__name__}).",
                    retryable=True,
                )
            if cancel.is_set():
                return
            yield {"type": "delta", "text": value}
