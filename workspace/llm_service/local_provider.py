"""CPU Gemma implementation used only by the Local Voice LLM service."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, AsyncIterator

SYSTEM_PROMPT = (
    "You are a concise Russian-speaking Dota 2 companion. Answer in Russian unless the user "
    "asks otherwise. Focus on Dota 2 strategy, heroes, items, and game concepts. Do not claim "
    "live match data or current patch facts you cannot verify. Keep replies natural and short "
    "enough to speak aloud; avoid markdown and lists unless useful."
)
MODEL_RELATIVE_PATH = "llm/gemma-3-1b-it-Q4_K_M.gguf"
MODEL_TOOL = "/opt/el-voice/download_models.py"


class CancelOnToken:
    """Ask llama.cpp to emit EOS at the next token boundary after cancellation."""

    def __init__(self, cancel_event: threading.Event, eos_token: int):
        self.cancel_event = cancel_event
        self.eos_token = eos_token

    def __call__(self, _input_ids: Any, scores: Any) -> Any:
        if self.cancel_event.is_set():
            scores.fill(float("-inf"))
            scores[self.eos_token] = 0.0
        return scores


class LocalGemmaProvider:
    def __init__(self, model_path: str | Path | None = None, context_tokens: int | None = None,
                 max_response_tokens: int | None = None, cpu_threads: int | None = None):
        model_root = Path(os.environ.get("EL_VOICE_MODELS", "/models")).resolve()
        self.model_path = Path(model_path or os.getenv(
            "EL_VOICE_LOCAL_MODEL", str(model_root / MODEL_RELATIVE_PATH))).resolve()
        try:
            relative_model = self.model_path.relative_to(model_root).as_posix()
        except ValueError as exc:
            raise RuntimeError("Local LLM model must stay inside the configured models directory") from exc
        if relative_model != MODEL_RELATIVE_PATH:
            raise RuntimeError("Local LLM model path is not listed in the configured model manifest")
        if not self.model_path.is_file():
            raise RuntimeError("Local LLM model file is missing")
        verification = subprocess.run(
            [sys.executable, MODEL_TOOL, "--verify", "--only", MODEL_RELATIVE_PATH],
            check=False, capture_output=True, text=True,
        )
        if verification.returncode != 0:
            raise RuntimeError("Local LLM model verification failed")
        context_tokens = context_tokens or int(os.getenv("LLM_CONTEXT_TOKENS", "3072"))
        max_response_tokens = max_response_tokens or int(os.getenv("LLM_MAX_RESPONSE_TOKENS", "192"))
        cpu_threads = cpu_threads or int(os.getenv("LLM_CPU_THREADS", "4"))
        from llama_cpp import Llama

        self.model = Llama(model_path=str(self.model_path), n_ctx=context_tokens,
                           n_threads=cpu_threads, n_batch=256, verbose=False)
        self.max_response_tokens = max_response_tokens
        self.lock = threading.Lock()

    async def stream(self, text: str, history: list[dict[str, str]],
                     cancelled: asyncio.Event) -> AsyncIterator[dict[str, Any]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                    {"role": "user", "content": text}]
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=8)

        stop_generation = threading.Event()
        cancel_callback = CancelOnToken(stop_generation, self.model.token_eos())

        async def watch_cancellation() -> None:
            while not cancelled.is_set():
                await asyncio.sleep(0.02)
            stop_generation.set()

        cancellation_watcher = asyncio.create_task(watch_cancellation())

        def push(kind: str, value: Any = None) -> bool:
            future = asyncio.run_coroutine_threadsafe(queue.put((kind, value)), loop)
            while True:
                try:
                    future.result(timeout=0.25)
                    return True
                except TimeoutError:
                    if stop_generation.is_set():
                        future.cancel()
                        return False
                except Exception:
                    future.cancel()
                    return False

        def generate() -> None:
            try:
                with self.lock:
                    result = self.model.create_chat_completion(
                        messages=messages, stream=True, temperature=0.55, top_p=0.9,
                        max_tokens=self.max_response_tokens,
                        logits_processor=[cancel_callback],
                    )
                    try:
                        for item in result:
                            if cancelled.is_set() or not push("delta", item["choices"][0]["delta"].get("content", "")):
                                break
                    finally:
                        close = getattr(result, "close", None)
                        if close:
                            close()
                push("done")
            except Exception:
                push("error", "local_inference_failed")

        worker = asyncio.create_task(asyncio.to_thread(generate))
        text_parts = []
        try:
            while not cancelled.is_set():
                try:
                    kind, value = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if worker.done():
                        break
                    continue
                if kind == "done":
                    break
                if kind == "error":
                    raise RuntimeError(value)
                if value:
                    text_parts.append(value)
                    yield {"type": "delta", "text": value}
        finally:
            cancelled.set() if asyncio.current_task().cancelling() else None
            if cancelled.is_set():
                stop_generation.set()
            cancellation_watcher.cancel()
            await asyncio.gather(cancellation_watcher, return_exceptions=True)
            if not worker.done():
                while not worker.done():
                    await asyncio.sleep(0.02)
            await asyncio.gather(worker, return_exceptions=True)
        if not cancelled.is_set() and text_parts:
            yield {"type": "done", "text": "".join(text_parts)}

    async def close(self) -> None:
        return None
