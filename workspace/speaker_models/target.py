"""Offline Nomo-PVAD wrapper with explicit, verified local checkpoint paths."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from speaker_models.nomo_pvad import CHUNK_SAMPLES, NomoPVAD, NomoPVADSession


class TargetSpeakerModel:
    def __init__(self, model_root: Path, cpu_threads: int = 2):
        self.model_root = model_root
        import torch

        # Share the voice container's six CPU cores with STT; Nomo's rolling-window
        # scorer otherwise competes with CTranslate2 and builds a stale audio queue.
        torch.set_num_threads(cpu_threads)
        os.environ["MODELSCOPE_OFFLINE"] = "1"
        os.environ["MODELSCOPE_DISABLE_UPDATE"] = "1"
        os.environ.setdefault("MODELSCOPE_CACHE", str(model_root / ".modelscope-cache"))
        self.scorer = NomoPVAD(
            str(model_root / "speaker/nomo_pvad.pt"),
            str(model_root / "speaker/eres2netv2"),
            device="cpu",
        )
        self.sessions: dict[str, NomoPVADSession] = {}
        self.last_timings_ms: list[float] = []
        self.thread_lock = __import__("threading").Lock()

    def enroll(self, audio: np.ndarray) -> list[float]:
        with self.thread_lock:
            embedding = self.scorer.extract_emb(np.asarray(audio, dtype=np.float32))
        return embedding.detach().cpu().numpy().reshape(192).astype(np.float32).tolist()

    def score(self, profiles: dict[str, list[float]], audio: np.ndarray) -> tuple[float, float]:
        started = time.perf_counter()
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.shape != (CHUNK_SAMPLES,):
            raise ValueError("Speaker scoring expects exactly 160 ms of 16 kHz audio")
        scores = []
        with self.thread_lock:
            for profile_id, embedding in profiles.items():
                session = self.sessions.get(profile_id)
                if session is None:
                    session = NomoPVADSession(self.scorer)
                    session.set_enrollment_emb(np.asarray(embedding, dtype=np.float32))
                    self.sessions[profile_id] = session
                scores.append(session.process_chunk(chunk))
        elapsed = (time.perf_counter() - started) * 1000
        self.last_timings_ms.append(elapsed)
        del self.last_timings_ms[:-1000]
        return max(scores, default=0.0), elapsed

    def clear(self, profile_id: str) -> None:
        with self.thread_lock:
            self.sessions.pop(profile_id, None)

    def warmup(self, profile_id: str, embedding: list[float]) -> None:
        """Warm the scorer's full rolling window without retaining raw session audio."""
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        for _ in range(round(2.4 / 0.16)):
            self.score({profile_id: embedding}, silence)
        self.clear(profile_id)
        self.last_timings_ms.clear()
