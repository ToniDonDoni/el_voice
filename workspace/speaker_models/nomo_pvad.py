# -*- coding: utf-8 -*-
"""Nomo-PVAD streaming scorer adapted from the pinned upstream release.

Wrap the Nomo-PVAD network as an enrollment plus streaming target score interface.
Use the vendored pinned core and a separately provisioned ModelScope ERes2NetV2 encoder.

Usage:
    m = NomoPVAD(ckpt, eres2netv2_dir, device='cpu')
    sess = NomoPVADSession(m)
    sess.set_enrollment(enroll_pcm_float32_16k)   # Enroll the target speaker.
    for chunk in chunks:                          # Each 160 ms (2560 samples at 16 kHz).
        prob = sess.process_chunk(chunk_float32)  # Target-speaker probability [0, 1].
    # Or score a complete utterance offline:
    probs = sess.score_utterance(pcm_float32)     # Probability array, one value per chunk.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

logger = logging.getLogger("nomo_pvad")

#: Allowlisted architecture fields that may be read from checkpoint model_cfg.
#: Ignore all others to prevent config injection. Keep this list aligned with NomoPVADNet;
#: this class injects eres2netv2_model_dir, so it is intentionally not included here.
_ALLOWED_CFG_KEYS = frozenset({
    "spk_embed_dim", "film_hidden", "film_dropout", "d_model",
    "adapter_kernel", "adapter_dropout", "speech_encoder_depth",
    "classifier_hidden_dim", "classifier_dropout", "num_classes",
    "decoder_depth", "eres2netv2_ckpt_filename",
})

SR = 16000
CHUNK_SAMPLES = SR * 160 // 1000      # 2560 = 160ms


def _eres2netv2_fbank(wav: torch.Tensor, sample_rate: int = SR) -> torch.Tensor:
    """Compute 80-bin fbank (25 ms window, 10 ms hop, no dither) and subtract its time mean."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav * (1 << 15)
    feat = kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10,
                       sample_frequency=sample_rate, dither=0.0)
    return feat - feat.mean(dim=0, keepdim=True)


# ERes2NetV2 is a third-party 3D-Speaker/ModelScope backbone. Its weights are not bundled;
# provision them from the official ModelScope source before runtime.
ERES2NETV2_MODELSCOPE_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"


def _resolve_eres_dir(eres_dir: Optional[str]) -> str:
    """Require a pre-provisioned checkpoint; runtime downloads are forbidden."""
    ckpt = "pretrained_eres2netv2.ckpt"
    if eres_dir and os.path.isfile(os.path.join(eres_dir, ckpt)):
        return eres_dir
    raise FileNotFoundError("The pinned ERes2NetV2 checkpoint must be provisioned before runtime")


class NomoPVAD:
    """NomoPVAD network and ERes2NetV2 extractor, loaded once and shared across sessions."""

    def __init__(self, ckpt: str, eres2netv2_dir: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        # Use the pre-provisioned third-party ERes2NetV2 checkpoint.
        self._eres2netv2_dir = _resolve_eres_dir(eres2netv2_dir)

        from speaker_models.core.model import build_model

        # Load only tensors and primitive types; do not allow a tampered pickle to execute code.
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        # Accept only allowlisted architecture fields to prevent checkpoint config injection.
        raw_cfg = state.get("model_cfg", {}) if isinstance(state, dict) else {}
        cfg = {k: v for k, v in raw_cfg.items() if k in _ALLOWED_CFG_KEYS}
        cfg["eres2netv2_model_dir"] = self._eres2netv2_dir
        self.model = build_model(cfg)
        self.model.load_state_dict(state.get("model_state_dict", state), strict=False)
        self.model.eval().to(self.device)

        self._eres = None   # Lazily loaded ERes2NetV2 extractor.
        logger.info("NomoPVAD ready (device=%s)", device)

    def _ensure_eres(self):
        if self._eres is not None:
            return
        from modelscope.models.audio.sv.ERes2NetV2 import ERes2NetV2   # noqa
        m = ERes2NetV2(feat_dim=80, embed_dim=192, baseWidth=26, scale=2, expansion=2)
        ck = os.path.join(self._eres2netv2_dir, "pretrained_eres2netv2.ckpt")
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=False)
        m.eval().to(self.device)
        for p in m.parameters():
            p.requires_grad_(False)
        self._eres = m

    @torch.no_grad()
    def extract_emb(self, pcm_f32: np.ndarray) -> torch.Tensor:
        """Extract an L2-normalized 192-dimensional speaker d-vector from enrollment audio."""
        self._ensure_eres()
        feat = _eres2netv2_fbank(torch.from_numpy(np.asarray(pcm_f32, dtype=np.float32))).unsqueeze(0).to(self.device)
        return F.normalize(self._eres(feat).float(), dim=-1)   # (1, 192)

    @torch.no_grad()
    def probs_on(self, spk_emb: torch.Tensor, pcm_f32: np.ndarray) -> np.ndarray:
        """Return the target-speaker probability for each chunk in an audio segment."""
        fb = _eres2netv2_fbank(torch.from_numpy(np.asarray(pcm_f32, dtype=np.float32))).unsqueeze(0).to(self.device)
        lens = torch.tensor([fb.shape[1]], device=self.device)
        out = self.model(spk_emb, fb, lens)
        mask = out["chunk_mask"][0].cpu().numpy()
        probs = F.softmax(out["chunk_logits"], dim=-1)[0, :, 1].cpu().numpy()
        return probs[mask]


class NomoPVADSession:
    """Per-connection scorer holding a speaker embedding and sliding window; shares NomoPVAD."""

    def __init__(self, model: NomoPVAD, window_sec: float = 2.4):
        self.m = model
        self.window_samples = int(SR * window_sec)
        self.spk_emb: Optional[torch.Tensor] = None
        self._buf = np.zeros(0, dtype=np.float32)

    def set_enrollment(self, pcm_f32: np.ndarray):
        """Extract an embedding from enrollment audio and reset the sliding window."""
        self.spk_emb = self.m.extract_emb(pcm_f32)
        self.reset()

    def set_enrollment_emb(self, emb_192: np.ndarray):
        """Set a precomputed 192-dimensional enrollment embedding and reset the window."""
        e = torch.from_numpy(np.asarray(emb_192, dtype=np.float32)).reshape(1, -1)
        self.spk_emb = F.normalize(e, dim=-1).to(self.m.device)
        self.reset()

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)

    def process_chunk(self, chunk_f32: np.ndarray) -> float:
        """Score each 160 ms chunk in the sliding window and return its target probability."""
        assert self.spk_emb is not None, "Call set_enrollment first."
        self._buf = np.concatenate([self._buf, np.asarray(chunk_f32, dtype=np.float32)])
        if self._buf.size > self.window_samples:
            self._buf = self._buf[-self.window_samples:]
        if self._buf.size < CHUNK_SAMPLES:
            return 0.0
        probs = self.m.probs_on(self.spk_emb, self._buf)
        return float(probs[-1]) if len(probs) else 0.0

    def score_utterance(self, pcm_f32: np.ndarray) -> np.ndarray:
        """Score all chunks in an utterance at once (offline)."""
        assert self.spk_emb is not None, "Call set_enrollment first."
        return self.m.probs_on(self.spk_emb, pcm_f32)
