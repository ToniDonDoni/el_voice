#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speaker-conditioned FiLM modulation layer.

Generate per-channel affine parameters (γ, β) from the enrolled-speaker embedding and modulate
frame-level features:
    out = γ(spk) · (feat + β(spk))
γ and β are broadcast along the time dimension. Zero-initializing the final layer makes the
initial transform an identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMSpeakerCond(nn.Module):
    """Speaker-conditioned FiLM modulation for frame-level features.

    γ(zspk), β(zspk) ∈ R^C are channel-wise modulation params,
    broadcast along time dimension.
    """

    def __init__(self,
                 spk_dim: int = 192,
                 feat_dim: int = 1024,
                 hidden: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.spk_dim = spk_dim
        self.feat_dim = feat_dim

        self.proj = nn.Sequential(
            nn.LayerNorm(spk_dim),
            nn.Linear(spk_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * feat_dim),
        )
        # Zero initialization makes γ=1 and β=0 at startup, so output initially equals input.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self,
                frame_feat: torch.Tensor,
                spk_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_feat: (B, T, C) frame-level features
            spk_emb:    (B, D_spk) L2-normalized speaker embedding
        Returns:
            (B, T, C) FiLM-modulated features
        """
        gb = self.proj(spk_emb)                          # (B, 2C)
        gamma_tilde, beta = gb.chunk(2, dim=-1)          # Each has shape (B, C).
        gamma = (1.0 + gamma_tilde).unsqueeze(1)         # (B, 1, C), broadcast over time.
        beta = beta.unsqueeze(1)                         # (B, 1, C)
        return gamma * (frame_feat + beta)
