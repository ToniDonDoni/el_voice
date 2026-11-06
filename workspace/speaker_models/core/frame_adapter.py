#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Causal 1D convolutional frame adapter (Causal Conv1d ResBlock + GLU).

This temporal module preserves the input/output shape (B, T, D).
Each residual block applies CausalConv1d(D→2D), GLU, LayerNorm, Dropout, and a residual connection.
It is strictly causal (left padding only), has no self-attention, and supports streaming inference
by caching only the last kernel-1 frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm1d(nn.Module):
    """Apply LayerNorm across channels independently at each time step of a (B, C, T) input.

    Unlike nn.GroupNorm(1, C), which computes statistics across (C, T) and uses future frames,
    ChannelLayerNorm1d computes statistics across C only and remains strictly causal.

    This is equivalent to nn.LayerNorm(C), with transposes used to normalize the desired axis.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) → (B, T, C) → LayerNorm(C) → (B, C, T)
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
        return x


class CausalConv1dResBlock(nn.Module):
    """Causal 1D convolutional residual block with GLU.

    Data flow:
        x (B, D, T)
          → left pad (kernel-1) → CausalConv1d(D → 2D, k)
          → GLU (split channels in half) → (B, D, T)
          → ChannelLayerNorm (per-frame channel normalization, strictly causal)
          → Dropout
          → x + residual
        → ReLU
        → return (B, D, T)
    """

    def __init__(self,
                 d_model: int,
                 kernel_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        assert kernel_size >= 1
        self.kernel_size = kernel_size
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,   # GLU requires twice the channel count.
            kernel_size=kernel_size,
            padding=0,                  # Apply causal padding explicitly.
            bias=True,
        )
        self.norm = ChannelLayerNorm1d(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T)
        Returns:
            out: (B, D, T)
        """
        residual = x
        x = F.pad(x, (self.left_pad, 0))             # Left padding only keeps the operation causal.
        x = self.conv(x)                              # (B, 2D, T)
        x = F.glu(x, dim=1)                           # (B, D, T)
        x = self.norm(x)
        x = self.drop(x)
        return F.relu(x + residual, inplace=False)


class CausalFrameAdapter(nn.Module):
    """Stack three causal 1D convolutional residual blocks."""

    def __init__(self,
                 d_model: int = 256,
                 depth: int = 3,
                 kernel_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.blocks = nn.ModuleList([
            CausalConv1dResBlock(d_model=d_model,
                                 kernel_size=kernel_size,
                                 dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, T, D)
        """
        x = x.transpose(1, 2).contiguous()           # (B, D, T)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).contiguous()           # (B, T, D)
        return x
