#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ERes2NetV2 backbone wrapper.

Reuse the pretrained ModelScope ERes2NetV2, remove its final pooling/segmentation layers, and
expose frame-level features:
  - Input fbank: (B, T, 80) with a 10 ms hop and time mean subtracted, matching ERes2NetV2 training.
  - Internal 8x temporal downsampling gives 12.5 Hz frames (80 ms each); mean-pool frequency bins.
  - Output features: (B, T/8, 1024); every two frames form one 160 ms chunk.

At inference, the backbone runs in eval mode and its weights are not modified.
"""
from __future__ import annotations
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _try_import_modelscope_eres2netv2():
    """Resolve the ERes2NetV2 class across supported ModelScope module paths."""
    try:
        from modelscope.models.audio.sv.ERes2NetV2 import ERes2NetV2
        return ERes2NetV2
    except Exception:
        pass
    try:
        from modelscope.models.audio.sv.eres2netv2 import ERes2NetV2
        return ERes2NetV2
    except Exception:
        pass
    return None


class ERes2NetV2Backbone(nn.Module):
    """ERes2NetV2 backbone without pooling; outputs frame features shaped (B, T/8, 1024)."""

    # Standard ERes2NetV2 hyperparameters matching the ModelScope checkpoint.
    EXPECTED_FEAT_DIM = 80
    EXPECTED_EMBED_DIM = 192
    EXPECTED_BASE_WIDTH = 26
    EXPECTED_SCALE = 2
    EXPECTED_EXPANSION = 2
    M_CHANNELS = 64
    TIME_DOWNSAMPLE = 8       # Total temporal downsampling; layers 2/3/4 each use stride 2.

    def __init__(self,
                 model_dir: str,
                 ckpt_filename: str = 'pretrained_eres2netv2.ckpt'):
        """
        Args:
            model_dir: Directory containing the ERes2NetV2 checkpoint.
            ckpt_filename: Checkpoint filename.
        """
        super().__init__()
        self.model_dir = os.path.abspath(model_dir)
        ckpt_path = os.path.join(self.model_dir, ckpt_filename)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ERes2NetV2 checkpoint does not exist: {ckpt_path}")

        ERes2NetV2 = _try_import_modelscope_eres2netv2()
        if ERes2NetV2 is None:
            raise ImportError("ModelScope ERes2NetV2 definition not found; install `modelscope`.")
        net = ERes2NetV2(
            feat_dim=self.EXPECTED_FEAT_DIM,
            embed_dim=self.EXPECTED_EMBED_DIM,
            baseWidth=self.EXPECTED_BASE_WIDTH,
            scale=self.EXPECTED_SCALE,
            expansion=self.EXPECTED_EXPANSION,
        )
        # Load only tensors and primitive types; do not allow a tampered pickle to execute code.
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        net.load_state_dict(state, strict=False)

        # Keep only the trunk (conv1/bn1/layers 1-4/layer3_ds/fuse34); omit pool/segmentation.
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.layer3_ds = net.layer3_ds
        self.fuse34 = net.fuse34

        self.output_dim = self.M_CHANNELS * 8 * self.EXPECTED_EXPANSION   # 64*8*2 = 1024
        logger.info(f"ERes2NetV2Backbone: output_dim={self.output_dim}, "
                    f"time_downsample={self.TIME_DOWNSAMPLE}x")

    def _trunk_forward(self, fbank: torch.Tensor) -> torch.Tensor:
        """Run the ERes2NetV2 trunk through fuse_out34, without the pooling layer.

        Args:
            fbank: (B, T, 80) with time mean subtracted
        Returns:
            fuse_out34: (B, 1024, F'=10, T'=T/8)
        """
        x = fbank.permute(0, 2, 1)        # (B, T, F) → (B, F, T)
        x = x.unsqueeze_(1)               # (B, 1, F, T)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3_ds = self.layer3_ds(out3)
        return self.fuse34(out4, out3_ds)  # (B, 1024, 10, T/8)

    def forward(self, fbank_eres2netv2: torch.Tensor) -> torch.Tensor:
        """Run a full-segment forward pass.

        Args:
            fbank_eres2netv2: (B, T, 80) with time mean subtracted
        Returns:
            frame_features: (B, T/8, 1024), frequency-mean-pooled at 12.5 Hz
        """
        fuse_out34 = self._trunk_forward(fbank_eres2netv2)   # (B, C, F', T')
        return fuse_out34.mean(dim=2).transpose(1, 2)        # (B, T', C)
