"""
ECG-Founder Encoder Adapter
============================
Paper: https://arxiv.org/abs/2410.04133
Model sampling frequency: 500 Hz
Embedding dimension: 1024
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ECG_FM_BENCH = Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code")
sys.path.insert(0, str(ECG_FM_BENCH))


class ECGFounderEncoder(nn.Module):
    """
    ECG-Founder encoder wrapper.

    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, 5000) at 500Hz (10s) — cropped to 2.5s (1250 samples) internally
      - sequence_features: (B, seq_len, 1024)
      - pooled_features:   (B, 1024)
    """

    def __init__(self, checkpoint=None):
        super().__init__()
        from clinical_ts.models.ecg_foundation_models.ecg_founder import Net1D

        self.model = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=1,  # dummy, will be removed
        )

        self.feature_dim = self.model.dense.in_features  # 1024
        self.model.dense = nn.Identity()

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            state = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("dense.")}
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[ECGFounderEncoder] Missing keys: {missing[:5]}...")
        print(f"[ECGFounderEncoder] Loaded from {path}")

    def forward(self, x):
        """x: (B, 12, 5000) at 500Hz → crop to 1250 samples (2.5s)"""
        x = torch.nan_to_num(x)
        # Resample to 500Hz × 10s = 5000 samples (already at 500Hz, identity)
        x = F.interpolate(x, size=5000, mode="linear", align_corners=False)
        # Crop to first input_size × fs_model = 2.5s × 500Hz = 1250 samples
        x = x[:, :, :1250]
        out = self.model.first_conv(x)
        if self.model.use_bn:
            out = self.model.first_bn(out)
        out = self.model.first_activation(out)

        for i_stage in range(self.model.n_stages):
            out = self.model.stage_list[i_stage](out)

        # out: (B, 1024, seq_len)
        pooled = out.mean(dim=-1)             # (B, 1024)
        seq = out.permute(0, 2, 1)            # (B, seq_len, 1024)
        return seq, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, param in self.model.named_parameters():
            if name.startswith(("first_conv", "first_bn")) or \
               any(name.startswith(f"stage_list.{i}.") for i in [0, 1, 2]):
                early.append(param)
            else:
                late.append(param)
        return {"early": early, "late": late}
