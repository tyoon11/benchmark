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
from pathlib import Path

ECG_FM_BENCH = Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code")
sys.path.insert(0, str(ECG_FM_BENCH))


class ECGFounderEncoder(nn.Module):
    """
    ECG-Founder encoder wrapper.

    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, 2500)
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
        """x: (B, 12, 2500) at 500Hz"""
        x = torch.nan_to_num(x)
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
