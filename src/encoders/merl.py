"""
MERL Encoder Adapter (ResNet18)
=================================
Paper: https://arxiv.org/abs/2403.06659
Model sampling frequency: 500 Hz
Embedding dimension: 512
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ECG_FM_BENCH = Path(os.environ.get(
    "ECG_FM_BENCH_DIR",
    "/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code",
))
sys.path.insert(0, str(ECG_FM_BENCH))


class MerlResNetEncoder(nn.Module):
    """
    MERL ResNet18 encoder wrapper.
    Input: (B, 12, T) at data target_fs → 1250 samples (2.5s @ 500Hz).
    """

    # Paper: input_size=2.5s, fs_model=500 → 1250 samples per window.
    chunk_seconds = 2.5
    model_fs = 500
    model_seq_len = 1250

    def __init__(self, checkpoint=None):
        super().__init__()
        from clinical_ts.models.ecg_foundation_models.merl.resnet1d import ResNet18

        self.model = ResNet18(num_classes=1)  # dummy n_classes
        self.feature_dim = 512

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k: v for k, v in state.items() if not k.startswith("linear.")}
        missing, _ = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[MerlResNetEncoder] Missing keys: {missing[:5]}...")
        print(f"[MerlResNetEncoder] Loaded from {path}")

    def forward(self, x):
        """x: (B, 12, T) at data target_fs → 1250 samples (2.5s @ 500Hz)"""
        x = torch.nan_to_num(x)
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len, mode="linear", align_corners=False)

        out = torch.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)

        # out: (B, 512, T')
        seq = out.permute(0, 2, 1)  # (B, T', 512)
        pooled = self.model.avgpool(out).view(out.size(0), -1)  # (B, 512)
        return seq, pooled


