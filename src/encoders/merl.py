"""
MERL Encoder Adapters (ResNet18 & ViT-Tiny)
=============================================
Paper: https://arxiv.org/abs/2403.06659
Model sampling frequency: 500 Hz
Embedding dimension: ResNet=512, ViT=192
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ECG_FM_BENCH = Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code")
sys.path.insert(0, str(ECG_FM_BENCH))


class MerlResNetEncoder(nn.Module):
    """
    MERL ResNet18 encoder wrapper.
    Input: (B, 12, 5000) at 500Hz (10s) — cropped to 2.5s (1250 samples) internally.
    """

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
        """x: (B, 12, 5000) at 500Hz → crop to 1250 samples (2.5s)"""
        x = torch.nan_to_num(x)
        # Resample to fs_model × 10s = 500Hz × 10s = 5000 (identity)
        x = F.interpolate(x, size=5000, mode="linear", align_corners=False)
        # Crop to input_size × fs_model = 2.5s × 500Hz = 1250 samples
        x = x[:, :, :1250]

        out = torch.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)

        # out: (B, 512, T')
        seq = out.permute(0, 2, 1)  # (B, T', 512)
        pooled = self.model.avgpool(out).view(out.size(0), -1)  # (B, 512)
        return seq, pooled


class MerlViTEncoder(nn.Module):
    """
    MERL ViT-Tiny encoder wrapper.
    Input: (B, 12, 5000) at 500Hz (10s) — cropped to 2.5s (1250 samples),
    then zero-padded back to pretrained seq_len=5000 to keep the fixed
    pos_embedding shape.
    """

    def __init__(self, checkpoint=None):
        super().__init__()
        from clinical_ts.models.ecg_foundation_models.merl.vit1d import vit_tiny

        self.model = vit_tiny(num_leads=12, num_classes=1, seq_len=5000, patch_size=50)
        self.feature_dim = self.model.width  # 192

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k: v for k, v in state.items() if not k.startswith("linear.")}
        missing, _ = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[MerlViTEncoder] Missing keys: {missing[:5]}...")
        print(f"[MerlViTEncoder] Loaded from {path}")

    def forward(self, x):
        """x: (B, 12, 5000) at 500Hz → crop to 1250 samples (2.5s), pad back to 5000"""
        from einops import rearrange

        x = torch.nan_to_num(x)
        # Resample to fs_model × 10s = 500Hz × 10s = 5000 (identity)
        x = F.interpolate(x, size=5000, mode="linear", align_corners=False)
        # Crop to paper's input_size: 2.5s × 500Hz = 1250 samples
        x = x[:, :, :1250]
        # Zero-pad to pretrained seq_len=5000 for fixed pos_embedding compatibility
        x = F.pad(x, (0, 5000 - 1250))

        x = self.model.to_patch_embedding(x)
        x = rearrange(x, "b c n -> b n c")
        x = x + self.model.pos_embedding

        seq = self.model.dropout(x)
        for i in range(self.model.depth):
            seq = getattr(self.model, f"block{i}")(seq)

        pooled = torch.mean(seq, dim=1)
        pooled = self.model.norm(pooled)

        return seq, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, param in self.model.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                early.append(param)
            elif name.startswith(("layer3", "layer4")):
                late.append(param)
        return {"early": early, "late": late}
