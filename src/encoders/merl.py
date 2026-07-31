"""
MERL Encoder Adapter (ResNet18)
=================================
Paper: https://arxiv.org/abs/2403.06659
Model sampling frequency: 500 Hz
Embedding dimension: 512
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# clinical_ts subset is bundled under benchmark/src/external/
EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "external"
sys.path.insert(0, str(EXTERNAL_DIR))

from ._contract import ensure_length


class MerlResNetEncoder(nn.Module):
    """
    MERL ResNet18 encoder wrapper.
    Input: (B, 12, T) from the dataset: 1250 samples (2.5s @ 500Hz).
    """

    # Encoder contract (original run.sh: --input-size 2.5 --fs-model 500).
    # The dataset crops at the native rate and band-limit resamples to
    # model_fs, so the tensor arriving here is already model_seq_len long.
    input_size = 2.5          # seconds
    model_fs = 500
    model_seq_len = 1250
    lead_order = "standard"    # I,II,III,aVR,aVL,aVF,V1..V6
    chunk_seconds = 2.5       # deprecated alias for input_size

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
        """x: (B, 12, T) from the dataset: 1250 samples (2.5s @ 500Hz)"""
        x = torch.nan_to_num(x)
        x = ensure_length(x, self.model_seq_len, type(self).__name__)

        out = torch.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)

        # out: (B, 512, T')
        seq = out.permute(0, 2, 1)  # (B, T', 512)
        pooled = self.model.avgpool(out).view(out.size(0), -1)  # (B, 512)
        return seq, pooled

    def get_layer_groups(self):
        """Discriminative-LR groups, matching MerlWrapper.get_params() in the original.

        early (lr x factor^2): conv1, bn1, layer1, layer2
        late  (lr x factor)  : layer3, layer4
        Anything else (the discarded `linear` head) is left out of the optimiser,
        exactly as the original does.
        """
        early, late = [], []
        for name, param in self.model.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                early.append(param)
            elif name.startswith(("layer3", "layer4")):
                late.append(param)
        return {"early": early, "late": late}


