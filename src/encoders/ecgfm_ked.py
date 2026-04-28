"""
ECG-FM-KED Encoder Adapter
============================
Paper: https://doi.org/10.1016/j.xcrm.2024.101875
Pretraining sampling frequency: 100 Hz
Benchmark fs (run.sh): fs_model=500, input_size=10s → 5000 samples
Embedding dimension: 768
"""

import sys
import types
import enum
import re
import inspect
from typing import Optional, Collection

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ECG_FM_BENCH = Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code")
sys.path.insert(0, str(ECG_FM_BENCH))

# fastai v1 compatibility shim: ecgfm_ked.py uses `from fastai.core import *`
# which expects typing names, Enum, Floats, etc.
if "fastai.core" not in sys.modules:
    import fastcore.foundation as _ff
    import fastcore.basics as _fb

    _core = types.ModuleType("fastai.core")
    for _m in [_ff, _fb]:
        _core.__dict__.update({k: v for k, v in _m.__dict__.items() if not k.startswith("__")})
    # Add typing names that fastai v1 re-exported
    _core.__dict__["Optional"] = Optional
    _core.__dict__["Collection"] = Collection
    _core.__dict__["Floats"] = float
    _core.__dict__["Enum"] = enum.Enum
    # Ensure real stdlib modules aren't shadowed
    _core.__dict__["re"] = re
    _core.__dict__["inspect"] = inspect
    sys.modules["fastai.core"] = _core


class EcgFmKEDEncoder(nn.Module):
    """
    ECG-FM-KED (xresnet1d101) encoder wrapper.

    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, T) at data target_fs → 5000 samples (10s @ 500Hz)
        (paper's run.sh uses fs_model=500, NOT the model's pretraining 100Hz —
         xresnet1d101 is a CNN that accepts any rate)
      - pooled_features: (B, 768)
    """

    # Paper run.sh: input_size=10s, fs_model=500 → 5000 samples (full ECG).
    chunk_seconds = 10.0
    model_fs = 500
    model_seq_len = 5000

    def __init__(self, checkpoint=None):
        super().__init__()
        from clinical_ts.models.ecg_foundation_models.ecgfm_ked import xresnet1d101

        self.model = xresnet1d101(
            num_classes=768,
            input_channels=12,
            kernel_size=5,
            ps_head=0.5,
        )
        self.feature_dim = 768

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {}
        for k, v in ckpt.items():
            if k.startswith("ecg_model."):
                state[k.replace("ecg_model.", "")] = v
            elif not k.startswith("model."):
                state[k] = v
        missing, _ = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[EcgFmKEDEncoder] Missing keys: {missing[:5]}...")
        print(f"[EcgFmKEDEncoder] Loaded from {path}")

    def forward(self, x):
        """x: (B, 12, T) at data target_fs → 5000 samples (10s @ 500Hz, paper run.sh)"""
        from einops import rearrange

        x = torch.nan_to_num(x)
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len, mode="linear", align_corners=False)

        # nn.Sequential forward — DataParallel safe (모델이 같은 device에 있음)
        seq = nn.Sequential.forward(self.model, x)  # (B, 768, T')
        seq = rearrange(seq, "b c l -> b l c")
        pooled = torch.mean(seq, dim=1)

        return seq, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, param in self.model.named_parameters():
            if name.startswith(("0.", "1.", "2.")):
                early.append(param)
            elif name.startswith(("4.", "5.", "6.", "7.")):
                late.append(param)
        return {"early": early, "late": late}
