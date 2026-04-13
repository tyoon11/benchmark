"""
HuBERT-ECG Encoder Adapter
============================
Model sampling frequency: 500 Hz (with bandpass preprocessing)
Embedding dimension: 768
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ECG_FM_BENCH = Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code")
sys.path.insert(0, str(ECG_FM_BENCH))


class HuBERTECGEncoder(nn.Module):
    """
    HuBERT-ECG encoder wrapper.

    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, 2500) at 500Hz
      - pooled_features: (B, 768)
    """

    def __init__(self, checkpoint=None):
        super().__init__()
        from clinical_ts.models.ecg_foundation_models.hubert_ecg.hubert_ecg import HuBERTECG
        from clinical_ts.models.ecg_foundation_models.hubert_ecg.config import hubert_config

        self.hubert_config = hubert_config
        self.encoder = HuBERTECG(hubert_config)
        self.feature_dim = hubert_config.hidden_size  # 768

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            state = {k: f.get_tensor(k) for k in f.keys()}
        missing, _ = self.encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"[HuBERTECGEncoder] Missing keys: {missing[:5]}...")
        print(f"[HuBERTECGEncoder] Loaded from {path}")

    def _preprocess(self, x_np):
        """Bandpass filter + scaling per sample"""
        from clinical_ts.models.ecg_foundation_models.hubert_ecg.utils import ecg_preprocessing

        processed = []
        for sig in x_np:
            processed.append(ecg_preprocessing(sig))
        return np.stack(processed, axis=0)

    def forward(self, x):
        """x: (B, 12, 2500) at 500Hz"""
        x = torch.nan_to_num(x)
        x_np = x.detach().cpu().numpy()
        x_np = self._preprocess(x_np)
        x = torch.from_numpy(x_np).to(x.device).float()
        x = x.reshape(x.shape[0], -1)  # (B, 12*2500)

        encodings = self.encoder(x, return_dict=True)
        seq = encodings.last_hidden_state     # (B, T, 768)
        pooled = seq.mean(dim=1)              # (B, 768)

        return seq, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, param in self.named_parameters():
            if "feature_extractor" in name or "feature_projection" in name:
                early.append(param)
            elif "encoder.layers" in name:
                try:
                    layer_num = int(name.split("encoder.layers.")[1].split(".")[0])
                    if layer_num < 6:
                        early.append(param)
                    else:
                        late.append(param)
                except (IndexError, ValueError):
                    late.append(param)
            elif "encoder" in name:
                early.append(param)
        return {"early": early, "late": late}
