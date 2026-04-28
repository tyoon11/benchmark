"""
ST-MEM Encoder Adapter
=======================
Paper: https://arxiv.org/abs/2402.09450
Model sampling frequency: 250 Hz
Embedding dimension: 768
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


class StMemEncoder(nn.Module):
    """
    ST-MEM encoder wrapper.

    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, T) at data target_fs — resampled to 600 samples (2.4s @ 250Hz)
      - pooled_features: (B, 768)

    Note: 사전 학습 체크포인트는 pos_embedding이 seq_len=2250 (30 patch + 2 SEP)
          으로 만들어져 있지만, paper처럼 600 샘플만 forward하면
          `pos_embedding[:, 1:n+1]` 슬라이싱으로 첫 8 patch만 사용됨 — zero pad
          불필요.
    """

    # Paper: input_size=2.4s, fs_model=250 → 600 samples per window.
    chunk_seconds = 2.4
    model_fs = 250
    model_seq_len = 600

    def __init__(self, checkpoint=None, seq_len=2250, patch_size=75):
        super().__init__()
        from clinical_ts.models.ecg_foundation_models.st_mem.st_mem import (
            st_mem_vit_base_dec256d4b,
        )

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.model = st_mem_vit_base_dec256d4b(
            seq_len=seq_len, patch_size=patch_size, num_leads=12,
        )
        self.feature_dim = self.model.encoder.width  # 768

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[StMemEncoder] Missing keys: {missing[:5]}...")
        print(f"[StMemEncoder] Loaded from {path}")

    def forward(self, x):
        """x: (B, 12, T) at data target_fs → 600 samples (2.4s @ 250Hz)"""
        from einops import rearrange

        x = torch.nan_to_num(x)
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len, mode="linear", align_corners=False)

        enc = self.model.encoder
        num_leads = x.shape[1]

        x = enc.to_patch_embedding(x)
        b, _, n, _ = x.shape
        x = x + enc.pos_embedding[:, 1 : n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep = enc.sep_embedding[None, None, None, :]
        left_sep = sep.expand(b, num_leads, -1, -1) + enc.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep.expand(b, num_leads, -1, -1) + enc.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)

        lead_emb = torch.stack(list(enc.lead_embeddings)).unsqueeze(0)
        lead_emb = lead_emb.unsqueeze(2).expand(b, -1, n + 2, -1)
        x = x + lead_emb
        x = rearrange(x, "b c n p -> b (c n) p")

        x = enc.dropout(x)
        for i in range(enc.depth):
            x = getattr(enc, f"block{i}")(x)

        # remove SEP embeddings
        x = rearrange(x, "b (c n) p -> b c n p", c=num_leads)
        x = x[:, :, 1:-1, :]
        seq_feat = rearrange(x, "b c n p -> b (c n) p")

        pooled = torch.mean(x, dim=(1, 2))
        pooled = enc.norm(pooled)

        return seq_feat, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, param in self.model.encoder.named_parameters():
            if any(name.startswith(p) for p in ["pos_embedding", "sep_embedding",
                   "lead_embeddings", "to_patch_embedding"]) or \
               any(name.startswith(f"block{i}.") for i in [0, 1, 2]):
                early.append(param)
            else:
                late.append(param)
        return {"early": early, "late": late}
