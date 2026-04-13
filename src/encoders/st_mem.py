"""
ST-MEM Encoder Adapter
=======================
Paper: https://arxiv.org/abs/2402.09450
Model sampling frequency: 250 Hz
Embedding dimension: 768
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ECG_FM_BENCH = Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code")
sys.path.insert(0, str(ECG_FM_BENCH))


class StMemEncoder(nn.Module):
    """
    ST-MEM encoder wrapper.

    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, 2500) at 500Hz — resampled to 250Hz internally
      - pooled_features: (B, 768)
    """

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

    def _resample(self, x):
        """500Hz 2500 → 250Hz 1250 → zero-pad to seq_len"""
        x = F.interpolate(x, size=1250, mode="linear", align_corners=False)
        if 1250 < self.seq_len:
            x = F.pad(x, (0, self.seq_len - 1250))
        else:
            x = x[:, :, : self.seq_len]
        return x

    def forward(self, x):
        """x: (B, 12, 2500) at 500Hz"""
        from einops import rearrange

        x = torch.nan_to_num(x)
        x = self._resample(x)  # (B, 12, seq_len)

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
