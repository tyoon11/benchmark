"""
ECG-JEPA Encoder Adapter for Benchmark
========================================
ecg_jepa 프로젝트의 MaskTransformer를 벤치마크 인터페이스로 래핑합니다.

사용:
  python run.py --task ptbxl_super --eval_mode linear_probe \
      --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
      --encoder_ckpt /path/to/best.pth
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ecg_jepa 프로젝트 경로 추가
ECG_JEPA_ROOT = Path("/home/irteam/local-node-d/tykim/ecg_jepa")
sys.path.insert(0, str(ECG_JEPA_ROOT))


class ECGJEPAEncoder(nn.Module):
    """
    ECG-JEPA encoder wrapper.

    Benchmark 인터페이스:
      forward(x) → (sequence_features, pooled_features)
        - x: (B, 12, 5000) — 12리드, 500Hz × 10초 → 8리드 선택 후 250Hz로 리샘플
        - sequence_features: (B, 400, 768) — 8리드 × 50패치
        - pooled_features: (B, 768) — GAP

    Note: 체크포인트가 8채널(I, II, V1-V6)로 학습되었으므로 c=8 사용.
          12리드 입력에서 [0, 1, 6, 7, 8, 9, 10, 11] 채널을 선택합니다.
    """

    # 12리드 중 8채널 선택: I, II, V1, V2, V3, V4, V5, V6
    SELECTED_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]

    # Paper: input_size=10s, fs_model=250 → 2500 samples per window (full ECG).
    chunk_seconds = 10.0
    model_fs = 250
    model_seq_len = 2500

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        c: int = 8,
        p: int = 50,
        t: int = 50,
        drop_path_rate: float = 0.0,
        pos_type: str = "sincos",
        checkpoint: str = None,
    ):
        super().__init__()
        from models.ecg_jepa.model import MaskTransformer

        self.feature_dim = embed_dim
        self.embed_dim = embed_dim
        self.c = c

        self.encoder = MaskTransformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            c=c, p=p, t=t,
            drop_path_rate=drop_path_rate,
            pos_type=pos_type,
        )

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "encoder" in ckpt:
            state = ckpt["encoder"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"[ECGJEPAEncoder] Missing keys: {missing}")
        if unexpected:
            print(f"[ECGJEPAEncoder] Unexpected keys: {unexpected}")
        print(f"[ECGJEPAEncoder] Loaded from {path} (epoch={ckpt.get('epoch', '?')})")

    def forward(self, x):
        """
        x: (B, 12, T) at data target_fs → 8채널 선택 후 2500 samples (10s @ 250Hz)
        → (sequence_features, pooled_features)
        """
        x = torch.nan_to_num(x)

        # 12리드에서 8채널 선택
        if x.shape[1] == 12:
            x = x[:, self.SELECTED_LEADS, :]

        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len, mode="linear", align_corners=False)

        B, L, _ = x.shape
        x_patch = x.reshape(B, -1, self.encoder.t)   # (B, L*p, t)
        x_embed = self.encoder.W_P(x_patch)           # (B, L*p, embed_dim)

        pos_embed = self.encoder.pos_embed
        attn_mask = self.encoder._cross_attention_mask().to(x.device)

        pos_embed = pos_embed.unsqueeze(0)
        seq_feat = self.encoder.encoder_blocks(x_embed, pos_embed, attn_mask)
        if self.encoder.norm:
            seq_feat = self.encoder.norm(seq_feat)

        pooled = seq_feat.mean(dim=1)  # (B, embed_dim)
        return seq_feat, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, param in self.encoder.named_parameters():
            if name in ["pos_embed", "W_P.weight", "W_P.bias"] or \
               any(name.startswith(f"encoder_blocks.blocks.{i}.") for i in range(3)):
                early.append(param)
            else:
                late.append(param)
        return {"early": early, "late": late}
