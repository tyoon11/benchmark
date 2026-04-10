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
from pathlib import Path

# ecg_jepa 프로젝트 경로 추가
ECG_JEPA_ROOT = Path("/home/irteam/local-node-d/tykim/ecg_jepa")
sys.path.insert(0, str(ECG_JEPA_ROOT))


class ECGJEPAEncoder(nn.Module):
    """
    ECG-JEPA encoder wrapper.

    Benchmark 인터페이스:
      forward(x) → (sequence_features, pooled_features)
        - x: (B, 12, 2500) — 12리드, 500Hz × 5초
        - sequence_features: (B, 600, 768) — 12리드 × 50패치
        - pooled_features: (B, 768) — GAP

    Args:
        embed_dim:      인코더 임베딩 차원 (기본 768)
        depth:          트랜스포머 깊이 (기본 12)
        num_heads:      어텐션 헤드 수 (기본 16)
        c:              리드 수 (기본 12)
        p:              리드당 패치 수 (기본 50)
        t:              패치당 타임포인트 (기본 50)
        checkpoint:     체크포인트 경로 (None이면 로드 안 함)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        c: int = 12,
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
        x: (B, n_leads, seq_len)
        → (sequence_features, pooled_features)
        """
        # representation: (B, embed_dim) — GAP
        pooled = self.encoder.representation(x)

        # sequence features: patchify → encode (마스킹 없이)
        B, L, _ = x.shape
        x_patch = x.reshape(B, -1, self.encoder.t)  # (B, L*p, t)
        x_embed = self.encoder.W_P(x_patch)          # (B, L*p, embed_dim)

        pos_embed = self.encoder.pos_embed
        attn_mask = self.encoder._cross_attention_mask().to(x.device)

        if L < self.encoder.c:
            lead_idx = list(range(L))
            rows = torch.cat([torch.arange(i * self.encoder.p, (i + 1) * self.encoder.p)
                              for i in lead_idx])
            pos_embed = pos_embed[rows]
            attn_mask = attn_mask[rows][:, rows]

        pos_embed = pos_embed.unsqueeze(0)
        seq_feat = self.encoder.encoder_blocks(x_embed, pos_embed, attn_mask)
        if self.encoder.norm:
            seq_feat = self.encoder.norm(seq_feat)
        # seq_feat: (B, L*p, embed_dim)

        return seq_feat, pooled
