"""
Model Wrapper
==============
어떤 ECG encoder든 통일된 인터페이스로 감싸서 다운스트림 태스크에 사용합니다.

Encoder 요구사항:
  forward(x) → (sequence_features, pooled_features)
    - x:                  (batch, n_leads, seq_len)
    - sequence_features:  (batch, seq_len', embed_dim) 또는 None
    - pooled_features:    (batch, feature_dim)

  만약 encoder가 pooled만 반환하면 sequence_features=None 가능.
  만약 encoder가 하나의 tensor만 반환하면 자동으로 GAP 적용.

Eval modes:
  - linear_probe:         Frozen encoder + Linear head
  - attention_probe:      Frozen encoder + Attention Pooling head
  - finetune_linear:      Full finetune + Linear head
  - finetune_attention:   Full finetune + Attention Pooling head
"""

import torch
import torch.nn as nn
from .heads import build_head


EVAL_MODES = [
    "linear_probe",
    "attention_probe",
    "finetune_linear",
    "finetune_attention",
]


class DownstreamWrapper(nn.Module):
    """
    Encoder + Head wrapper for downstream tasks.

    Args:
        encoder:        nn.Module — ECG encoder
        feature_dim:    int — encoder output dimension
        num_classes:    int — number of output classes
        eval_mode:      str — one of EVAL_MODES
        seq_feature_dim: int — sequence feature dim (for attention head, None=feature_dim)
        head_kwargs:    dict — extra kwargs for head (dropout, num_heads, etc.)
    """

    def __init__(
        self,
        encoder:         nn.Module,
        feature_dim:     int,
        num_classes:     int,
        eval_mode:       str = "linear_probe",
        seq_feature_dim: int = None,
        head_kwargs:     dict = None,
    ):
        super().__init__()
        assert eval_mode in EVAL_MODES, f"eval_mode must be one of {EVAL_MODES}"

        self.encoder = encoder
        self.feature_dim = feature_dim
        self.seq_feature_dim = seq_feature_dim or feature_dim
        self.num_classes = num_classes
        self.eval_mode = eval_mode
        self.is_frozen = eval_mode in ("linear_probe", "attention_probe")

        # Encoder freeze
        if self.is_frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # Head
        if "attention" in eval_mode:
            self.head = build_head("attention", self.seq_feature_dim, num_classes,
                                   **(head_kwargs or {}))
            self.use_seq_features = True
        else:
            self.head = build_head("linear", feature_dim, num_classes,
                                   **(head_kwargs or {}))
            self.use_seq_features = False

    def forward(self, x, **kwargs):
        """
        x: (batch, n_leads, seq_len)
        → (batch, num_classes)
        """
        x = torch.nan_to_num(x)

        # Encoder forward
        if self.is_frozen:
            with torch.no_grad():
                enc_out = self.encoder(x, **kwargs)
        else:
            enc_out = self.encoder(x, **kwargs)

        # Encoder output 파싱
        seq_feat, pooled_feat = self._parse_encoder_output(enc_out)

        # Head
        if self.use_seq_features and seq_feat is not None:
            logits = self.head(seq_feat)
        else:
            logits = self.head(pooled_feat)

        return torch.nan_to_num(logits)

    def _parse_encoder_output(self, enc_out):
        """
        Encoder 출력을 (sequence_features, pooled_features)로 정규화.

        지원 패턴:
          1. tuple (seq_feat, pooled_feat) → 그대로
          2. dict {"seq": ..., "pooled": ...} → 추출
          3. single tensor (B, D) → (None, pooled)
          4. single tensor (B, L, D) → (seq, GAP(seq))
        """
        if isinstance(enc_out, tuple) and len(enc_out) == 2:
            seq_feat, pooled_feat = enc_out
            if seq_feat is not None and seq_feat.dim() == 3 and seq_feat.shape[1] == seq_feat.shape[2]:
                pass  # (B, L, D)
            elif seq_feat is not None and seq_feat.dim() == 3 and seq_feat.shape[2] != self.seq_feature_dim:
                # (B, D, L) → transpose
                seq_feat = seq_feat.transpose(1, 2)
            return seq_feat, pooled_feat

        if isinstance(enc_out, dict):
            seq_feat = enc_out.get("seq", enc_out.get("sequence_features"))
            pooled_feat = enc_out.get("pooled", enc_out.get("pooled_features"))
            return seq_feat, pooled_feat

        if isinstance(enc_out, torch.Tensor):
            if enc_out.dim() == 2:
                # (B, D) — pooled only
                return None, enc_out
            elif enc_out.dim() == 3:
                # (B, L, D) — sequence features → GAP for pooled
                seq_feat = enc_out
                pooled_feat = enc_out.mean(dim=1)
                return seq_feat, pooled_feat

        raise ValueError(f"Cannot parse encoder output type: {type(enc_out)}")

    def get_param_groups(self, lr: float, discriminative_lr_factor: float = 0.1):
        """
        Discriminative LR parameter groups.

        Returns:
          - Frozen: [head_params at lr]
          - Finetune: [head at lr, encoder at lr * factor]
        """
        head_params = list(self.head.parameters())

        if self.is_frozen:
            return [{"params": head_params, "lr": lr}]

        encoder_params = list(self.encoder.parameters())
        return [
            {"params": head_params, "lr": lr},
            {"params": encoder_params, "lr": lr * discriminative_lr_factor},
        ]

    def train(self, mode=True):
        """Frozen 모드에서는 encoder를 항상 eval로 유지"""
        super().train(mode)
        if self.is_frozen:
            self.encoder.eval()
        return self
