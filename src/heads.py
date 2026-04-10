"""
Downstream Head 모듈
=====================
어떤 인코더 모델이든 공통으로 사용할 수 있는 Head 구현.

지원 모드:
  - linear:     Frozen encoder + Linear head (Linear Probe)
  - attention:  Frozen encoder + Attention Pooling head
  - finetune:   Full finetuning + Linear head
  - finetune_attention: Full finetuning + Attention Pooling head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
from dataclasses import dataclass


class LinearHead(nn.Module):
    """Pooled features → Linear → logits"""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """x: (batch, feature_dim) → (batch, num_classes)"""
        return self.fc(self.dropout(x))


class AttentionPoolingHead(nn.Module):
    """
    Learnable Query Attention Pooling (V-JEPA style).
    Sequence features → Attention pool → logits.
    """

    def __init__(self, embed_dim: int, num_classes: int, num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim ** 0.5)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, num_classes, bias=False)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim) — sequence features
        → (batch, num_classes)
        """
        # x: (B, S, E) → permute for MHA: (S, B, E)
        x_t = x.permute(1, 0, 2)
        query = self.query.expand(-1, x.shape[0], -1)  # (1, B, E)

        attn_out, _ = F.multi_head_attention_forward(
            query=query, key=x_t, value=x_t,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=None,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return self.dropout(attn_out[0])  # (B, num_classes)


class MLPHead(nn.Module):
    """2-layer MLP Head"""

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """x: (batch, feature_dim) → (batch, num_classes)"""
        return self.net(x)


def build_head(head_type: str, feature_dim: int, num_classes: int, **kwargs) -> nn.Module:
    """Head factory"""
    if head_type == "linear":
        return LinearHead(feature_dim, num_classes, dropout=kwargs.get("dropout", 0.0))
    elif head_type == "attention":
        return AttentionPoolingHead(
            feature_dim, num_classes,
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.0),
        )
    elif head_type == "mlp":
        return MLPHead(
            feature_dim, num_classes,
            hidden_dim=kwargs.get("hidden_dim", 256),
            dropout=kwargs.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")
