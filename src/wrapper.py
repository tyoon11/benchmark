"""
Model Wrapper
==============
 ECG encoder caseed is  downstream task in use.

Encoder :
  forward(x) → (sequence_features, pooled_features)
    - x:                  (batch, n_leads, seq_len)
    - sequence_features:  (batch, seq_len', embed_dim) or None
    - pooled_features:    (batch, feature_dim)

   only encoder pooled only returnthen sequence_features=None available.
   only encoder one of tensor only returnthen automatically GAP apply.

Eval modes (left column is this repo, right column the original CLI value):
  - linear_probe        == --eval-mode linear              (frozen encoder + Linear head)
  - attention_probe     == --eval-mode frozen              (frozen encoder + attention pooling head)
  - finetune_linear     == --eval-mode finetuning_linear
  - finetune_attention  == --eval-mode finetuning_nonlinear

Frozen-mode semantics
---------------------
The original wrappers call ``encoder.eval()`` once in ``__init__`` but never
override ``nn.Module.train()``, so Lightning's per-epoch ``model.train()`` puts
the "frozen" encoder back into **train mode**: dropout stays active and
BatchNorm running statistics keep updating even though the weights are frozen.
That materially changes linear-probe results for BatchNorm/dropout backbones
(MERL, ECGFM-KED, HuBERT-ECG, ST-MEM), so ``paper_faithful_frozen=True``
(the default) reproduces it. Set it to False for the arguably-more-correct
behaviour of holding the encoder in eval mode.
"""

import inspect

import torch
import torch.nn as nn
from .heads import build_head


EVAL_MODES = [
    "linear_probe",
    "attention_probe",
    "finetune_linear",
    "finetune_attention",
]

# this repo's eval_mode -> original --eval-mode
ORIGINAL_EVAL_MODE = {
    "linear_probe": "linear",
    "attention_probe": "frozen",
    "finetune_linear": "finetuning_linear",
    "finetune_attention": "finetuning_nonlinear",
}


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
        paper_faithful_frozen: keep the frozen encoder in train mode during
                        training, as the original does (see module docstring).
    """

    def __init__(
        self,
        encoder:         nn.Module,
        feature_dim:     int,
        num_classes:     int,
        eval_mode:       str = "linear_probe",
        seq_feature_dim: int = None,
        head_kwargs:     dict = None,
        paper_faithful_frozen: bool = True,
    ):
        super().__init__()
        assert eval_mode in EVAL_MODES, f"eval_mode must be one of {EVAL_MODES}"

        self.encoder = encoder
        self.feature_dim = feature_dim
        self.seq_feature_dim = seq_feature_dim or feature_dim
        self.num_classes = num_classes
        self.eval_mode = eval_mode
        self.is_frozen = eval_mode in ("linear_probe", "attention_probe")
        self.paper_faithful_frozen = paper_faithful_frozen

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

        # Pre-compute which extra kwargs the encoder.forward will accept.
        # Most encoders only take `x`; moryecg additionally takes `ecg_filepath`
        # and `ecg_seg_idx` for cache lookup. Filtering here lets the trainer
        # always pass them without breaking other encoders.
        try:
            sig = inspect.signature(self.encoder.forward)
            params = sig.parameters
            self._encoder_accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            self._encoder_param_names = set(params.keys())
        except (TypeError, ValueError):
            self._encoder_accepts_var_kw = True
            self._encoder_param_names = set()

    def _filter_kwargs(self, kwargs):
        """Drop kwargs the encoder.forward does not accept."""
        if self._encoder_accepts_var_kw:
            return kwargs
        return {k: v for k, v in kwargs.items() if k in self._encoder_param_names}

    def forward(self, x, **kwargs):
        """
        x: (batch, n_leads, seq_len)
        → (batch, num_classes)
        """
        x = torch.nan_to_num(x)

        kwargs = self._filter_kwargs(kwargs)

        # Encoder forward
        if self.is_frozen:
            with torch.no_grad():
                enc_out = self.encoder(x, **kwargs)
        else:
            enc_out = self.encoder(x, **kwargs)

        # Encoder output parsing
        seq_feat, pooled_feat = self._parse_encoder_output(enc_out)

        # Head
        if self.use_seq_features and seq_feat is not None:
            logits = self.head(seq_feat)
        else:
            logits = self.head(pooled_feat)

        return torch.nan_to_num(logits)

    def _parse_encoder_output(self, enc_out):
        """
        Encoder output (sequence_features, pooled_features) by normalization.

        support :
          1. tuple (seq_feat, pooled_feat) → as-is
          2. dict {"seq": ..., "pooled": ...} → extract
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
         per discriminative LR parameter groups.

        Encoder get_layer_groups() then 3stage LR apply:
          - Head:                lr
          - Predictor/Top:       lr × factor
          - Encoder/Early:       lr × factor²

         at 2stage:
          - Head:    lr
          - Encoder: lr × factor
        """
        head_params = list(self.head.parameters())
        f = discriminative_lr_factor

        if self.is_frozen:
            return [{"params": head_params, "lr": lr}]

        # encoder get_layer_groups() then use
        if hasattr(self.encoder, "get_layer_groups"):
            groups = self.encoder.get_layer_groups()
            # groups: {"early": [...], "late": [...]} or list of param lists
            if isinstance(groups, dict):
                early = groups.get("early", [])
                late = groups.get("late", [])
                return [
                    {"params": head_params,  "lr": lr},
                    {"params": late,         "lr": lr * f},
                    {"params": early,        "lr": lr * f * f},
                ]
            elif isinstance(groups, list) and len(groups) >= 2:
                result = [{"params": head_params, "lr": lr}]
                for i, g in enumerate(groups):
                    result.append({"params": g, "lr": lr * (f ** (i + 1))})
                return result

        # fallback: 2stage
        encoder_params = list(self.encoder.parameters())
        return [
            {"params": head_params, "lr": lr},
            {"params": encoder_params, "lr": lr * f},
        ]

    def train(self, mode=True):
        """Propagate train/eval mode.

        With ``paper_faithful_frozen`` the frozen encoder follows the wrapper
        into train mode — matching the original, where nothing pins it to eval
        (dropout active, BatchNorm running stats still updating). Otherwise it
        is held in eval mode.
        """
        super().train(mode)
        if self.is_frozen and not self.paper_faithful_frozen:
            self.encoder.eval()
        return self
