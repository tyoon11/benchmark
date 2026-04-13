"""
ECG-FM Encoder Adapter
=======================
Wav2Vec2 스타일 ECG encoder. fairseq_signals 의존성 없이 직접 구현.
feature_extractor (4 conv layers) + transformer encoder (12 layers)

Model: 12-lead, embed_dim=768, 12 layers, 12 heads
Checkpoint: mimic_iv_ecg_physionet_pretrained.pt
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFeatureExtractor(nn.Module):
    """Wav2Vec2 스타일 CNN feature extractor"""
    def __init__(self, in_channels=12, conv_layers=None):
        super().__init__()
        if conv_layers is None:
            conv_layers = [(256, 2, 2)] * 4  # (dim, kernel, stride)

        layers = nn.ModuleList()
        in_d = in_channels
        for i, (dim, k, s) in enumerate(conv_layers):
            layers.append(nn.Sequential(
                nn.Conv1d(in_d, dim, k, stride=s, bias=False),
                nn.Dropout(0.0),
                nn.GroupNorm(1, dim) if i == 0 else nn.Identity(),
                nn.GELU() if i == 0 else nn.GELU(),
            ))
            in_d = dim
        self.conv_layers = layers
        self.out_dim = in_d

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ConvPositionalEncoding(nn.Module):
    """Wav2Vec2 스타일 convolutional positional encoding (weight_norm 호환)"""
    def __init__(self, embed_dim=768, kernel_size=128, groups=16):
        super().__init__()
        conv = nn.Conv1d(embed_dim, embed_dim, kernel_size,
                         padding=kernel_size // 2, groups=groups)
        self.pos_conv = nn.Sequential(
            nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, T, C)
        x_t = x.transpose(1, 2)  # (B, C, T)
        x_t = self.pos_conv[0](x_t)
        x_t = x_t[:, :, :x.shape[1]]  # trim
        x_t = self.activation(x_t)
        return x + x_t.transpose(1, 2)


class SelfAttention(nn.Module):
    """Separate q/k/v projections (fairseq 호환)"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, ffn_dim=3072, num_heads=12,
                 dropout=0.0, attention_dropout=0.0, activation_dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        return x


class ECGFMModel(nn.Module):
    """ECG-FM (Wav2Vec2-based ECG Transformer)"""
    def __init__(self, in_channels=12, embed_dim=768, ffn_dim=3072,
                 num_heads=12, num_layers=12, conv_layers=None):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(in_channels, conv_layers)
        feat_dim = self.feature_extractor.out_dim

        self.layer_norm = nn.LayerNorm(feat_dim)
        self.post_extract_proj = nn.Linear(feat_dim, embed_dim)
        self.conv_pos = ConvPositionalEncoding(embed_dim, kernel_size=128, groups=16)

        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, ffn_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.encoder.layer_norm = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: (B, 12, T)
        Returns: (B, T', embed_dim) sequence features
        """
        # Feature extraction
        features = self.feature_extractor(x)  # (B, feat_dim, T')
        features = features.transpose(1, 2)    # (B, T', feat_dim)
        features = self.layer_norm(features)          # (B, T', feat_dim)
        features = self.post_extract_proj(features)  # (B, T', embed_dim)

        # Positional encoding
        features = self.conv_pos(features)

        # Transformer
        for layer in self.encoder.layers:
            features = layer(features)

        features = self.encoder.layer_norm(features)
        return features  # (B, T', embed_dim)


class ECGFMEncoder(nn.Module):
    """
    ECG-FM encoder wrapper for benchmark.

    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, seq_len) at 500Hz
      - sequence_features: (B, T', 768)
      - pooled_features: (B, 768) via GAP
    """

    def __init__(self, checkpoint=None):
        super().__init__()
        self.model = ECGFMModel(
            in_channels=12,
            embed_dim=768,
            ffn_dim=3072,
            num_heads=12,
            num_layers=12,
            conv_layers=[(256, 2, 2)] * 4,
        )
        self.feature_dim = 768

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        # fairseq 키 → 우리 모델 키 매핑
        new_state = {}
        skip_prefixes = ("mask_emb", "quantizer.", "project_q.", "final_proj.")
        for k, v in state.items():
            if k in ("mask_emb",) or any(k.startswith(p) for p in skip_prefixes):
                continue
            # conv_pos: pos_conv.0.* → pos_conv.0.parametrizations.weight.original0/1
            if k == "conv_pos.pos_conv.0.weight_g":
                new_state["conv_pos.pos_conv.0.parametrizations.weight.original0"] = v
                continue
            if k == "conv_pos.pos_conv.0.weight_v":
                new_state["conv_pos.pos_conv.0.parametrizations.weight.original1"] = v
                continue
            if k == "conv_pos.pos_conv.0.bias":
                new_state["conv_pos.pos_conv.0.bias"] = v
                continue
            new_state[k] = v

        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        print(f"[ECGFMEncoder] Loaded from {path}")
        if missing:
            print(f"  Missing: {len(missing)} keys ({missing[:3]}...)")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)} keys ({unexpected[:3]}...)")

    def forward(self, x):
        x = torch.nan_to_num(x)
        seq_feat = self.model(x)             # (B, T', 768)
        pooled = seq_feat.mean(dim=1)        # (B, 768)
        return seq_feat, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in ["feature_extractor", "layer_norm",
                   "post_extract_proj", "conv_pos"]) or \
               any(name.startswith(f"encoder.layers.{i}.") for i in range(6)):
                early.append(param)
            else:
                late.append(param)
        return {"early": early, "late": late}
