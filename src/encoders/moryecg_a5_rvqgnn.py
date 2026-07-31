"""
MoRyECG **A5 / GNN-RVQ** Encoder Adapter (Axial S4D + MHA over patch tokens).
============================================================================
This is a *separate* adapter from `moryecg_a5.py`. That one loads the beat-based
single-VQ A5 backbone (`model_cfg["backbone"] == "axial_s4"`, forward over
beat/RR/STFT). This one loads the checkpoints produced by

    configs/pretrain/axial_s4_a5_rvqgnn_patch100.yaml
    -> training/pretrain/train_rvq_gnn.py
    -> models.s4.ecg_s4_rvq_model.AxialS4RVQModel  (backbone == "axial_s4_rvq")

The RVQ-A5 backbone consumes a grid of *discrete patch tokens* produced by the
frozen GNN-RVQ codec (hbkimi/rvq `gnn_patch100/best.pt`), NOT beats/RR/STFT:

    x (B, 12, 5000)  --frozen GNNRVQTokenizer.tokenize-->  codes (B, N, L, n_q)
    AxialS4RVQModel(codes) -> (B, 1 + N*L, D);  out[:, 0] = [CLS]/global.

Preprocessing MUST reproduce exactly what the frozen codec was trained on
(rvq.data.HEEDBWaveformDataset with rpeak_detection=false, overlap=false):

    align leads -> resample to 500 Hz -> center-crop/right-pad to 5000
                -> record_mad_normalize(scale=5.0, clip=8.0)

The benchmark already delivers signals in HEEDB lead order (same contract the
other moryecg adapters rely on), so no lead reordering is done here. We import
`record_mad_normalize` straight from the rvq repo so normalization is byte-exact.

Checkpoint layout:
  pretrain_ckpt :  .../pretrain_axial_s4_a5_rvqgnn_patch100/best.pt
                   contains ['model', 'model_cfg', 'tokenizer_ckpt'] with
                   model_cfg["backbone"] == "axial_s4_rvq".
  tokenizer_ckpt:  taken from ckpt['tokenizer_ckpt'] (override via
                   MORYECG_A5_TOKENIZER / tokenizer_ckpt=).
"""

from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .moryecg import _resolve_repo_root, _resolve_pool_mode, pool_tokens


def _import_rvq_a5(repo_root: Path):
    """Import AxialS4RVQModel + GNNRVQTokenizer from the MoRyECG repo."""
    repo = str(repo_root)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from models.s4.ecg_s4_rvq_model import AxialS4RVQModel      # noqa: E402
    from models.tokenizer.rvq_gnn_tokenizer import GNNRVQTokenizer  # noqa: E402
    return AxialS4RVQModel, GNNRVQTokenizer


def _get_record_mad(repo_root: Path):
    """Import rvq.data.record_mad_normalize (RVQ_REPO put on path by tokenizer)."""
    # GNNRVQTokenizer's _ensure_rvq_on_path already added RVQ_REPO to sys.path
    # by the time forward() runs, but import here defensively.
    try:
        from rvq.data import record_mad_normalize  # noqa: E402
        return record_mad_normalize
    except Exception:
        from models.tokenizer.rvq_gnn_tokenizer import _ensure_rvq_on_path
        _ensure_rvq_on_path()
        from rvq.data import record_mad_normalize  # noqa: E402
        return record_mad_normalize


class MoRyECGA5RVQGNNEncoder(nn.Module):
    """MoRyECG A5 (Axial S4D + MHA) over frozen GNN-RVQ patch tokens.

    Required class attrs (paper-fair contract, mirrors MoRyECGA5Encoder):
      chunk_seconds : 10.0  — pre-trained on 10-second windows
      model_fs      : 500   — Hz at which the codec/backbone operate
      model_seq_len : 5000  — chunk_seconds * model_fs (codec input_len)
      feature_dim   : d_model from model_cfg (384 for A5)
    """

    # Encoder contract. MoRyECG was pre-trained on the HEEDB channel order
    # (I,II,III,V1..V6,aVF,aVL,aVR) and its beat/STFT preprocessing cache is keyed
    # to that layout, so this adapter asks the dataset for HEEDB order while the
    # published baselines get the standard order. See src/leads.py.
    input_size = 10.0          # seconds
    lead_order = "heedb"
    chunk_seconds = 10.0       # deprecated alias for input_size
    model_fs = 500
    model_seq_len = 5000
    feature_dim = 384  # overridden per-instance from model_cfg["d_model"]

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        tokenizer_ckpt: Optional[str] = None,
        repo_root: Optional[str] = None,
        record_mad_scale: float = 5.0,
        record_mad_clip: float = 8.0,
        pool_mode: Optional[str] = None,
    ):
        super().__init__()

        self.pool_mode = _resolve_pool_mode(pool_mode)
        repo = _resolve_repo_root(repo_root)
        AxialS4RVQModel, GNNRVQTokenizer = _import_rvq_a5(repo)

        if checkpoint is None:
            raise ValueError(
                "MoRyECGA5RVQGNNEncoder requires "
                "checkpoint=.../pretrain_axial_s4_a5_rvqgnn_patch100/best.pt"
            )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model_cfg" not in ckpt or "model" not in ckpt:
            raise ValueError(
                f"Pretrain checkpoint at {checkpoint} is missing 'model' or 'model_cfg'."
            )
        model_cfg = dict(ckpt["model_cfg"])
        self.model_cfg = model_cfg

        backbone = str(model_cfg.get("backbone", "")).lower()
        if backbone != "axial_s4_rvq":
            raise ValueError(
                f"MoRyECGA5RVQGNNEncoder expects model_cfg['backbone']=='axial_s4_rvq', "
                f"got '{backbone}'. Use MoRyECGA5Encoder (src.encoders.moryecg_a5) for "
                "the beat-based single-VQ A5 backbone ('axial_s4')."
            )

        self.feature_dim = int(model_cfg.get("d_model", self.feature_dim))
        self.record_mad_scale = float(record_mad_scale)
        self.record_mad_clip = float(record_mad_clip)

        # ── Frozen GNN-RVQ tokenizer ────────────────────────────────────────
        if tokenizer_ckpt is None:
            tokenizer_ckpt = (
                os.environ.get("MORYECG_A5_TOKENIZER")
                or ckpt.get("tokenizer_ckpt")
            )
        if not tokenizer_ckpt or not os.path.exists(tokenizer_ckpt):
            raise FileNotFoundError(
                f"GNN-RVQ tokenizer checkpoint not found: {tokenizer_ckpt!r}. "
                "Set MORYECG_A5_TOKENIZER or pass tokenizer_ckpt=."
            )
        self.tokenizer = GNNRVQTokenizer(tokenizer_ckpt, map_location="cpu")
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        self._record_mad = _get_record_mad(repo)

        # ── Pretrained RVQ-A5 backbone ──────────────────────────────────────
        self.model = AxialS4RVQModel(model_cfg)
        miss, unx = self.model.load_state_dict(ckpt["model"], strict=False)
        if miss:
            warnings.warn(f"[MoRyECGA5RVQGNNEncoder] model missing keys: {len(miss)} "
                          f"(e.g. {list(miss)[:4]})")
        if unx:
            warnings.warn(f"[MoRyECGA5RVQGNNEncoder] model unexpected keys: {len(unx)} "
                          f"(e.g. {list(unx)[:4]})")

        tmeta = self.tokenizer.meta
        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"[MoRyECGA5RVQGNNEncoder] backbone=axial_s4_rvq  d_model={self.feature_dim}  "
            f"layers={model_cfg.get('num_layers')}  params={n_params/1e6:.1f}M  "
            f"pretrain_epoch={ckpt.get('epoch', '?')}  "
            f"tokenizer(patch={tmeta.patch_count},leads={tmeta.n_leads},"
            f"n_q={tmeta.n_q},cb={tmeta.codebook_size},input_len={tmeta.input_len})"
        )

    # ── waveform -> codec input (record_mad, byte-exact with rvq training) ────
    def _prep_waveform(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, T) raw at the encoder contract rate -> (B, 12, 5000) normalized float."""
        x = torch.nan_to_num(x)
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(
                x, size=self.model_seq_len, mode="linear", align_corners=False
            )
        # record_mad is a per-record numpy op (global median + per-lead MAD);
        # replicate rvq.data exactly on CPU, per sample.
        x_np = x.detach().to("cpu", torch.float32).numpy()
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.empty_like(x_np)
        for i in range(x_np.shape[0]):
            out[i] = self._record_mad(
                x_np[i], self.record_mad_scale, self.record_mad_clip
            )
        return torch.from_numpy(out)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, **_unused):
        """
        x: (B, 12, T) raw ECG at task target_fs (HEEDB lead order).
        Returns: (sequence_features (B, N*L, D), pooled (B, D)).
        pooled = learned [CLS]/global token (out[:, 0]) when pool_mode=="cls",
        else the mean over the content tokens (pool_mode=="mean").
        """
        device = x.device
        xin = self._prep_waveform(x).to(device, non_blocking=True)

        with torch.no_grad():
            self.tokenizer.eval()
            codes = self.tokenizer.tokenize(xin)   # (B, N, L, n_q) long

        out = self.model(codes)                    # (B, 1 + N*L, D)
        seq_feat, pooled = pool_tokens(out, self.pool_mode)
        return seq_feat, pooled

    # ── discriminative LR groups (only used by finetune_* modes) ─────────────
    def get_layer_groups(self):
        """Split params into {early, late} for discriminative fine-tune LR.

        Embeddings (code/patch/lead) + first half of blocks -> early; second
        half of blocks + [CLS] aggregator + final norm -> late.
        """
        early, late = [], []
        for mod in (self.model.code_emb, self.model.patch_pos_emb, self.model.lead_emb):
            early += list(mod.parameters())
        n = len(self.model.blocks)
        half = n // 2
        for blk in self.model.blocks[:half]:
            early += list(blk.parameters())
        for blk in self.model.blocks[half:]:
            late += list(blk.parameters())
        late += [self.model.glob_token]
        late += list(self.model.glob_agg.parameters())
        late += list(self.model.norm.parameters())
        return [
            {"params": early, "name": "early"},
            {"params": late, "name": "late"},
        ]
