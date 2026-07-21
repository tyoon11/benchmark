"""
MoRyECG **A5** Encoder Adapter (Axial S4D + MHA backbone) for the benchmark.
====================================================================
This is a *separate* adapter from `moryecg.py` (the v4/v5/v6 transformer
family). It loads checkpoints produced by
    configs/pretrain/axial_s4_a5_heedb_full_cb1024.yaml
i.e. model_cfg["backbone"] == "axial_s4", built by models.s4.ecg_s4_model.AxialS4ECGModel.

Why a dedicated adapter (rather than extending MoRyECGEncoder):
  * The A5 backbone is d_model=384 (vs 512 for the transformer family), so
    feature_dim differs.
  * AxialS4ECGModel exposes the *same* (B, 1 + N*L, D) output layout as the
    v4 ECGFoundationModel — out[:, 0] = pooled [GLOB], out[:, 1:] = beat
    tokens — and takes the same positional forward(indices, rr, stft). So the
    forward path here is exactly the v4 path; none of the moryecg/v6
    forward_flat / beat_valid_mask machinery is needed.

The heavy, version-independent preprocessing (R-peak → beat → resample →
record_mad → STFT, plus the .npz disk cache) is reused verbatim from
moryecg.py so the two adapters stay byte-for-byte consistent on inputs.

Checkpoint layout:
  pretrain_ckpt :  .../pretrain_axial_s4_a5_heedb_full_cb1024/best.pt
                   must contain ['model', 'model_cfg'] with
                   model_cfg["backbone"] == "axial_s4".
  tokenizer_ckpt:  .../tokenizer_heedb_full_cb{K}_v4/best.pt
                   auto-derived from pretrain_ckpt path when not given.
"""

from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the shared, version-independent preprocessing + repo-resolution helpers.
from .moryecg import (
    _resolve_repo_root,
    _import_pretrain_modules,
    _autodetect_tokenizer_ckpt,
    _resolve_cache_root,
    preprocess_signal,
    cache_path,
    load_cache,
    MODEL_FS,
    MODEL_SEQ_LEN,
    N_LEADS,
    BEAT_LENGTH,
    DEFAULT_MAX_BEATS,
    DEFAULT_RECORD_MAD_SCALE,
)


def _resolve_tokenizer_ckpt(checkpoint: str, codebook_size: int) -> str:
    """Locate the frozen VQ tokenizer for an A5 pretrain checkpoint.

    moryecg._autodetect_tokenizer_ckpt assumes the layout
    `checkpoints/<pretrain_dir>/best.pt` (tokenizer one level up). That breaks
    when the checkpoint is nested deeper — e.g. an archived run at
    `checkpoints/<pretrain_dir>/_run1_archive_*/best.pt`. So we (1) honor an
    explicit env override, (2) try the standard autodetect, then (3) walk up
    the ancestor dirs looking for `tokenizer_heedb_full_cb{K}_v4/best.pt`.
    """
    env = os.environ.get("MORYECG_A5_TOKENIZER") or os.environ.get("MORYECG_TOKENIZER")
    if env:
        return env
    try:
        return _autodetect_tokenizer_ckpt(checkpoint, codebook_size)
    except FileNotFoundError:
        pass
    name = f"tokenizer_heedb_full_cb{codebook_size}_v4"
    for ancestor in Path(checkpoint).resolve().parents:
        cand = ancestor / name / "best.pt"
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"Could not locate tokenizer '{name}/best.pt' near {checkpoint}. "
        "Pass tokenizer_ckpt= or set MORYECG_A5_TOKENIZER."
    )


def _import_axial_model(repo_root: Path):
    """Import AxialS4ECGModel (A5 backbone) from the MoRyECG repo."""
    import sys
    repo_root = str(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from models.s4.ecg_s4_model import AxialS4ECGModel  # noqa
    return AxialS4ECGModel


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────
class MoRyECGA5Encoder(nn.Module):
    """MoRyECG A5 (Axial S4D + MHA) encoder for the downstream benchmark.

    Required class attrs (paper-fair contract):
      chunk_seconds : 10.0  — pre-trained on 10-second windows
      model_fs      : 500   — Hz at which preprocessing operates
      model_seq_len : 5000  — chunk_seconds * model_fs
      feature_dim   : set per-instance from model_cfg["d_model"] (384 for A5)
    """

    chunk_seconds = 10.0
    model_fs = MODEL_FS
    model_seq_len = MODEL_SEQ_LEN
    feature_dim = 384  # instance __init__ overrides from model_cfg["d_model"]

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        tokenizer_ckpt: Optional[str] = None,
        repo_root: Optional[str] = None,
        cache_root: Optional[str] = None,
    ):
        super().__init__()

        repo = _resolve_repo_root(repo_root)
        mods = _import_pretrain_modules(repo)   # VQVAE + preprocessing fns
        self._mods = mods
        AxialS4ECGModel = _import_axial_model(repo)

        if checkpoint is None:
            raise ValueError(
                "MoRyECGA5Encoder requires "
                "checkpoint=.../pretrain_axial_s4_a5_heedb_full_cb1024/best.pt"
            )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model_cfg" not in ckpt or "model" not in ckpt:
            raise ValueError(
                f"Pretrain checkpoint at {checkpoint} is missing 'model' or 'model_cfg'."
            )
        model_cfg = dict(ckpt["model_cfg"])
        self.model_cfg = model_cfg

        backbone = str(model_cfg.get("backbone", "")).lower()
        if backbone != "axial_s4":
            raise ValueError(
                f"MoRyECGA5Encoder expects model_cfg['backbone']=='axial_s4', got "
                f"'{backbone}'. Use MoRyECGEncoder (src.encoders.moryecg) for the "
                "transformer family (v4/v5/v6)."
            )

        self.feature_dim = int(model_cfg.get("d_model", self.feature_dim))
        self.codebook_size = int(model_cfg["codebook_size"])
        self.max_beats = int(model_cfg.get("max_beats", DEFAULT_MAX_BEATS))
        self.normalize_mode = str(model_cfg.get("normalize", "record_mad"))
        self.record_mad_scale = float(
            model_cfg.get("record_mad_scale", DEFAULT_RECORD_MAD_SCALE)
        )

        # ── Frozen tokenizer ────────────────────────────────────────────────
        if tokenizer_ckpt is None:
            tokenizer_ckpt = _resolve_tokenizer_ckpt(checkpoint, self.codebook_size)
        tok_ckpt = torch.load(tokenizer_ckpt, map_location="cpu", weights_only=False)
        VQVAE = mods["VQVAE"]
        self.tokenizer = VQVAE(tok_ckpt["model_cfg"])
        miss, unx = self.tokenizer.load_state_dict(tok_ckpt["model"], strict=False)
        if miss:
            warnings.warn(f"[MoRyECGA5Encoder] tokenizer missing keys: {len(miss)}")
        if unx:
            warnings.warn(f"[MoRyECGA5Encoder] tokenizer unexpected keys: {len(unx)}")
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        self.tokenizer.eval()

        # ── Pretrained A5 axial S4 backbone ─────────────────────────────────
        self.model = AxialS4ECGModel(model_cfg)
        miss, unx = self.model.load_state_dict(ckpt["model"], strict=False)
        if miss:
            warnings.warn(f"[MoRyECGA5Encoder] model missing keys: {len(miss)}")
        if unx:
            warnings.warn(f"[MoRyECGA5Encoder] model unexpected keys: {len(unx)}")

        # ── Cache root ──────────────────────────────────────────────────────
        self.cache_root = _resolve_cache_root(cache_root)
        self._cache_hits = 0
        self._cache_misses = 0

        n_params = sum(p.numel() for p in self.model.parameters())
        cache_str = f"cache={self.cache_root}" if self.cache_root else "cache=off"
        print(
            f"[MoRyECGA5Encoder] backbone=axial_s4  codebook={self.codebook_size}  "
            f"d_model={model_cfg['d_model']}  layers={model_cfg['num_layers']}  "
            f"params={n_params/1e6:.1f}M  pretrain_epoch={ckpt.get('epoch', '?')}  "
            f"{cache_str}"
        )

    # ── batch preprocessing (cache-first, live fallback) ─────────────────────
    def _preprocess_batch(
        self,
        x: torch.Tensor,
        ecg_filepath: Optional[list] = None,
        ecg_seg_idx: Optional[list] = None,
    ):
        """x: (B, 12, T). For each sample: try cache, else live preprocess."""
        B = x.shape[0]

        if x.shape[-1] != self.model_seq_len:
            x_resampled = F.interpolate(
                x, size=self.model_seq_len, mode="linear", align_corners=False
            )
        else:
            x_resampled = x
        x_np_lazy = None

        beats_b = rr_b = stft_b = None
        n_valid_list = []

        if ecg_seg_idx is None:
            seg_list = [0] * B
        elif isinstance(ecg_seg_idx, torch.Tensor):
            seg_list = ecg_seg_idx.detach().cpu().tolist()
        else:
            seg_list = list(ecg_seg_idx)

        for i in range(B):
            bundle = None
            if self.cache_root is not None and ecg_filepath is not None:
                fp = ecg_filepath[i] if i < len(ecg_filepath) else None
                if fp:
                    bundle = load_cache(cache_path(self.cache_root, fp, seg_list[i]))
                    if bundle is not None:
                        self._cache_hits += 1
            if bundle is None:
                self._cache_misses += 1
                if x_np_lazy is None:
                    x_np_lazy = x_resampled.detach().to("cpu", torch.float32).numpy()
                    x_np_lazy = np.nan_to_num(x_np_lazy, nan=0.0, posinf=0.0, neginf=0.0)
                bundle = preprocess_signal(
                    x_np_lazy[i], self._mods,
                    max_beats=self.max_beats,
                    normalize_mode=self.normalize_mode,
                    record_mad_scale=self.record_mad_scale,
                )

            if beats_b is None:
                F_, T_ = bundle["stft"].shape[1], bundle["stft"].shape[2]
                beats_b = np.zeros((B, self.max_beats, N_LEADS, BEAT_LENGTH), dtype=np.float32)
                rr_b = np.zeros((B, self.max_beats, N_LEADS, 3), dtype=np.float32)
                stft_b = np.zeros((B, N_LEADS, F_, T_), dtype=np.float32)
            beats_b[i] = bundle["beats"]
            rr_b[i] = bundle["rr_feats"]
            sb = bundle["stft"]
            if sb.shape != stft_b[i].shape:
                Tt = min(sb.shape[2], stft_b[i].shape[2])
                stft_b[i, :, :, :Tt] = sb[:, : stft_b[i].shape[1], :Tt]
            else:
                stft_b[i] = sb
            n_valid_list.append(bundle["n_valid"])

        return beats_b, rr_b, stft_b, n_valid_list

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        ecg_filepath: Optional[list] = None,
        ecg_seg_idx: Optional[list] = None,
        cached_beats: Optional[torch.Tensor] = None,
        cached_rr: Optional[torch.Tensor] = None,
        cached_stft: Optional[torch.Tensor] = None,
        cached_n_valid=None,
        **_unused,
    ):
        """
        x: (B, 12, T) raw ECG at task target_fs.
        Returns: (sequence_features (B, max_beats*12, D), pooled (B, D)).

        The axial S4 model treats zero-padded beats as valid (it has no
        beat-mask in its forward), matching how it was pre-trained — so no
        beat_valid_mask is constructed here.
        """
        x = torch.nan_to_num(x)
        device = x.device

        if cached_beats is not None:
            self._cache_hits += int(cached_beats.shape[0])
            beats = cached_beats.to(device, dtype=torch.float32, non_blocking=True)
            rr = cached_rr.to(device, dtype=torch.float32, non_blocking=True)
            stft = cached_stft.to(device, dtype=torch.float32, non_blocking=True)
        else:
            beats_np, rr_np, stft_np, _ = self._preprocess_batch(
                x, ecg_filepath=ecg_filepath, ecg_seg_idx=ecg_seg_idx,
            )
            beats = torch.from_numpy(beats_np).to(device, non_blocking=True)
            rr = torch.from_numpy(rr_np).to(device, non_blocking=True)
            stft = torch.from_numpy(stft_np).to(device, non_blocking=True)

        # Tokenizer encode (frozen, no_grad)
        B, N, L, W = beats.shape
        with torch.no_grad():
            self.tokenizer.eval()
            beats_flat = beats.view(B * N * L, 1, W)
            _zq, idx_flat = self.tokenizer.encode(beats_flat)
        indices = idx_flat.view(B, N, L).long()

        out = self.model(indices, rr, stft)   # (B, 1 + N*L, D)
        pooled = out[:, 0, :]
        seq_feat = out[:, 1:, :]
        return seq_feat, pooled

    # ── layer-dependent LR groups (paper finetune contract) ──────────────────
    def get_layer_groups(self):
        """Split params into {early, late} for discriminative fine-tune LR.

        Embeddings + STFT context → early; the axial block stack is split in
        half; the [GLOB] cross-attn aggregator + final norm → late.
        """
        early, late = [], []
        for mod in (self.model.morph_emb, self.model.lead_emb,
                    self.model.pos_emb, self.model.rhythm_mlp,
                    self.model.global_ctx):
            for p in mod.parameters():
                early.append(p)
        n_layers = len(self.model.blocks)
        split = n_layers // 2
        for i, blk in enumerate(self.model.blocks):
            grp = early if i < split else late
            for p in blk.parameters():
                grp.append(p)
        for p in self.model.glob_agg.parameters():
            late.append(p)
        for p in self.model.norm.parameters():
            late.append(p)
        return {"early": early, "late": late}

    # ── debug helper ──────────────────────────────────────────────────────────
    def cache_stats(self) -> str:
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return "no preprocessing yet"
        hit_pct = 100.0 * self._cache_hits / total
        return (f"cache hits={self._cache_hits} misses={self._cache_misses} "
                f"({hit_pct:.1f}%)")
