"""
MoRyECG Encoder Adapter for the downstream benchmark.
====================================================================
Beat-tokenized 12-lead ECG Transformer (Pre-LN, d_model=512, 12 layers).

Pipeline (raw 12-lead ECG -> pooled record-level embedding):
  (B, 12, T) at fs ──► R-peak detect (Lead II, neurokit2)
                  ──► extract beats (±before/after_ms around R)
                  ──► resample each beat → 256 samples
                  ──► record_mad normalization (preserves inter-lead amplitude)
                  ──► VQ-VAE encode (frozen) → indices  (B, N, 12)
                  ──► RR features                       (B, N, 12, 3) [seconds]
                  ──► STFT log-magnitude                (B, 12, F, T')
                  ──► ECGFoundationModel transformer    (B, 1+N*12, 512)
                  ──► out[:, 0]   = pooled GLOB         (B, 512)
                      out[:, 1:]  = sequence features   (B, N*12, 512)

The repository housing the pretraining code (model + preprocessing) is
discovered in this order:
  1. constructor argument `repo_root`
  2. environment variable MORYECG_REPO
  3. the parent of the current benchmark/ directory  (default — works when
     the benchmark folder lives inside the MoRyECG repository, as in the
     release layout)

Caching of CPU preprocessing (R-peak detection is the bottleneck):
  set MORYECG_CACHE=<dir> and run a one-off precompute script.  When
  forward() receives ecg_filepath= kwarg AND a cache file exists, the
  encoder skips neurokit2 / numpy preprocessing entirely and loads
  beats / rr / stft from the .npz sidecar.  Cache miss falls back to live
  computation.

Checkpoint layout (any codebook size works):
  pretrain_ckpt :  .../pretrain_heedb_cb{K}_v4/best.pt   (or last.pt / epoch_NNN.pt)
                   must contain ['model', 'model_cfg'] with model_cfg.codebook_size=K
  tokenizer_ckpt:  .../tokenizer_heedb_full_cb{K}_v4/best.pt
                   auto-derived from pretrain_ckpt path when not given.
"""

from __future__ import annotations
import hashlib
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing constants — must match the pretrain config (v4)
# ──────────────────────────────────────────────────────────────────────────────
PREPROC_VERSION = "v4"
MODEL_FS = 500
MODEL_SEQ_LEN = 5000
N_LEADS = 12
BEAT_LENGTH = 256
BEFORE_MS = 200
AFTER_MS = 400
STFT_N_FFT = 256
STFT_HOP = 64
LEAD_II_INDEX = 1   # HEEDB lead order: I, II, III, V1..V6, aVF, aVL, aVR
DEFAULT_MAX_BEATS = 30
DEFAULT_RECORD_MAD_SCALE = 5.0

# Token-aggregation for the pooled (linear_probe / finetune_linear) feature
# vector. "mean" (the default) averages the content tokens (out[:, 1:]) the way
# the mean-pooling baselines (ecg_founder, merl, st_mem, …) build their pooled
# vector — so MoRyECG is compared on the same footing. "cls" uses the learned
# [GLOB]/[CLS] token (out[:, 0]). Env override ($MORYECG_POOL_MODE) lets run
# scripts flip the mode without editing configs.
POOL_MODES = ("cls", "mean")


def _resolve_pool_mode(explicit: Optional[str] = None) -> str:
    """Pick pooling mode from an explicit arg, else $MORYECG_POOL_MODE, else 'mean'."""
    mode = (explicit or os.environ.get("MORYECG_POOL_MODE") or "mean").lower()
    if mode not in POOL_MODES:
        raise ValueError(f"pool_mode must be one of {POOL_MODES}, got '{mode}'.")
    return mode


def pool_tokens(out: torch.Tensor, pool_mode: str) -> tuple:
    """Split a (B, 1 + M, D) backbone output into (seq_feat, pooled).

    seq_feat is always the content tokens out[:, 1:]. pooled is the [CLS]/[GLOB]
    token out[:, 0] when pool_mode == "cls", or the mean over the content tokens
    when pool_mode == "mean" (unmasked, matching the mean-pooling baselines).
    """
    seq_feat = out[:, 1:, :]
    if pool_mode == "mean":
        pooled = seq_feat.mean(dim=1)
    else:
        pooled = out[:, 0, :]
    return seq_feat, pooled


# ─── Locate the pretrain repo so we can reuse its model + preprocessing code ──
def _resolve_repo_root(explicit: Optional[str] = None) -> Path:
    """Find the MoRyECG repo that contains models/transformer/ecg_model.py."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MORYECG_REPO"):
        candidates.append(Path(os.environ["MORYECG_REPO"]))
    # Default: benchmark/ lives inside the MoRyECG repo, so the repo root is
    # benchmark/.parent (i.e. the directory containing models/, training/, etc).
    here = Path(__file__).resolve()
    candidates.append(here.parent.parent.parent.parent)   # .../benchmark/src/encoders/moryecg.py -> repo root
    for p in candidates:
        if (p / "models" / "transformer" / "ecg_model.py").exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find the MoRyECG pretrain repo. Set MORYECG_REPO or pass "
        f"repo_root=. Tried: {candidates}"
    )


def _import_pretrain_modules(repo_root: Path):
    """Lazily import ECGFoundationModel / VQVAE + preprocessing helpers."""
    repo_root = str(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from models.transformer.ecg_model import ECGFoundationModel  # noqa
    from models.tokenizer.vqvae import VQVAE  # noqa
    from data.preprocessing.beat_segmentor import process_ecg_record  # noqa
    from data.preprocessing.resampler import (
        resample_beat,
        compute_record_norm_stats,
        apply_record_norm,
    )  # noqa
    from data.preprocessing.stft_extractor import compute_stft_map  # noqa

    # Optional variants — only required when loading checkpoints with
    # model_cfg["arch"] in {"moryecg", "moryecg_v6"}.  Importing them eagerly
    # keeps the adapter's behavior independent of which version is on disk.
    try:
        from models.transformer.moryecg_model import MoRyECGFoundationModel  # noqa
    except Exception as e:
        warnings.warn(f"[moryecg] MoRyECGFoundationModel unavailable: {e}")
        MoRyECGFoundationModel = None
    try:
        from models.transformer.moryecg_v6_model import MoRyECGv6FoundationModel  # noqa
    except Exception as e:
        warnings.warn(f"[moryecg] MoRyECGv6FoundationModel unavailable: {e}")
        MoRyECGv6FoundationModel = None

    return dict(
        ECGFoundationModel=ECGFoundationModel,
        MoRyECGFoundationModel=MoRyECGFoundationModel,
        MoRyECGv6FoundationModel=MoRyECGv6FoundationModel,
        VQVAE=VQVAE,
        process_ecg_record=process_ecg_record,
        resample_beat=resample_beat,
        compute_record_norm_stats=compute_record_norm_stats,
        apply_record_norm=apply_record_norm,
        compute_stft_map=compute_stft_map,
    )


def _autodetect_tokenizer_ckpt(pretrain_ckpt_path: str, codebook_size: int) -> str:
    p = Path(pretrain_ckpt_path)
    sib = p.parent.parent / f"tokenizer_heedb_full_cb{codebook_size}_v4" / "best.pt"
    if sib.exists():
        return str(sib)
    raise FileNotFoundError(
        f"Could not auto-locate tokenizer checkpoint for codebook={codebook_size}. "
        f"Expected at {sib}. Pass tokenizer_ckpt= explicitly."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pure-function preprocessing (used by encoder live path AND precompute script)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_signal(
    sig: np.ndarray,
    pp_modules: dict,
    *,
    max_beats: int = DEFAULT_MAX_BEATS,
    normalize_mode: str = "record_mad",
    record_mad_scale: float = DEFAULT_RECORD_MAD_SCALE,
):
    """
    Run R-peak detect + beat extract + resample + record_mad + STFT for one
    12-lead ECG already at MODEL_FS=500 Hz with shape (12, T).

    Returns dict with float32 arrays:
        beats    : (max_beats, 12, BEAT_LENGTH=256)
        rr_feats : (max_beats, 12, 3)              # RR seconds (prev/next/median)
        stft     : (12, F=129, T')
        n_valid  : int  number of real (non-pad) beats
    """
    process_ecg_record = pp_modules["process_ecg_record"]
    resample_beat = pp_modules["resample_beat"]
    compute_record_norm_stats = pp_modules["compute_record_norm_stats"]
    apply_record_norm = pp_modules["apply_record_norm"]
    compute_stft_map = pp_modules["compute_stft_map"]

    L = N_LEADS
    W = BEAT_LENGTH
    N = max_beats

    # STFT first (always computed; no dependency on R-peaks)
    stft = compute_stft_map(sig, MODEL_FS, n_fft=STFT_N_FFT, hop_length=STFT_HOP)

    zero_beats = np.zeros((N, L, W), dtype=np.float32)
    zero_rr = np.zeros((N, L, 3), dtype=np.float32)

    try:
        result = process_ecg_record(
            sig, MODEL_FS,
            ref_lead_idx=LEAD_II_INDEX,
            before_ms=BEFORE_MS, after_ms=AFTER_MS,
            rpeak_method="neurokit",
        )
    except Exception:
        result = None

    if result is None or result["n_beats"] < 2:
        return {"beats": zero_beats, "rr_feats": zero_rr,
                "stft": stft.astype(np.float32), "n_valid": 0}

    beats_raw = result["beats"]
    rr_list = result["rr_feats"]
    Nraw = beats_raw.shape[0]
    n_valid = min(Nraw, N)

    if normalize_mode == "record_mad":
        rec_med, rec_mad = compute_record_norm_stats(sig)

    full_beats = zero_beats
    for b in range(n_valid):
        for l in range(L):
            seg = resample_beat(beats_raw[b, l, :], W)
            if normalize_mode == "record_mad":
                seg = apply_record_norm(seg, rec_med, rec_mad,
                                        scale=record_mad_scale)
            elif normalize_mode == "zscore":
                mu = seg.mean(); sd = seg.std() + 1e-8
                seg = (seg - mu) / sd
            full_beats[b, l, :] = seg

    full_rr = zero_rr
    for b in range(n_valid):
        rr = rr_list[b]
        v = np.array([rr["prev_rr"], rr["next_rr"], rr["median_rr"]],
                     dtype=np.float32)
        full_rr[b, :, :] = v[None, :]

    return {"beats": full_beats, "rr_feats": full_rr,
            "stft": stft.astype(np.float32), "n_valid": int(n_valid)}


# ──────────────────────────────────────────────────────────────────────────────
# Disk cache (sidecar .npz keyed by absolute H5 path + seg_idx)
# ──────────────────────────────────────────────────────────────────────────────
def cache_key(filepath: str, seg_idx: int = 0) -> str:
    """Stable 16-char SHA1 prefix of the absolute path + segment id."""
    raw = f"{os.path.abspath(filepath)}::seg{int(seg_idx)}::pp{PREPROC_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(cache_root: str, filepath: str, seg_idx: int = 0) -> Path:
    return Path(cache_root) / f"{cache_key(filepath, seg_idx)}.npz"


def save_cache(path: Path, bundle: dict) -> None:
    """Write bundle as float16 to halve disk; n_valid stays int."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        beats=bundle["beats"].astype(np.float16),
        rr_feats=bundle["rr_feats"].astype(np.float16),
        stft=bundle["stft"].astype(np.float16),
        n_valid=np.int32(bundle["n_valid"]),
    )


def load_cache(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with np.load(str(path)) as f:
            return {
                "beats": f["beats"].astype(np.float32),
                "rr_feats": f["rr_feats"].astype(np.float32),
                "stft": f["stft"].astype(np.float32),
                "n_valid": int(f["n_valid"]),
            }
    except Exception as e:
        warnings.warn(f"[moryecg] failed to load cache {path}: {e}")
        return None


def _resolve_cache_root(explicit: Optional[str] = None) -> Optional[Path]:
    """Cache root is opt-in via env var / constructor; None disables cache."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("MORYECG_CACHE")
    if env:
        return Path(env)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────
class MoRyECGEncoder(nn.Module):
    """
    MoRyECG encoder for the downstream benchmark.

    Required class attrs (paper-fair contract):
      chunk_seconds  : 10.0   — the model was pre-trained on 10-second windows
      model_fs       : 500    — Hz at which preprocessing operates
      model_seq_len  : 5000   — chunk_seconds * model_fs
      feature_dim    : 512    — GLOB / CLS embedding size
    """

    # Encoder contract. MoRyECG was pre-trained on the HEEDB channel order
    # (I,II,III,V1..V6,aVF,aVL,aVR) and its beat/STFT preprocessing cache is keyed
    # to that layout, so this adapter asks the dataset for HEEDB order while the
    # published baselines get the standard order. See src/leads.py.
    input_size = 10.0          # seconds
    lead_order = "heedb"
    chunk_seconds = 10.0       # deprecated alias for input_size
    model_fs = MODEL_FS
    model_seq_len = MODEL_SEQ_LEN
    feature_dim = 512

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        tokenizer_ckpt: Optional[str] = None,
        repo_root: Optional[str] = None,
        cache_root: Optional[str] = None,
        pool_mode: Optional[str] = None,
    ):
        super().__init__()

        self.pool_mode = _resolve_pool_mode(pool_mode)
        repo = _resolve_repo_root(repo_root)
        mods = _import_pretrain_modules(repo)
        self._mods = mods

        if checkpoint is None:
            raise ValueError(
                "MoRyECGEncoder requires checkpoint=path/to/pretrain_heedb_cb*_v4/best.pt"
            )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model_cfg" not in ckpt or "model" not in ckpt:
            raise ValueError(
                f"Pretrain checkpoint at {checkpoint} is missing 'model' or 'model_cfg'."
            )
        model_cfg = dict(ckpt["model_cfg"])
        self.model_cfg = model_cfg
        self.codebook_size = int(model_cfg["codebook_size"])
        self.max_beats = int(model_cfg.get("max_beats", DEFAULT_MAX_BEATS))
        self.normalize_mode = str(model_cfg.get("normalize", "record_mad"))
        self.record_mad_scale = float(
            model_cfg.get("record_mad_scale", DEFAULT_RECORD_MAD_SCALE)
        )

        # ── Frozen tokenizer ───────────────────────────────────────────────
        if tokenizer_ckpt is None:
            tokenizer_ckpt = _autodetect_tokenizer_ckpt(checkpoint, self.codebook_size)
        tok_ckpt = torch.load(tokenizer_ckpt, map_location="cpu", weights_only=False)
        VQVAE = mods["VQVAE"]
        self.tokenizer = VQVAE(tok_ckpt["model_cfg"])
        miss, unx = self.tokenizer.load_state_dict(tok_ckpt["model"], strict=False)
        if miss:
            warnings.warn(f"[MoRyECGEncoder] tokenizer missing keys: {len(miss)}")
        if unx:
            warnings.warn(f"[MoRyECGEncoder] tokenizer unexpected keys: {len(unx)}")
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        self.tokenizer.eval()

        # ── Pretrained MoRyECG transformer ──────────────────────────────────
        # arch routing:
        #   "moryecg_v6" → MoRyECGv6FoundationModel (v5 + final GlobalRefineBlock)
        #   "moryecg"    → MoRyECGFoundationModel (v5: factorized only)
        #   else         → v4 ECGFoundationModel (flat seq native)
        self.arch = str(model_cfg.get("arch", "v4_flat")).lower()
        if self.arch == "moryecg_v6":
            ModelCls = mods.get("MoRyECGv6FoundationModel")
            if ModelCls is None:
                raise RuntimeError(
                    "Checkpoint declares arch=moryecg_v6 but MoRyECGv6FoundationModel "
                    "is not importable from the MoRyECG repo."
                )
            self.model = ModelCls(model_cfg)
        elif self.arch == "moryecg":
            MoRyECGFoundationModel = mods.get("MoRyECGFoundationModel")
            if MoRyECGFoundationModel is None:
                raise RuntimeError(
                    "Checkpoint declares arch=moryecg but MoRyECGFoundationModel "
                    "is not importable from the MoRyECG repo."
                )
            self.model = MoRyECGFoundationModel(model_cfg)
        else:
            ECGFoundationModel = mods["ECGFoundationModel"]
            self.model = ECGFoundationModel(model_cfg)
        miss, unx = self.model.load_state_dict(ckpt["model"], strict=False)
        if miss:
            warnings.warn(f"[MoRyECGEncoder] model missing keys: {len(miss)}")
        if unx:
            warnings.warn(f"[MoRyECGEncoder] model unexpected keys: {len(unx)}")

        # ── Cache root ─────────────────────────────────────────────────────
        self.cache_root = _resolve_cache_root(cache_root)
        self._cache_hits = 0
        self._cache_misses = 0

        n_params = sum(p.numel() for p in self.model.parameters())
        cache_str = f"cache={self.cache_root}" if self.cache_root else "cache=off"
        print(
            f"[MoRyECGEncoder] codebook={self.codebook_size}  "
            f"d_model={model_cfg['d_model']}  layers={model_cfg['num_layers']}  "
            f"params={n_params/1e6:.1f}M  pretrain_epoch={ckpt.get('epoch', '?')}  "
            f"{cache_str}"
        )

    # ── batch preprocessing (cache-first, live fallback) ────────────────────
    def _preprocess_batch(
        self,
        x: torch.Tensor,
        ecg_filepath: Optional[list] = None,
        ecg_seg_idx: Optional[list] = None,
    ):
        """
        x: (B, 12, T). For each sample: try cache, else live preprocess.
        Returns (beats_b, rr_b, stft_b) as numpy arrays and n_valid list.
        """
        B = x.shape[0]

        # Resample to model_seq_len for the live path (cache stores at model_fs already)
        if x.shape[-1] != self.model_seq_len:
            x_resampled = F.interpolate(
                x, size=self.model_seq_len, mode="linear", align_corners=False
            )
        else:
            x_resampled = x
        x_np_lazy = None  # only realize numpy when we need live preprocessing

        beats_b = None
        rr_b = None
        stft_b = None
        n_valid_list = []

        # Normalize seg_idx into a list of ints
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

            # Lazy buffer alloc once we know STFT shape
            if beats_b is None:
                F_, T_ = bundle["stft"].shape[1], bundle["stft"].shape[2]
                beats_b = np.zeros(
                    (B, self.max_beats, N_LEADS, BEAT_LENGTH), dtype=np.float32
                )
                rr_b = np.zeros((B, self.max_beats, N_LEADS, 3), dtype=np.float32)
                stft_b = np.zeros((B, N_LEADS, F_, T_), dtype=np.float32)
            beats_b[i] = bundle["beats"]
            rr_b[i] = bundle["rr_feats"]
            # Tolerate STFT shape variance from a cache_root that used a different
            # signal length (rare; pad / crop along T).
            sb = bundle["stft"]
            if sb.shape != stft_b[i].shape:
                Tt = min(sb.shape[2], stft_b[i].shape[2])
                stft_b[i, :, :, :Tt] = sb[:, : stft_b[i].shape[1], :Tt]
            else:
                stft_b[i] = sb
            n_valid_list.append(bundle["n_valid"])

        return beats_b, rr_b, stft_b, n_valid_list

    # ── forward ─────────────────────────────────────────────────────────────
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
        x: (B, 12, T) raw ECG at the encoder contract rate (input_size x model_fs).

        Three preprocessing paths (in priority order):
          1. cached_*       pre-loaded by DataLoader workers (parallel, fastest)
          2. ecg_filepath   encoder loads cache from disk (serial main-thread)
          3. live           full R-peak + beat + STFT pipeline (slowest)

        Returns: (sequence_features (B, max_beats*12, 512), pooled (B, 512))

        Note: pad_mask is None to match pretrain (zero-padded beats are attended).
        """
        x = torch.nan_to_num(x)
        device = x.device

        if cached_beats is not None:
            # Fast path: tensors already loaded (and pin-memory'd) by workers.
            # Just upcast fp16 -> fp32 on device, no disk I/O on the main thread.
            self._cache_hits += int(cached_beats.shape[0])
            beats = cached_beats.to(device, dtype=torch.float32, non_blocking=True)
            rr = cached_rr.to(device, dtype=torch.float32, non_blocking=True)
            stft = cached_stft.to(device, dtype=torch.float32, non_blocking=True)
            self._last_n_valid = None
        else:
            beats_np, rr_np, stft_np, _n_valid = self._preprocess_batch(
                x, ecg_filepath=ecg_filepath, ecg_seg_idx=ecg_seg_idx,
            )
            beats = torch.from_numpy(beats_np).to(device, non_blocking=True)
            rr = torch.from_numpy(rr_np).to(device, non_blocking=True)
            stft = torch.from_numpy(stft_np).to(device, non_blocking=True)
            # Live path: stash n_valid so the forward path's beat_valid_mask
            # construction can pick it up without changing the public signature.
            self._last_n_valid = _n_valid

        # Tokenizer encode (frozen, no_grad)
        B, N, L, W = beats.shape
        with torch.no_grad():
            self.tokenizer.eval()
            beats_flat = beats.view(B * N * L, 1, W)
            _zq, idx_flat = self.tokenizer.encode(beats_flat)
        indices = idx_flat.view(B, N, L).long()

        # ── Build beat_valid_mask for the MoRyECG family (train/test consistent) ──
        # During pretrain, MoRyECG / v6 receives a real beat_valid_mask derived
        # from the dataset's n_valid; padded slots are excluded from
        # rhythm-attention key_padding_mask AND zeroed at the end of each block.
        # For fine-tune we mirror that contract — otherwise padded beats inject
        # garbage activations through the GlobalRefineBlock's full attention
        # (most damaging in v6) or contaminate the lead-mean GLOB aggregation
        # (v5). Source priority:
        #   1) cached_n_valid       (provided by DataLoader fast path)
        #   2) self._last_n_valid   (set by _preprocess_batch live path above)
        #   3) None                 (all beats treated as valid — v4 legacy)
        beat_valid_mask = None
        if self.arch in ("moryecg", "moryecg_v6"):
            if cached_n_valid is not None:
                if isinstance(cached_n_valid, torch.Tensor):
                    n_valid_t = cached_n_valid.to(device, dtype=torch.long, non_blocking=True)
                else:
                    n_valid_t = torch.as_tensor(cached_n_valid, device=device, dtype=torch.long)
                arange_n = torch.arange(self.max_beats, device=device).unsqueeze(0)
                beat_valid_mask = arange_n < n_valid_t.unsqueeze(1)            # (B, N)
            elif getattr(self, "_last_n_valid", None) is not None:
                n_valid_t = torch.as_tensor(self._last_n_valid, device=device, dtype=torch.long)
                arange_n = torch.arange(self.max_beats, device=device).unsqueeze(0)
                beat_valid_mask = arange_n < n_valid_t.unsqueeze(1)
            # else: None (no info available — fall back to "all valid")

        # MoRyECG transformer.
        # For the MoRyECG family we use forward_flat to expose the
        # (B, 1+N*L, D) layout the benchmark contract expects: out[:, 0] is
        # the pooled global token g and out[:, 1:] are the (beat, lead) tokens.
        if self.arch == "moryecg_v6":
            out = self.model.forward_flat(indices, rr, stft,
                                          beat_valid_mask=beat_valid_mask)
        elif self.arch == "moryecg":
            out = self.model.forward_flat(indices, rr, stft,
                                          beat_valid_mask=beat_valid_mask)
        else:
            out = self.model(indices, rr, stft)
        seq_feat, pooled = pool_tokens(out, self.pool_mode)
        return seq_feat, pooled

    # ── layer-dependent LR groups (paper finetune contract) ─────────────────
    def get_layer_groups(self):
        early, late = [], []
        if self.arch in ("moryecg", "moryecg_v6"):
            # Embeddings + STFT encoder + glob_token + RR bias MLP go to early.
            early_modules = [self.model.morph_emb, self.model.lead_emb,
                             self.model.pos_emb, self.model.rhythm_mlp,
                             self.model.global_ctx]
            if self.model.rr_bias_mlp is not None:
                early_modules.append(self.model.rr_bias_mlp)
            for mod in early_modules:
                for p in mod.parameters():
                    early.append(p)
            early.append(self.model.glob_token)
            n_layers = len(self.model.blocks)
            split = n_layers // 2
            for i, blk in enumerate(self.model.blocks):
                grp = early if i < split else late
                for p in blk.parameters():
                    grp.append(p)
            # The v6 refine block belongs to "late" (final layer; benefits
            # most from a higher fine-tune LR).
            if self.arch == "moryecg_v6" and getattr(self.model, "refine", None) is not None:
                for p in self.model.refine.parameters():
                    late.append(p)
            for p in self.model.norm_h.parameters(): late.append(p)
            for p in self.model.norm_g.parameters(): late.append(p)
            return {"early": early, "late": late}

        # v4 flat-seq path
        n_layers = len(self.model.transformer.layers)
        for mod in (self.model.morph_emb, self.model.lead_emb,
                    self.model.pos_emb, self.model.rhythm_mlp,
                    self.model.global_ctx):
            for p in mod.parameters():
                early.append(p)
        early.append(self.model.cls_token)
        split = n_layers // 2
        for i, layer in enumerate(self.model.transformer.layers):
            grp = early if i < split else late
            for p in layer.parameters():
                grp.append(p)
        for p in self.model.norm.parameters():
            late.append(p)
        return {"early": early, "late": late}

    # ── debug helper ────────────────────────────────────────────────────────
    def cache_stats(self) -> str:
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return "no preprocessing yet"
        hit_pct = 100.0 * self._cache_hits / total
        return (f"cache hits={self._cache_hits} misses={self._cache_misses} "
                f"({hit_pct:.1f}%)")
