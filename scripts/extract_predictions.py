"""
Test-set predictions/targets extract (bootstrap preprocessing)
====================================================
 (model, task, mode) result directory in for best.pt loadand
test loader by inference  preds.npy / targets.npy / ids.npy save.

Usage:
  python scripts/extract_predictions.py --result_dir <RESULT_DIR>/<model>_<task>_<mode>
  python scripts/extract_predictions.py --root <RESULT_DIR>           # all directory automatic extract
  python scripts/extract_predictions.py --root <RESULT_DIR> --filter cpc_chapman  # prefix matching

:
  ECG_DATA_ROOT   default ${ECG_DATA_ROOT}
  ECG_CKPT_ROOT   default ${ECG_CKPT_ROOT}

output:
  <result_dir>/preds.npy        (N, C) float32 — sigmoid  () / raw (time)
  <result_dir>/targets.npy      (N, C) float32
  <result_dir>/ids.npy          (N,)   key — multi-window if so, ecg_id,  then row index
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import H5ECGDataset
from src.dataset_numpy import EchoNextDataset
from src.wrapper import DownstreamWrapper

# run.py re-use
from run import load_config, load_encoder

# ──────────────────────────────────────────────────────────────────
# Model registry (configs/models.sh of bash assoc array mirror)
# ──────────────────────────────────────────────────────────────────
ECG_CKPT_ROOT = os.environ.get("ECG_CKPT_ROOT", "/path/to/checkpoints")

MORYECG_REPO = os.environ.get("MORYECG_REPO", str(Path(__file__).resolve().parents[2]))

MODEL_CLS_MAP = {
    "ecg_founder":    "src.encoders.ecg_founder.ECGFounderEncoder",
    "ecg_jepa":       "src.encoders.ecg_jepa.ECGJEPAEncoder",
    "st_mem":         "src.encoders.st_mem.StMemEncoder",
    "merl":           "src.encoders.merl.MerlResNetEncoder",
    "ecgfm_ked":      "src.encoders.ecgfm_ked.EcgFmKEDEncoder",
    "hubert_ecg":     "src.encoders.hubert_ecg.HuBERTECGEncoder",
    "ecg_fm":         "src.encoders.ecg_fm.ECGFMEncoder",
    "cpc":            "src.encoders.cpc.CPCEncoder",
    "moryecg_cb1024": "src.encoders.moryecg.MoRyECGEncoder",
    "moryecg_a5":     "src.encoders.moryecg_a5.MoRyECGA5Encoder",
}
MODEL_CKPT_MAP = {
    "ecg_founder":    f"{ECG_CKPT_ROOT}/ecg_founder/12_lead_ECGFounder.pth",
    "ecg_jepa":       f"{ECG_CKPT_ROOT}/ecg_jepa/multiblock_epoch100.pth",
    "st_mem":         f"{ECG_CKPT_ROOT}/st_mem/st_mem_vit_base_full.pth",
    "merl":           f"{ECG_CKPT_ROOT}/merl/res18_best_encoder.pth",
    "ecgfm_ked":      f"{ECG_CKPT_ROOT}/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt",
    "hubert_ecg":     f"{ECG_CKPT_ROOT}/hubert_ecg/hubert_ecg_base.safetensors",
    "ecg_fm":         f"{ECG_CKPT_ROOT}/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt",
    "cpc":            f"{ECG_CKPT_ROOT}/cpc/last_11597276.ckpt",
    "moryecg_cb1024": f"{MORYECG_REPO}/checkpoints/pretrain_heedb_cb1024_v4/best.pt",
    # NOTE: the benchmark heads were trained with run1's encoder (epoch 5,
    # val_acc_nontop=0.474). A new run2 pretrain has since overwritten the main
    # best.pt, so for feature-consistent extraction we MUST point at the archived
    # run1 checkpoint. Override with MORYECG_A5_CKPT if you re-benchmark on run2.
    "moryecg_a5":     os.environ.get(
        "MORYECG_A5_CKPT",
        f"{MORYECG_REPO}/checkpoints/pretrain_axial_s4_a5_heedb_full_cb1024/_run1_archive_20260608/best.pt",
    ),
}

EVAL_MODES = ["attention_probe", "finetune_attention", "finetune_linear", "linear_probe"]
# sort:  model name  (ecgfm_ked ecg_* first, matching lengthorder)
MODEL_NAMES_BY_LEN = sorted(MODEL_CLS_MAP.keys(), key=len, reverse=True)

logger = logging.getLogger("extract_preds")


# ──────────────────────────────────────────────────────────────────
# directory name parsing: <model>_<task>_<mode>  →  (model, task, mode)
# ──────────────────────────────────────────────────────────────────
def parse_dirname(dirname: str):
    for mode in EVAL_MODES:
        suffix = "_" + mode
        if dirname.endswith(suffix):
            rest = dirname[: -len(suffix)]
            for model in MODEL_NAMES_BY_LEN:
                pref = model + "_"
                if rest.startswith(pref):
                    task = rest[len(pref):]
                    return model, task, mode
            return None
    return None


# ──────────────────────────────────────────────────────────────────
# Test dataloader  (run.py of  mirror)
# ──────────────────────────────────────────────────────────────────
def build_test_loader(cfg):
    data_cfg = cfg.get("data", {})
    task_cfg = cfg.get("task", {})
    fold_cfg = cfg.get("fold", {})
    fold_col = fold_cfg.get("col", "strat_fold")
    auto_split = fold_cfg.get("auto_split", True)

    # auto_split: table CSV from strat_fold automatic  test=max_fold
    if auto_split:
        for path in (data_cfg.get("table_csv", ""), data_cfg.get("label_csv", "")):
            if path and os.path.exists(path):
                _df = pd.read_csv(path, usecols=lambda c: c == fold_col, nrows=1)
                if fold_col in _df.columns:
                    _df_full = pd.read_csv(path, usecols=[fold_col])
                    max_fold = int(_df_full[fold_col].max())
                    data_cfg["fold_col"] = fold_col
                    data_cfg["train_folds"] = list(range(0, max_fold - 1))
                    data_cfg["val_folds"] = [max_fold - 1]
                    data_cfg["test_folds"] = [max_fold]
                    break

    # joint task: load cls_cols / reg_cols from schema JSON (mirrors run.py logic)
    task_type = task_cfg.get("task_type", "binary")
    data_cfg["task_type"] = task_type
    if task_type == "classification_and_regression":
        import json
        schema_path = data_cfg.get("schema_json")
        if not schema_path:
            label_csv = data_cfg.get("label_csv", "")
            schema_path = str(Path(label_csv).with_suffix(".json")) if label_csv else None
        if schema_path and os.path.exists(schema_path):
            with open(schema_path) as fh:
                schema = json.load(fh)
            if "cls_cols" not in data_cfg:
                data_cfg["cls_cols"] = schema.get("cls_cols", [])
            if "reg_cols" not in data_cfg:
                data_cfg["reg_cols"] = schema.get("reg_cols", [])

    # regression z-norm (train fold mean/std)
    if task_type == "regression" and data_cfg.get("label_csv") and data_cfg.get("train_folds"):
        label_df = pd.read_csv(data_cfg["label_csv"], low_memory=False)
        train_rows = label_df[label_df[fold_col].isin(data_cfg["train_folds"])]
        cols = data_cfg.get("label_cols")
        if cols and all(c in train_rows.columns for c in cols):
            data_cfg["target_mean"] = train_rows[cols].mean(axis=0).values.astype("float32").tolist()
            data_cfg["target_std"]  = train_rows[cols].std(axis=0).values.astype("float32").tolist()

    loader_type = data_cfg.get("loader_type", "h5")
    if loader_type == "echonext_numpy":
        if "test" not in data_cfg.get("waveforms", {}):
            return None, None, None
        ds = EchoNextDataset(
            waveform_npy=data_cfg["waveforms"]["test"],
            metadata_csv=data_cfg["metadata_csv"],
            split="test",
            split_col=data_cfg.get("split_col", "split"),
            label_cols=data_cfg["label_cols"],
            source_fs=int(data_cfg.get("source_fs", 250)),
            target_fs=data_cfg.get("target_fs"),
            target_length=data_cfg.get("target_length"),
            chunk_length=data_cfg.get("chunk_length"),
            random_crop=False,
            normalize=bool(data_cfg.get("normalize", False)),
            mean=data_cfg.get("mean"),
            std=data_cfg.get("std"),
            n_leads=int(data_cfg.get("n_leads", 12)),
            layout=str(data_cfg.get("layout", "NHWC")),
        )
    else:
        if not data_cfg.get("test_folds"):
            return None, None, None
        ds = H5ECGDataset(
            h5_root=data_cfg["h5_root"],
            table_csv=data_cfg["table_csv"],
            label_csv=data_cfg.get("label_csv"),
            label_cols=data_cfg.get("label_cols"),
            target_fs=data_cfg.get("target_fs"),
            target_length=data_cfg.get("target_length"),
            chunk_length=data_cfg.get("chunk_length"),
            random_crop=False,
            seg_idx=data_cfg.get("seg_idx", None),
            normalize=data_cfg.get("normalize", False),
            fold_col=data_cfg.get("fold_col"),
            fold_ids=data_cfg.get("test_folds"),
            mean=data_cfg.get("mean"),
            std=data_cfg.get("std"),
            task_type=task_type,
            cls_cols=data_cfg.get("cls_cols"),
            reg_cols=data_cfg.get("reg_cols"),
            target_mean=data_cfg.get("target_mean"),
            target_std=data_cfg.get("target_std"),
        )

    nw = int(os.environ.get("NUM_WORKERS", data_cfg.get("num_workers", 4)))
    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    return ds, loader, task_type


# ──────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device, task_type):
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    for batch in loader:
        signal = batch["signal"].to(device)
        label = batch["label"]
        # Forward MoRyECG preprocessing-cache keys when the dataset provides them
        # so the encoder loads cached beats/rr/stft instead of recomputing R-peaks
        # live (orders of magnitude faster, esp. for large MIMIC test sets).
        # DownstreamWrapper filters these out for encoders that don't accept them.
        enc_kwargs = {}
        if "ecg_filepath" in batch:
            enc_kwargs["ecg_filepath"] = batch["ecg_filepath"]
        if "ecg_seg_idx" in batch:
            enc_kwargs["ecg_seg_idx"] = batch["ecg_seg_idx"]
        logits = model(signal, **enc_kwargs)
        if task_type == "regression":
            preds = logits.cpu().numpy()
        else:
            preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(label.numpy())
        if "ecg_id" in batch:
            ids = batch["ecg_id"]
            ids = ids.numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
            all_ids.append(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_ids = np.concatenate(all_ids, axis=0) if all_ids else None

    # multi-window aggregation: ecg_id per  (paper §3.3)
    if all_ids is not None and len(all_ids) != len(np.unique(all_ids)):
        unique_ids = np.unique(all_ids)
        agg_preds = np.empty((len(unique_ids), all_preds.shape[1]), dtype=all_preds.dtype)
        agg_targets = np.empty((len(unique_ids), all_targets.shape[1]), dtype=all_targets.dtype)
        for i, uid in enumerate(unique_ids):
            mask = (all_ids == uid)
            agg_preds[i] = all_preds[mask].mean(axis=0)
            agg_targets[i] = all_targets[mask][0]
        all_preds, all_targets, all_ids = agg_preds, agg_targets, unique_ids

    if all_ids is None:
        all_ids = np.arange(len(all_preds), dtype=np.int64)
    return all_preds, all_targets, all_ids


# ──────────────────────────────────────────────────────────────────
# result_dir handling
# ──────────────────────────────────────────────────────────────────
def process_result_dir(result_dir: Path, device: str = None, force: bool = False):
    parsed = parse_dirname(result_dir.name)
    if parsed is None:
        logger.warning(f"[SKIP] cannot parse dirname: {result_dir.name}")
        return False
    model_name, task, mode = parsed

    ckpt_path = result_dir / "best.pt"
    if not ckpt_path.exists():
        logger.warning(f"[SKIP] no best.pt: {result_dir}")
        return False

    out_preds = result_dir / "preds.npy"
    if out_preds.exists() and not force:
        logger.info(f"[SKIP-EXIST] {result_dir.name}")
        return True

    encoder_cls  = MODEL_CLS_MAP.get(model_name)
    encoder_ckpt = MODEL_CKPT_MAP.get(model_name)
    if encoder_cls is None:
        logger.warning(f"[SKIP] unknown model {model_name}")
        return False

    cfg = load_config(task, overrides={"eval_mode": mode})
    task_cfg = cfg.get("task", {})
    head_cfg = cfg.get("head", {})
    num_classes = task_cfg.get("num_classes", 5)

    encoder, feature_dim = load_encoder(encoder_cls, encoder_ckpt)

    # multi-window extension (encoder chunk_seconds then)
    data_cfg = cfg.get("data", {})
    chunk_seconds = getattr(encoder, "chunk_seconds", None)
    if chunk_seconds is not None and data_cfg.get("target_fs"):
        chunk_length = int(round(chunk_seconds * float(data_cfg["target_fs"])))
        if chunk_length < int(data_cfg.get("target_length", 0)):
            data_cfg["chunk_length"] = chunk_length

    model = DownstreamWrapper(
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=num_classes,
        eval_mode=mode,
        head_kwargs=head_cfg,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    logger.warning(f"  missing keys: {len(missing)}")
    if unexpected: logger.warning(f"  unexpected keys: {len(unexpected)}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)

    out = build_test_loader(cfg)
    if out is None or out[1] is None:
        logger.warning(f"[SKIP] no test loader for task={task}")
        return False
    ds, loader, task_type = out
    logger.info(f"  test_size={len(ds):,} | task_type={task_type} | chunk={data_cfg.get('chunk_length')}")

    t0 = time.time()
    preds, targets, ids = run_inference(model, loader, dev, task_type)
    logger.info(f"  inference done in {time.time()-t0:.1f}s — N={len(preds)}")

    np.save(out_preds, preds.astype(np.float32))
    np.save(result_dir / "targets.npy", targets.astype(np.float32))
    np.save(result_dir / "ids.npy", ids)
    # metadata (bootstrap stage from task_type/label_cols required)
    meta = {
        "model": model_name, "task": task, "mode": mode,
        "task_type": task_type,
        "num_classes": int(num_classes),
        "label_cols": data_cfg.get("label_cols"),
        "n_test": int(len(preds)),
    }
    import json
    (result_dir / "preds_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"  saved → {result_dir}/preds.npy")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", type=str, default=None,
                   help="single experiment directory (example: results/.../cpc_chapman_attention_probe)")
    p.add_argument("--root", type=str, default=None,
                   help="all timestamp directory (example: results/20260428_203028)")
    p.add_argument("--filter", type=str, default=None,
                   help="root  inside directory name substring filter")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.result_dir:
        process_result_dir(Path(args.result_dir), args.device, args.force)
        return

    if not args.root:
        p.error("--result_dir or --root  of one required")

    root = Path(args.root)
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.filter:
        dirs = [d for d in dirs if args.filter in d.name]
    logger.info(f"Total dirs: {len(dirs)}")
    n_ok = n_skip = 0
    for d in dirs:
        try:
            ok = process_result_dir(d, args.device, args.force)
            n_ok += int(ok); n_skip += int(not ok)
        except Exception as e:
            logger.error(f"[FAIL] {d.name}: {e}", exc_info=True)
            n_skip += 1
    logger.info(f"Done: ok={n_ok} skip/fail={n_skip}")


if __name__ == "__main__":
    main()
