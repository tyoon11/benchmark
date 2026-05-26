"""
ECG Downstream Benchmark
=========================
H5-backed ECG downstream task benchmark.

Usage:
  # Single GPU
  python run.py --task ptbxl_super_jepa --eval_mode linear_probe \
      --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
      --encoder_ckpt weights/encoder.pt

  # Multi-GPU (e.g., 4 cards)
  torchrun --nproc_per_node=4 run.py --task ptbxl_super_jepa \
      --eval_mode finetune_linear \
      --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
      --encoder_ckpt weights/encoder.pt

  # select a specific GPU
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run.py ...

  # dummy encoder test
  python run.py --task ptbxl_super_jepa --eval_mode linear_probe --dummy
"""

import os
import sys
import argparse
import json
import logging
import yaml
import importlib
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np

# add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import H5ECGDataset, build_dataloaders
from src.dataset_numpy import EchoNextDataset
from src.wrapper import DownstreamWrapper
from src.trainer import DownstreamTrainer


# ═══════════════════════════════════════════════════════════════
# DDP utilities
# ═══════════════════════════════════════════════════════════════
def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """init DDP from torchrun-configured environment variables"""
    if "RANK" not in os.environ:
        return False
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return True


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════
# dummy encoder (for testing)
# ═══════════════════════════════════════════════════════════════
class DummyEncoder(torch.nn.Module):
    """dummy encoder for testing. GAP → (B, feature_dim)"""
    def __init__(self, n_leads=12, feature_dim=256):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(n_leads, 64, 7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, feature_dim, 7, padding=3),
            torch.nn.ReLU(),
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        feat = self.conv(x)
        seq_feat = feat.transpose(1, 2)
        pooled = feat.mean(dim=2)
        return seq_feat, pooled


# ═══════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════
def _expand_env_vars(value):
    """Recursively expand ${VAR} and $VAR references in str values inside cfg."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_config(task_name: str, overrides: dict = None) -> dict:
    cfg_dir = SCRIPT_DIR / "configs"

    # default ECG_DATA_ROOT to a placeholder when unset.
    # in another environment, `export ECG_DATA_ROOT=/your/data/root`  to override.
    os.environ.setdefault("ECG_DATA_ROOT", "/path/to/ecg_data")

    default_path = cfg_dir / "default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    task_path = cfg_dir / "tasks" / f"{task_name}.yaml"
    if task_path.exists():
        with open(task_path) as f:
            task_cfg = yaml.safe_load(f)
        for section in task_cfg:
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(task_cfg[section])
            else:
                cfg[section] = task_cfg[section]

    if overrides:
        for k, v in overrides.items():
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v

    # 1) ${VAR} env var expansion (e.g., ${ECG_DATA_ROOT}/h5/...).
    cfg = _expand_env_vars(cfg)

    # 2) relative path resolve — inside the repo labelfile etc. ('labels/x.csv' SCRIPT_DIR reference).
    data_section = cfg.get("data", {})
    for key in ("label_csv", "table_csv", "h5_root", "metadata_csv"):
        v = data_section.get(key)
        if isinstance(v, str) and not os.path.isabs(v):
            data_section[key] = str(SCRIPT_DIR / v)
    if "waveforms" in data_section and isinstance(data_section["waveforms"], dict):
        for split, p in data_section["waveforms"].items():
            if isinstance(p, str) and not os.path.isabs(p):
                data_section["waveforms"][split] = str(SCRIPT_DIR / p)

    return cfg


def load_encoder(encoder_cls: str, encoder_ckpt: str = None, **kwargs):
    """
    Load the encoder.
    checkpoint adapter's __init__(checkpoint=...)  as before.
    adapter  _load_checkpoint  then  use.
    """
    module_path, cls_name = encoder_cls.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    # checkpoint adapter generate in before
    if encoder_ckpt:
        kwargs["checkpoint"] = encoder_ckpt
    encoder = cls(**kwargs)

    if is_main_process() and encoder_ckpt:
        logging.info(f"Loaded encoder from {encoder_ckpt}")

    feature_dim = getattr(encoder, "feature_dim", None)
    if feature_dim is None:
        feature_dim = getattr(encoder, "embed_dim", None)
    if feature_dim is None:
        raise ValueError("Encoder must have 'feature_dim' or 'embed_dim' attribute")

    return encoder, feature_dim


# ═══════════════════════════════════════════════════════════════
# DataLoader (DDP-aware)
# ═══════════════════════════════════════════════════════════════
def build_dataloaders_ddp(data_cfg, split="train"):
    """
    DDP  DataLoader generate.

    loader_type:
      - 'h5' (default): H5ECGDataset (fold based split)
      - 'echonext_numpy': EchoNextDataset (.npy + metadata.csv, before definitionsed split use)
    """
    from torch.utils.data import DataLoader

    loader_type = data_cfg.get("loader_type", "h5")

    if loader_type == "echonext_numpy":
        # split_overrides: smoke test from train→val same mapping (for)
        md_split = data_cfg.get("split_overrides", {}).get(split, split)
        ds = EchoNextDataset(
            waveform_npy=data_cfg["waveforms"][split],
            metadata_csv=data_cfg["metadata_csv"],
            split=md_split,
            split_col=data_cfg.get("split_col", "split"),
            label_cols=data_cfg["label_cols"],
            source_fs=int(data_cfg.get("source_fs", 250)),
            target_fs=data_cfg.get("target_fs"),
            target_length=data_cfg.get("target_length"),
            chunk_length=data_cfg.get("chunk_length"),
            random_crop=(split == "train"),
            normalize=bool(data_cfg.get("normalize", False)),
            mean=data_cfg.get("mean"),
            std=data_cfg.get("std"),
            n_leads=int(data_cfg.get("n_leads", 12)),
            layout=str(data_cfg.get("layout", "NHWC")),
        )
    else:
        ds = H5ECGDataset(
            h5_root=data_cfg["h5_root"],
            table_csv=data_cfg["table_csv"],
            label_csv=data_cfg.get("label_csv"),
            label_cols=data_cfg.get("label_cols"),
            target_fs=data_cfg.get("target_fs"),
            target_length=data_cfg.get("target_length"),
            chunk_length=data_cfg.get("chunk_length"),
            random_crop=(split == "train"),
            seg_idx=data_cfg.get("seg_idx", None),
            normalize=data_cfg.get("normalize", False),
            fold_col=data_cfg.get("fold_col"),
            fold_ids=data_cfg.get(f"{split}_folds"),
            mean=data_cfg.get("mean"),
            std=data_cfg.get("std"),
            task_type=data_cfg.get("task_type", "binary"),
            target_mean=data_cfg.get("target_mean"),
            target_std=data_cfg.get("target_std"),
            cls_cols=data_cfg.get("cls_cols"),
            reg_cols=data_cfg.get("reg_cols"),
        )

    sampler = None
    shuffle = (split == "train")
    if is_distributed():
        sampler = DistributedSampler(ds, shuffle=shuffle)
        shuffle = False  # sampler shuffle 

    nw = int(os.environ.get("NUM_WORKERS", data_cfg.get("num_workers", 4)))
    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    return ds, loader


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ECG Downstream Benchmark")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--eval_mode", type=str, default="linear_probe",
                        choices=["linear_probe", "attention_probe",
                                 "finetune_linear", "finetune_attention"])
    parser.add_argument("--encoder_cls", type=str, default=None)
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--dummy", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--train_folds", type=str, default=None)
    parser.add_argument("--val_folds", type=str, default=None)
    parser.add_argument("--test_folds", type=str, default=None)

    args = parser.parse_args()

    # DDP init
    use_ddp = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    # Logging (rank 0 only)
    if is_main_process():
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Config
    overrides = {}
    if args.epochs:     overrides["train.epochs"] = args.epochs
    if args.lr:         overrides["train.lr"] = args.lr
    if args.batch_size: overrides["data.batch_size"] = args.batch_size
    if args.device:     overrides["train.device"] = args.device
    overrides["eval_mode"] = args.eval_mode

    cfg = load_config(args.task, overrides)
    task_cfg = cfg.get("task", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    head_cfg = cfg.get("head", {})

    num_classes = task_cfg.get("num_classes", 5)
    eval_mode = cfg.get("eval_mode", "linear_probe")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or str(
        SCRIPT_DIR / "results" / timestamp / f"{args.task}_{eval_mode}"
    )

    if is_main_process():
        logging.info(f"Task: {args.task} | Mode: {eval_mode} | Classes: {num_classes}")
        logging.info(f"DDP: {use_ddp} | World size: {world_size} | Rank: {rank}")

    # ── Encoder ──
    if args.dummy:
        encoder = DummyEncoder(n_leads=12, feature_dim=256)
        feature_dim = 256
    elif args.encoder_cls:
        encoder, feature_dim = load_encoder(args.encoder_cls, args.encoder_ckpt)
    else:
        parser.error("--encoder_cls or --dummy required")

    if is_main_process():
        logging.info(f"Encoder feature_dim={feature_dim}")

    # ── Multi-window extension (paper §3.3) ──
    # encoder chunk_seconds then dataset ECG 1 N chunk by .
    # → training data N + eval at ecg_id .
    chunk_seconds = getattr(encoder, "chunk_seconds", None)
    if chunk_seconds is not None and data_cfg.get("target_fs"):
        chunk_length = int(round(chunk_seconds * float(data_cfg["target_fs"])))
        if chunk_length < int(data_cfg.get("target_length", 0)):
            data_cfg["chunk_length"] = chunk_length
            if is_main_process():
                logging.info(
                    f"Multi-window enabled: chunk_seconds={chunk_seconds} "
                    f"→ chunk_length={chunk_length} samples "
                    f"(target_length={data_cfg['target_length']})"
                )

    # ── Model Wrapper ──
    model = DownstreamWrapper(
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=num_classes,
        eval_mode=eval_mode,
        head_kwargs=head_cfg,
    )

    # Device config
    if use_ddp:
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        # DDP wrapper from original module 
        model_unwrapped = model.module
    else:
        device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)
        model_unwrapped = model

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # ── Data: Fold Split ──
    fold_cfg = cfg.get("fold", {})
    fold_col = fold_cfg.get("col", "strat_fold")
    auto_split = fold_cfg.get("auto_split", True)

    # CLI override  auto_split if so, table CSV from strat_fold automatic 
    if args.train_folds:
        data_cfg["fold_col"] = fold_col
        data_cfg["train_folds"] = [int(x) for x in args.train_folds.split(",")]
    if args.val_folds:
        data_cfg["fold_col"] = fold_col
        data_cfg["val_folds"] = [int(x) for x in args.val_folds.split(",")]
    if args.test_folds:
        data_cfg["fold_col"] = fold_col
        data_cfg["test_folds"] = [int(x) for x in args.test_folds.split(",")]

    if auto_split and not args.train_folds and not args.val_folds:
        # table CSV → label CSV order as strat_fold search
        # paper's split: strat_fold ∈ [0,18) train / 18 val / 19 test (18/1/1)
        table_path = data_cfg.get("table_csv", "")
        label_path = data_cfg.get("label_csv", "")

        fold_source = None
        _df_full = None
        for path in (table_path, label_path):
            if path and os.path.exists(path):
                _df = pd.read_csv(path, usecols=lambda c: c == fold_col, nrows=1)
                if fold_col in _df.columns:
                    _df_full = pd.read_csv(path, usecols=[fold_col])
                    fold_source = path
                    break

        if _df_full is not None:
            max_fold = int(_df_full[fold_col].max())
            data_cfg["fold_col"] = fold_col
            data_cfg["train_folds"] = list(range(0, max_fold - 1))
            data_cfg["val_folds"] = [max_fold - 1]
            data_cfg["test_folds"] = [max_fold]
            if is_main_process():
                train_n = len(_df_full[_df_full[fold_col] < max_fold - 1])
                val_n = len(_df_full[_df_full[fold_col] == max_fold - 1])
                test_n = len(_df_full[_df_full[fold_col] == max_fold])
                src = "table" if fold_source == table_path else "label"
                logging.info(f"Auto fold split [{src}]: train({train_n:,}) / val({val_n:,}) / test({test_n:,})")
        elif is_main_process():
            logging.warning(f"⚠ {fold_col} column table/label CSV  none — split inside  ")

    # task_type before (binary / multi-label-binary / regression / classification_and_regression)
    task_type = task_cfg.get("task_type", "binary")
    data_cfg["task_type"] = task_type

    # ── Joint task: pull cls_cols / reg_cols / report_groups from schema JSON ──
    # (paper main_lite_ecg.py: classification_and_regression with 158+6+1 cls + 35 reg)
    if task_type == "classification_and_regression":
        schema_path = data_cfg.get("schema_json")
        if not schema_path:
            # default: replace .csv with .json next to label_csv
            label_csv = data_cfg.get("label_csv", "")
            schema_path = str(Path(label_csv).with_suffix(".json")) if label_csv else None
        if schema_path and os.path.exists(schema_path):
            with open(schema_path) as fh:
                schema = json.load(fh)
            if "cls_cols" not in data_cfg:
                data_cfg["cls_cols"] = schema.get("cls_cols", [])
            if "reg_cols" not in data_cfg:
                data_cfg["reg_cols"] = schema.get("reg_cols", [])
            if "report_groups" not in data_cfg:
                data_cfg["report_groups"] = schema.get("report_groups", {})
        cls_cols = data_cfg.get("cls_cols") or []
        reg_cols = data_cfg.get("reg_cols") or []
        num_cls = len(cls_cols)
        num_reg = len(reg_cols)
        # head output dim = cls + reg
        num_classes = num_cls + num_reg
        task_cfg["num_cls"] = num_cls
        task_cfg["num_reg"] = num_reg
        if is_main_process():
            logging.info(
                f"  Joint MIMIC task: num_cls={num_cls}, num_reg={num_reg}, "
                f"total head dim={num_classes}")

    # ── Regression target z-normalization (paper-faithful: train fold stats) ──
    # IMPORTANT: dataloader generate before in target_mean/std data_cfg in
    # joint task: z-norm only the reg subset (cls part stays 0/1/NaN)
    znorm_cols = None
    if task_type == "regression":
        znorm_cols = data_cfg.get("label_cols")
    elif task_type == "classification_and_regression":
        znorm_cols = data_cfg.get("reg_cols")
    if znorm_cols and data_cfg.get("label_csv") and data_cfg.get("train_folds"):
        try:
            label_df_full = pd.read_csv(data_cfg["label_csv"], low_memory=False)
            fold_col = data_cfg.get("fold_col", "strat_fold")
            train_rows = label_df_full[label_df_full[fold_col].isin(data_cfg["train_folds"])]
            if all(c in train_rows.columns for c in znorm_cols):
                # ddof=0 to match sklearn.StandardScaler (paper mimic_preprocessing.py:431)
                t_mean = train_rows[znorm_cols].mean(axis=0).values.astype("float32")
                t_std = train_rows[znorm_cols].std(axis=0, ddof=0).values.astype("float32")
                data_cfg["target_mean"] = t_mean.tolist()
                data_cfg["target_std"] = t_std.tolist()
                if is_main_process():
                    logging.info(f"  Regression z-norm ({len(znorm_cols)} cols, train fold, ddof=0):")
                    logging.info(f"    mean[:5]={t_mean[:5].tolist()}")
                    logging.info(f"    std[:5]={t_std[:5].tolist()}")
        except Exception as e:
            if is_main_process():
                logging.warning(f"  z-norm compute failure (as-is rows): {e}")

    train_ds, train_loader = build_dataloaders_ddp(data_cfg, "train")
    val_ds, val_loader = build_dataloaders_ddp(data_cfg, "val")

    test_loader = None
    has_test = (
        data_cfg.get("test_folds")
        or (data_cfg.get("loader_type") == "echonext_numpy"
            and "test" in data_cfg.get("waveforms", {}))
    )
    if has_test:
        _, test_loader = build_dataloaders_ddp(data_cfg, "test")

    if is_main_process():
        logging.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}"
                     + (f" | Test: {len(test_loader.dataset):,}" if test_loader else ""))

    # ── Train ──
    if task_type == "classification_and_regression":
        label_names = list(data_cfg.get("cls_cols", [])) + list(data_cfg.get("reg_cols", []))
    else:
        label_names = data_cfg.get("label_cols")
    trainer_cfg = {
        **train_cfg,
        "save_dir": save_dir,
        "label_names": label_names,
        "device": str(device),
        "use_ddp": use_ddp,
        "rank": rank,
        "world_size": world_size,
        "task_type": task_type,
        "num_cls": task_cfg.get("num_cls", 0),
        "num_reg": task_cfg.get("num_reg", 0),
        "cls_cols": data_cfg.get("cls_cols"),
        "reg_cols": data_cfg.get("reg_cols"),
        "report_groups": data_cfg.get("report_groups"),
    }
    trainer = DownstreamTrainer(model, train_loader, val_loader, test_loader, trainer_cfg)
    results = trainer.train()

    # ── merge CSV in append (rank 0 only) ──
    if is_main_process():
        _append_result_csv(
            args=args,
            task=args.task,
            eval_mode=eval_mode,
            encoder_cls=args.encoder_cls or "dummy",
            num_classes=num_classes,
            save_dir=save_dir,
            train_size=len(train_ds),
            val_size=len(val_ds),
            test_size=len(test_loader.dataset) if test_loader else 0,
            results=results,
            task_type=task_type,
        )
        logging.info(f"Results saved to: {save_dir}")

    cleanup_distributed()
    return results


def _append_result_csv(args, task, eval_mode, encoder_cls, num_classes,
                      save_dir, train_size, val_size, test_size, results,
                      task_type="binary"):
    """
    each  results results_all.csv in row by add (thread-safe append).
    save_dir above directory(example: results/{timestamp}/) save.

    task_type per metric :
      - binary / multi-label-binary: AUROC / AUPRC / F1
      - regression: MAE / MSE / RMSE / R² / neg_MAE
    """
    from pathlib import Path
    import csv, fcntl

    save_path = Path(save_dir)
    parent = save_path.parent
    csv_path = parent / "results_all.csv"

    # model name (encoder_cls →  class)
    model_name = encoder_cls.rsplit(".", 1)[-1] if "." in encoder_cls else encoder_cls
    model_name = model_name.replace("Encoder", "").lower()

    # test_metrics.txt read
    test_m = _read_metrics(save_path / "test_metrics.txt")
    val_m = _read_metrics(save_path / "val_metrics.txt")

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "task": task,
        "task_type": task_type,
        "eval_mode": eval_mode,
        "num_classes": num_classes,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "best_val": results.get("best_val", results.get("best_val_auroc", float("nan"))),
        "best_epoch": results.get("best_epoch", -1),
        # Classification metrics (binary / multi-label-binary)
        "test_auroc_macro": test_m.get("auroc_macro", float("nan")),
        "test_auroc_micro": test_m.get("auroc_micro", float("nan")),
        "test_auprc_macro": test_m.get("auprc_macro", float("nan")),
        "test_f1_macro":    test_m.get("f1_macro", float("nan")),
        "val_auroc_macro": val_m.get("auroc_macro", float("nan")),
        # Regression metrics
        "test_mae_macro":   test_m.get("mae_macro", float("nan")),
        "test_mse_macro":   test_m.get("mse_macro", float("nan")),
        "test_rmse_macro":  test_m.get("rmse_macro", float("nan")),
        "test_r2_macro":    test_m.get("r2_macro", float("nan")),
        "val_neg_mae_macro": val_m.get("neg_mae_macro", float("nan")),
        # Joint task (classification_and_regression) — paper composite score
        "test_composite_score": test_m.get("composite_score", float("nan")),
        "test_auroc_macro_cls": test_m.get("auroc_macro_cls", float("nan")),
        "test_mae_global_reg":  test_m.get("mae_global_reg", float("nan")),
        "save_dir": str(save_dir),
    }
    # per-sub-task metrics for joint tasks — derived dynamically from test_metrics.txt
    if task_type == "classification_and_regression":
        for key, val in test_m.items():
            if key.endswith("_auroc_macro") and key != "auroc_macro" and key != "auroc_macro_cls":
                row[f"test_{key}"] = val
            elif key.endswith("_mae_macro") or key.endswith("_mae_global"):
                row[f"test_{key}"] = val

    # file lock + append (  concurrent run )
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_metrics(path):
    """metrics txt file from key: value parsing"""
    from pathlib import Path
    if not Path(path).exists():
        return {}
    result = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    result[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return result


if __name__ == "__main__":
    main()
