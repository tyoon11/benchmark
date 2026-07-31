"""
ECG Downstream Benchmark
========================
H5-backed ECG downstream task benchmark, aligned with the original
``ecg-fm-benchmarking`` (``code/main_lite.py`` + ``run.sh``).

The window fed to a model is determined by the **encoder contract**, not by the
task config: each adapter declares ``input_size`` (seconds), ``model_fs`` (Hz)
and ``lead_order``, mirroring ``--input-size`` / ``--fs-model`` in the original
``run.sh``. The dataset crops at the dataset's native rate, permutes leads into
the order the encoder was pretrained on, then band-limit resamples to
``model_fs``.

Usage:
  # Single GPU
  python run.py --task ptbxl_super --eval_mode linear_probe \
      --encoder_cls src.encoders.ecg_founder.ECGFounderEncoder \
      --encoder_ckpt weights/encoder.pt

  # Multi-GPU (note: DDP multiplies the effective batch size; the original ran
  # single-GPU with batch_size=64)
  torchrun --nproc_per_node=4 run.py --task ptbxl_super --eval_mode finetune_linear ...

  # dummy encoder test
  python run.py --task ptbxl_super --eval_mode linear_probe --dummy
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import build_dataset
from src.dataset_numpy import build_echonext_dataset
from src.trainer import DownstreamTrainer
from src.wrapper import ORIGINAL_EVAL_MODE, DownstreamWrapper


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
    torch.cuda.set_device(dist.get_rank())
    return True


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════
# dummy encoder (for testing)
# ═══════════════════════════════════════════════════════════════
class DummyEncoder(torch.nn.Module):
    """Dummy encoder for smoke tests. GAP → (B, feature_dim)."""

    input_size = 2.5
    model_fs = 500
    model_seq_len = 1250
    lead_order = "standard"

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
        return feat.transpose(1, 2), feat.mean(dim=2)


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
    import yaml

    cfg_dir = SCRIPT_DIR / "configs"
    os.environ.setdefault("ECG_DATA_ROOT", "/path/to/ecg_data")

    with open(cfg_dir / "default.yaml") as f:
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

    cfg = _expand_env_vars(cfg)

    # resolve repo-relative paths (e.g. 'labels/x.csv')
    data_section = cfg.get("data", {})
    for key in ("label_csv", "table_csv", "h5_root", "metadata_csv"):
        v = data_section.get(key)
        if isinstance(v, str) and not os.path.isabs(v):
            data_section[key] = str(SCRIPT_DIR / v)
    if isinstance(data_section.get("waveforms"), dict):
        for split, p in data_section["waveforms"].items():
            if isinstance(p, str) and not os.path.isabs(p):
                data_section["waveforms"][split] = str(SCRIPT_DIR / p)

    return cfg


def load_encoder(encoder_cls: str, encoder_ckpt: str = None, **kwargs):
    """Import and instantiate an encoder adapter, returning (encoder, feature_dim)."""
    module_path, cls_name = encoder_cls.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

    if encoder_ckpt:
        kwargs["checkpoint"] = encoder_ckpt
    encoder = cls(**kwargs)

    if is_main_process() and encoder_ckpt:
        logging.info(f"Loaded encoder from {encoder_ckpt}")

    feature_dim = getattr(encoder, "feature_dim", None) or getattr(encoder, "embed_dim", None)
    if feature_dim is None:
        raise ValueError("Encoder must expose a 'feature_dim' or 'embed_dim' attribute")
    return encoder, feature_dim


def encoder_contract(encoder):
    """Read (input_size seconds, model_fs Hz, lead_order) off an adapter.

    ``chunk_seconds`` is accepted as a legacy alias for ``input_size``.
    """
    input_size = getattr(encoder, "input_size", None)
    if input_size is None:
        input_size = getattr(encoder, "chunk_seconds", None)
    model_fs = getattr(encoder, "model_fs", None)
    if input_size is None or model_fs is None:
        raise ValueError(
            f"{type(encoder).__name__} must declare 'input_size' (seconds) and "
            f"'model_fs' (Hz) — these correspond to --input-size / --fs-model in the "
            f"original run.sh. See src/encoders/_contract.py.")
    return float(input_size), float(model_fs), getattr(encoder, "lead_order", "standard")


# ═══════════════════════════════════════════════════════════════
# DataLoader (DDP-aware)
# ═══════════════════════════════════════════════════════════════
def build_dataloaders_ddp(data_cfg, split="train"):
    """Build the dataset for ``split`` plus a DDP-aware DataLoader."""
    from torch.utils.data import DataLoader

    if data_cfg.get("loader_type") == "echonext_numpy":
        ds = build_echonext_dataset(data_cfg, split)
    else:
        ds = build_dataset(data_cfg, split)

    sampler = None
    shuffle = (split == "train")
    if is_distributed():
        sampler = DistributedSampler(ds, shuffle=shuffle)
        shuffle = False

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

    parser.add_argument("--precision", type=str, default=None,
                        choices=["32", "16-mixed", "bf16-mixed"],
                        help="original default: 16-mixed (32 for the S4/CPC models)")
    parser.add_argument("--bootstrap_iterations", type=int, default=None,
                        help="empirical bootstrap iterations on the test split (0 disables)")
    parser.add_argument("--export_predictions", action="store_true",
                        help="write test predictions as .npz (original --export-predictions)")
    parser.add_argument("--no_paper_frozen", action="store_true",
                        help="hold frozen encoders in eval mode instead of reproducing the "
                             "original's train-mode behaviour (BN stats / dropout)")

    parser.add_argument("--train_folds", type=str, default=None)
    parser.add_argument("--val_folds", type=str, default=None)
    parser.add_argument("--test_folds", type=str, default=None)

    args = parser.parse_args()

    use_ddp = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s")

    overrides = {}
    if args.epochs:     overrides["train.epochs"] = args.epochs
    if args.lr:         overrides["train.lr"] = args.lr
    if args.batch_size: overrides["data.batch_size"] = args.batch_size
    if args.device:     overrides["train.device"] = args.device
    if args.precision:  overrides["train.precision"] = args.precision
    if args.bootstrap_iterations is not None:
        overrides["train.bootstrap_iterations"] = args.bootstrap_iterations
    if args.export_predictions:
        overrides["train.export_predictions"] = True
    overrides["eval_mode"] = args.eval_mode

    cfg = load_config(args.task, overrides)
    task_cfg = cfg.get("task", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    head_cfg = cfg.get("head", {})

    num_classes = task_cfg.get("num_classes", 5)
    eval_mode = cfg.get("eval_mode", "linear_probe")

    if task_cfg.get("task_type", "binary") != "classification_and_regression":
        num_classes = _check_num_classes(task_cfg, data_cfg, num_classes)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or str(
        SCRIPT_DIR / "results" / timestamp / f"{args.task}_{eval_mode}")

    if is_main_process():
        logging.info(f"Task: {args.task} | Mode: {eval_mode} "
                     f"(original --eval-mode {ORIGINAL_EVAL_MODE[eval_mode]}) | "
                     f"Classes: {num_classes}")
        logging.info(f"DDP: {use_ddp} | World size: {world_size} | Rank: {rank}")
        if use_ddp and world_size > 1:
            logging.warning(
                "DDP multiplies the effective batch size to %d; the original "
                "benchmark trained single-GPU at batch_size=%d.",
                world_size * int(data_cfg.get("batch_size", 64)),
                int(data_cfg.get("batch_size", 64)))

    # ── Encoder ──
    if args.dummy:
        encoder, feature_dim = DummyEncoder(n_leads=12, feature_dim=256), 256
    elif args.encoder_cls:
        encoder, feature_dim = load_encoder(args.encoder_cls, args.encoder_ckpt)
    else:
        parser.error("--encoder_cls or --dummy required")

    # ── Encoder contract -> data config (replaces --input-size / --fs-model) ──
    input_size, fs_model, lead_order = encoder_contract(encoder)
    data_cfg["input_size"] = input_size
    data_cfg["fs_model"] = fs_model
    data_cfg.setdefault("lead_order", lead_order)
    if is_main_process():
        logging.info(
            f"Encoder: feature_dim={feature_dim} | window={input_size}s @ {fs_model}Hz "
            f"= {int(round(input_size * fs_model))} samples | lead_order={data_cfg['lead_order']}")

    # ── Model Wrapper ──
    model = DownstreamWrapper(
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=num_classes,
        eval_mode=eval_mode,
        head_kwargs=head_cfg,
        paper_faithful_frozen=not args.no_paper_frozen,
    )

    if use_ddp:
        device = torch.device(f"cuda:{rank}")
        model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    else:
        device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # ── Fold split (train = fold < max-1, val = max-1, test = max) ──
    fold_cfg = cfg.get("fold", {})
    fold_col = fold_cfg.get("col", "strat_fold")
    auto_split = fold_cfg.get("auto_split", True)

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
        table_path = data_cfg.get("table_csv", "")
        label_path = data_cfg.get("label_csv", "")

        fold_source, _df_full = None, None
        for path in (table_path, label_path):
            if path and os.path.exists(path):
                probe = pd.read_csv(path, usecols=lambda c: c == fold_col, nrows=1)
                if fold_col in probe.columns:
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
                logging.info(f"Auto fold split [{src}]: train({train_n:,}) / "
                             f"val({val_n:,}) / test({test_n:,})")
        elif is_main_process():
            logging.warning(f"⚠ no '{fold_col}' column in the table/label CSV — "
                            f"splits must be given explicitly")

    task_type = task_cfg.get("task_type", "binary")
    data_cfg["task_type"] = task_type

    # ── Joint task: pull cls_cols / reg_cols / report_groups from the schema JSON ──
    if task_type == "classification_and_regression":
        schema_path = data_cfg.get("schema_json")
        if not schema_path:
            label_csv = data_cfg.get("label_csv", "")
            schema_path = str(Path(label_csv).with_suffix(".json")) if label_csv else None
        if schema_path and os.path.exists(schema_path):
            with open(schema_path) as fh:
                schema = json.load(fh)
            data_cfg.setdefault("cls_cols", schema.get("cls_cols", []))
            data_cfg.setdefault("reg_cols", schema.get("reg_cols", []))
            data_cfg.setdefault("report_groups", schema.get("report_groups", {}))
        num_cls = len(data_cfg.get("cls_cols") or [])
        num_reg = len(data_cfg.get("reg_cols") or [])
        num_classes = num_cls + num_reg
        task_cfg["num_cls"], task_cfg["num_reg"] = num_cls, num_reg
        if is_main_process():
            logging.info(f"  Joint MIMIC task: num_cls={num_cls}, num_reg={num_reg}, "
                         f"total head dim={num_classes}")
        if model_head_dim(model) != num_classes:
            raise ValueError(
                f"head was built with {model_head_dim(model)} outputs but the joint task "
                f"needs {num_classes}; set task.num_classes accordingly in the task yaml.")

    # ── Regression target z-normalisation from train-fold statistics ──
    znorm_cols = None
    if task_type == "regression":
        znorm_cols = data_cfg.get("label_cols")
    elif task_type == "classification_and_regression":
        znorm_cols = data_cfg.get("reg_cols")
    if znorm_cols and data_cfg.get("label_csv") and data_cfg.get("train_folds"):
        try:
            label_df_full = pd.read_csv(data_cfg["label_csv"], low_memory=False)
            zfold_col = data_cfg.get("fold_col", "strat_fold")
            train_rows = label_df_full[label_df_full[zfold_col].isin(data_cfg["train_folds"])]
            if all(c in train_rows.columns for c in znorm_cols):
                # ddof=0 to match sklearn.StandardScaler (mimic_preprocessing.py:431)
                t_mean = train_rows[znorm_cols].mean(axis=0).values.astype("float32")
                t_std = train_rows[znorm_cols].std(axis=0, ddof=0).values.astype("float32")
                data_cfg["target_mean"] = t_mean.tolist()
                data_cfg["target_std"] = t_std.tolist()
                if is_main_process():
                    logging.info(f"  Regression z-norm ({len(znorm_cols)} cols, train fold, ddof=0): "
                                 f"mean[:5]={t_mean[:5].tolist()} std[:5]={t_std[:5].tolist()}")
        except Exception as e:
            if is_main_process():
                logging.warning(f"  z-norm computation failed (using raw targets): {e}")

    train_ds, train_loader = build_dataloaders_ddp(data_cfg, "train")
    val_ds, val_loader = build_dataloaders_ddp(data_cfg, "val")

    has_test = (data_cfg.get("test_folds")
                or (data_cfg.get("loader_type") == "echonext_numpy"
                    and "test" in data_cfg.get("waveforms", {})))
    test_loader = build_dataloaders_ddp(data_cfg, "test")[1] if has_test else None

    if is_main_process():
        logging.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}"
                     + (f" | Test: {len(test_loader.dataset):,}" if test_loader else ""))

    # ── Train ──
    if task_type == "classification_and_regression":
        label_names = list(data_cfg.get("cls_cols", [])) + list(data_cfg.get("reg_cols", []))
    else:
        label_names = data_cfg.get("label_cols") or getattr(train_ds, "label_cols", None)
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

    if is_main_process():
        _append_result_csv(
            args=args, task=args.task, eval_mode=eval_mode,
            encoder_cls=args.encoder_cls or "dummy", num_classes=num_classes,
            save_dir=save_dir, train_size=len(train_ds), val_size=len(val_ds),
            test_size=len(test_loader.dataset) if test_loader else 0,
            results=results, task_type=task_type)
        logging.info(f"Results saved to: {save_dir}")

    cleanup_distributed()
    return results


def _check_num_classes(task_cfg, data_cfg, num_classes):
    """Cross-check the head width against the label file and the original.

    Three cohorts (chapman, cpsc_extra, ningbo) currently yield one label fewer
    than the published benchmark because the H5 store holds fewer records than
    the source datasets, which moves labels across the ``min_cnt=10`` cut. That
    is an ingest-side gap, so it is reported rather than silently absorbed —
    ``scripts/audit_cohorts.py`` prints the full picture.
    """
    label_csv = data_cfg.get("label_csv")
    actual = None
    if data_cfg.get("label_cols"):
        actual = len(data_cfg["label_cols"])
    elif label_csv and os.path.exists(label_csv):
        from src.dataset import NON_LABEL_COLS

        header = pd.read_csv(label_csv, nrows=0)
        actual = len([c for c in header.columns if c not in NON_LABEL_COLS])

    if actual is not None and actual != num_classes:
        if is_main_process():
            logging.warning(
                "task.num_classes=%d but %s provides %d label columns — using %d.",
                num_classes, Path(label_csv).name if label_csv else "the config",
                actual, actual)
        num_classes = actual

    expected = task_cfg.get("expected_num_classes")
    if expected is not None and expected != num_classes and is_main_process():
        logging.warning(
            "⚠ label-vocabulary mismatch vs the original benchmark: this run has "
            "%d classes, main_lite_ecg.py uses %d. Absolute numbers are NOT "
            "directly comparable to the published table for this task "
            "(run scripts/audit_cohorts.py for the cause).", num_classes, expected)
    return num_classes


def model_head_dim(model):
    """Output dimension of the classification head (DDP-aware)."""
    unwrapped = model.module if hasattr(model, "module") else model
    return unwrapped.num_classes


def _append_result_csv(args, task, eval_mode, encoder_cls, num_classes,
                       save_dir, train_size, val_size, test_size, results,
                       task_type="binary"):
    """Append one row per run to ``results_all.csv`` next to the run directory."""
    import csv
    import fcntl

    save_path = Path(save_dir)
    csv_path = save_path.parent / "results_all.csv"

    model_name = encoder_cls.rsplit(".", 1)[-1] if "." in encoder_cls else encoder_cls
    model_name = model_name.replace("Encoder", "").lower()

    test_m = _read_metrics(save_path / "test_metrics.txt")
    val_m = _read_metrics(save_path / "val_best_metrics.txt") or _read_metrics(save_path / "val_metrics.txt")

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
        "best_val": results.get("best_val", float("nan")),
        "best_epoch": results.get("best_epoch", -1),
        # Classification (paper macro definition: mean per-class AUC, 0.5 for unscoreable)
        "test_auroc_macro": test_m.get("auroc_macro", float("nan")),
        "test_auroc_macro_low": test_m.get("auroc_macro_low", float("nan")),
        "test_auroc_macro_high": test_m.get("auroc_macro_high", float("nan")),
        "test_auroc_macro_skipnan": test_m.get("auroc_macro_skipnan", float("nan")),
        "test_auroc_macro_noagg": test_m.get("noagg_auroc_macro", float("nan")),
        "test_auroc_micro": test_m.get("auroc_micro", float("nan")),
        "test_auprc_macro": test_m.get("auprc_macro", float("nan")),
        "test_f1_macro": test_m.get("f1_macro", float("nan")),
        "val_auroc_macro": val_m.get("auroc_macro", float("nan")),
        # Regression
        "test_mae_macro": test_m.get("mae_macro", float("nan")),
        "test_mae_global": test_m.get("mae_global", float("nan")),
        "test_mse_macro": test_m.get("mse_macro", float("nan")),
        "test_rmse_macro": test_m.get("rmse_macro", float("nan")),
        "test_r2_macro": test_m.get("r2_macro", float("nan")),
        "val_neg_mae_macro": val_m.get("neg_mae_macro", float("nan")),
        # Joint task
        "test_composite_score": test_m.get("composite_score", float("nan")),
        "test_auroc_macro_cls": test_m.get("auroc_macro_cls", float("nan")),
        "test_mae_global_reg": test_m.get("mae_global_reg", float("nan")),
        "n_windows": test_m.get("n_windows", float("nan")),
        "n_records": test_m.get("n_records", float("nan")),
        "save_dir": str(save_dir),
    }
    if task_type == "classification_and_regression":
        for key, val in test_m.items():
            if key.endswith("_auroc_macro") and key not in ("auroc_macro", "auroc_macro_cls"):
                row[f"test_{key}"] = val
            elif key.endswith("_mae_macro") or key.endswith("_mae_global"):
                row[f"test_{key}"] = val

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
    """Parse a ``key: value`` metrics text file into a float dict."""
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
