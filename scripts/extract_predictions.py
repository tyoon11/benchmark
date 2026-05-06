"""
Test-set predictions/targets 추출 (bootstrap 전처리)
====================================================
한 (model, task, mode) result 폴더에 대해 best.pt를 로드하고
test loader로 inference 한 뒤 preds.npy / targets.npy / ids.npy 를 저장합니다.

사용법:
  python scripts/extract_predictions.py --result_dir <RESULT_DIR>/<model>_<task>_<mode>
  python scripts/extract_predictions.py --root <RESULT_DIR>           # 전체 폴더 자동 추출
  python scripts/extract_predictions.py --root <RESULT_DIR> --filter cpc_chapman  # 부분 매칭

환경:
  ECG_DATA_ROOT   기본 /home/irteam/ddn-opendata1
  ECG_CKPT_ROOT   기본 /home/irteam/ddn-opendata1/model/ECGFMs

출력:
  <result_dir>/preds.npy        (N, C) float32 — sigmoid 확률 (분류) / raw (회귀)
  <result_dir>/targets.npy      (N, C) float32
  <result_dir>/ids.npy          (N,)   key — multi-window이면 ecg_id, 아니면 row index
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

# run.py 재사용
from run import load_config, load_encoder

# ──────────────────────────────────────────────────────────────────
# Model registry (configs/models.sh 의 bash assoc array를 mirror)
# ──────────────────────────────────────────────────────────────────
ECG_CKPT_ROOT = os.environ.get("ECG_CKPT_ROOT", "/home/irteam/ddn-opendata1/model/ECGFMs")

MODEL_CLS_MAP = {
    "ecg_founder": "src.encoders.ecg_founder.ECGFounderEncoder",
    "ecg_jepa":    "src.encoders.ecg_jepa.ECGJEPAEncoder",
    "st_mem":      "src.encoders.st_mem.StMemEncoder",
    "merl":        "src.encoders.merl.MerlResNetEncoder",
    "ecgfm_ked":   "src.encoders.ecgfm_ked.EcgFmKEDEncoder",
    "hubert_ecg":  "src.encoders.hubert_ecg.HuBERTECGEncoder",
    "ecg_fm":      "src.encoders.ecg_fm.ECGFMEncoder",
    "cpc":         "src.encoders.cpc.CPCEncoder",
}
MODEL_CKPT_MAP = {
    "ecg_founder": f"{ECG_CKPT_ROOT}/ecg_founder/12_lead_ECGFounder.pth",
    "ecg_jepa":    f"{ECG_CKPT_ROOT}/ecg_jepa/multiblock_epoch100.pth",
    "st_mem":      f"{ECG_CKPT_ROOT}/st_mem/st_mem_vit_base_full.pth",
    "merl":        f"{ECG_CKPT_ROOT}/merl/res18_best_encoder.pth",
    "ecgfm_ked":   f"{ECG_CKPT_ROOT}/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt",
    "hubert_ecg":  f"{ECG_CKPT_ROOT}/hubert_ecg/hubert_ecg_base.safetensors",
    "ecg_fm":      f"{ECG_CKPT_ROOT}/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt",
    "cpc":         f"{ECG_CKPT_ROOT}/cpc/last_11597276.ckpt",
}

EVAL_MODES = ["attention_probe", "finetune_attention", "finetune_linear", "linear_probe"]
# 정렬: 긴 모델명 우선 (ecgfm_ked가 ecg_*보다 먼저 매칭되도록 길이순)
MODEL_NAMES_BY_LEN = sorted(MODEL_CLS_MAP.keys(), key=len, reverse=True)

logger = logging.getLogger("extract_preds")


# ──────────────────────────────────────────────────────────────────
# 폴더명 파싱: <model>_<task>_<mode>  →  (model, task, mode)
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
# Test dataloader 빌드 (run.py 의 로직 mirror)
# ──────────────────────────────────────────────────────────────────
def build_test_loader(cfg):
    data_cfg = cfg.get("data", {})
    task_cfg = cfg.get("task", {})
    fold_cfg = cfg.get("fold", {})
    fold_col = fold_cfg.get("col", "strat_fold")
    auto_split = fold_cfg.get("auto_split", True)

    # auto_split: table CSV에서 strat_fold 자동 감지하여 test=max_fold
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

    # regression z-norm (train fold mean/std)
    task_type = task_cfg.get("task_type", "binary")
    data_cfg["task_type"] = task_type
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
        logits = model(signal)
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

    # multi-window aggregation: ecg_id 별 평균 (paper §3.3)
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
# 한 result_dir 처리
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

    # multi-window 확장 (인코더가 chunk_seconds를 노출하면)
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
    # 메타 (bootstrap 단계에서 task_type/label_cols 필요)
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
                   help="단일 실험 폴더 (예: results/.../cpc_chapman_attention_probe)")
    p.add_argument("--root", type=str, default=None,
                   help="전체 timestamp 폴더 (예: results/20260428_203028)")
    p.add_argument("--filter", type=str, default=None,
                   help="root 내 폴더명 substring 필터")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.result_dir:
        process_result_dir(Path(args.result_dir), args.device, args.force)
        return

    if not args.root:
        p.error("--result_dir 또는 --root 중 하나는 필요")

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
