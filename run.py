"""
ECG Downstream Benchmark
=========================
H5 기반 ECG 다운스트림 태스크 벤치마크.

사용법:
  # Single GPU
  python run.py --task ptbxl_super_jepa --eval_mode linear_probe \
      --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
      --encoder_ckpt weights/encoder.pt

  # Multi GPU (예: 4장)
  torchrun --nproc_per_node=4 run.py --task ptbxl_super_jepa \
      --eval_mode finetune_linear \
      --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
      --encoder_ckpt weights/encoder.pt

  # 특정 GPU 지정
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run.py ...

  # 더미 인코더 테스트
  python run.py --task ptbxl_super_jepa --eval_mode linear_probe --dummy
"""

import os
import sys
import argparse
import logging
import yaml
import importlib
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np

# src 경로 추가
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import H5ECGDataset, build_dataloaders
from src.wrapper import DownstreamWrapper
from src.trainer import DownstreamTrainer


# ═══════════════════════════════════════════════════════════════
# DDP 유틸
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
    """torchrun이 설정한 환경변수로 DDP 초기화"""
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
# 더미 인코더 (테스트용)
# ═══════════════════════════════════════════════════════════════
class DummyEncoder(torch.nn.Module):
    """테스트용 더미 인코더. GAP → (B, feature_dim)"""
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
# Config 로딩
# ═══════════════════════════════════════════════════════════════
def load_config(task_name: str, overrides: dict = None) -> dict:
    cfg_dir = SCRIPT_DIR / "configs"
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

    return cfg


def load_encoder(encoder_cls: str, encoder_ckpt: str = None, **kwargs):
    module_path, cls_name = encoder_cls.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    encoder = cls(**kwargs)

    if encoder_ckpt:
        state = torch.load(encoder_ckpt, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=False)
        if is_main_process():
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
    """DDP를 고려한 DataLoader 생성"""
    from torch.utils.data import DataLoader

    ds = H5ECGDataset(
        h5_root=data_cfg["h5_root"],
        table_csv=data_cfg["table_csv"],
        label_csv=data_cfg.get("label_csv"),
        label_cols=data_cfg.get("label_cols"),
        target_fs=data_cfg.get("target_fs"),
        target_length=data_cfg.get("target_length"),
        seg_idx=data_cfg.get("seg_idx", None),
        normalize=data_cfg.get("normalize", False),
        fold_col=data_cfg.get("fold_col"),
        fold_ids=data_cfg.get(f"{split}_folds"),
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
    )

    sampler = None
    shuffle = (split == "train")
    if is_distributed():
        sampler = DistributedSampler(ds, shuffle=shuffle)
        shuffle = False  # sampler가 shuffle 담당

    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return ds, loader


# ═══════════════════════════════════════════════════════════════
# 메인
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

    # DDP 초기화
    use_ddp = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    # Logging (rank 0만)
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
        SCRIPT_DIR / "results" / f"{args.task}_{eval_mode}_{timestamp}"
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

    # ── Model Wrapper ──
    model = DownstreamWrapper(
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=num_classes,
        eval_mode=eval_mode,
        head_kwargs=head_cfg,
    )

    # Device 설정
    if use_ddp:
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        # DDP wrapper에서 원본 모듈 접근
        model_unwrapped = model.module
    else:
        device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)
        model_unwrapped = model

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # ── Data ──
    if args.train_folds:
        data_cfg["train_folds"] = [int(x) for x in args.train_folds.split(",")]
    if args.val_folds:
        data_cfg["val_folds"] = [int(x) for x in args.val_folds.split(",")]
    if args.test_folds:
        data_cfg["test_folds"] = [int(x) for x in args.test_folds.split(",")]

    train_ds, train_loader = build_dataloaders_ddp(data_cfg, "train")
    val_ds, val_loader = build_dataloaders_ddp(data_cfg, "val")

    test_loader = None
    if data_cfg.get("test_folds"):
        _, test_loader = build_dataloaders_ddp(data_cfg, "test")

    if is_main_process():
        logging.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}"
                     + (f" | Test: {len(test_loader.dataset):,}" if test_loader else ""))

    # ── Train ──
    trainer_cfg = {
        **train_cfg,
        "save_dir": save_dir,
        "label_names": data_cfg.get("label_cols"),
        "device": str(device),
        "use_ddp": use_ddp,
        "rank": rank,
        "world_size": world_size,
    }
    trainer = DownstreamTrainer(model, train_loader, val_loader, test_loader, trainer_cfg)
    results = trainer.train()

    if is_main_process():
        logging.info(f"Results saved to: {save_dir}")

    cleanup_distributed()
    return results


if __name__ == "__main__":
    main()
