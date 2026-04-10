"""
ECG Downstream Benchmark
=========================
H5 기반 ECG 다운스트림 태스크 벤치마크.

사용법:
  # Linear Probe (PTB-XL)
  python run.py --task ptbxl_super --eval_mode linear_probe \
      --encoder_cls my_models.MyEncoder --encoder_ckpt weights/encoder.pt

  # Attention Probe
  python run.py --task ptbxl_super --eval_mode attention_probe \
      --encoder_cls my_models.MyEncoder

  # Full Finetuning + Linear Head
  python run.py --task code15_diag --eval_mode finetune_linear \
      --encoder_cls my_models.MyEncoder --epochs 30

  # Full Finetuning + Attention Head
  python run.py --task physionet_all --eval_mode finetune_attention \
      --encoder_cls my_models.MyEncoder --lr 5e-4

  # 더미 인코더로 테스트
  python run.py --task ptbxl_super --eval_mode linear_probe --dummy
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
import numpy as np

# src 경로 추가
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import H5ECGDataset, build_dataloaders
from src.wrapper import DownstreamWrapper
from src.trainer import DownstreamTrainer


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
        # x: (B, C, T)
        feat = self.conv(x)           # (B, D, T)
        seq_feat = feat.transpose(1, 2)  # (B, T, D)
        pooled = feat.mean(dim=2)     # (B, D)
        return seq_feat, pooled


# ═══════════════════════════════════════════════════════════════
# Config 로딩
# ═══════════════════════════════════════════════════════════════
def load_config(task_name: str, overrides: dict = None) -> dict:
    """태스크 config + default config 병합"""
    cfg_dir = SCRIPT_DIR / "configs"

    # Default
    default_path = cfg_dir / "default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    # Task config
    task_path = cfg_dir / "tasks" / f"{task_name}.yaml"
    if task_path.exists():
        with open(task_path) as f:
            task_cfg = yaml.safe_load(f)
        # 병합 (task가 default를 덮어씀)
        for section in task_cfg:
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(task_cfg[section])
            else:
                cfg[section] = task_cfg[section]

    # CLI override
    if overrides:
        for k, v in overrides.items():
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v

    return cfg


def load_encoder(encoder_cls: str, encoder_ckpt: str = None, **kwargs):
    """
    문자열로 지정된 encoder 클래스를 import하고 인스턴스화합니다.

    encoder_cls: "module.path.ClassName" (예: "my_models.ECGEncoder")
    """
    module_path, cls_name = encoder_cls.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    encoder = cls(**kwargs)

    if encoder_ckpt:
        state = torch.load(encoder_ckpt, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=False)
        logging.info(f"Loaded encoder from {encoder_ckpt}")

    # feature_dim 추출
    feature_dim = getattr(encoder, "feature_dim", None)
    if feature_dim is None:
        feature_dim = getattr(encoder, "embed_dim", None)
    if feature_dim is None:
        raise ValueError("Encoder must have 'feature_dim' or 'embed_dim' attribute")

    return encoder, feature_dim


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ECG Downstream Benchmark")
    parser.add_argument("--task", type=str, required=True,
                        help="태스크 이름 (configs/tasks/*.yaml)")
    parser.add_argument("--eval_mode", type=str, default="linear_probe",
                        choices=["linear_probe", "attention_probe",
                                 "finetune_linear", "finetune_attention"],
                        help="평가 모드")
    parser.add_argument("--encoder_cls", type=str, default=None,
                        help="인코더 클래스 (module.ClassName)")
    parser.add_argument("--encoder_ckpt", type=str, default=None,
                        help="인코더 체크포인트 경로")
    parser.add_argument("--dummy", action="store_true",
                        help="더미 인코더로 테스트")

    # Override
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    # Fold 설정
    parser.add_argument("--train_folds", type=str, default=None,
                        help="train fold IDs (쉼표 구분, 예: 0,1,2,...,7)")
    parser.add_argument("--val_folds", type=str, default=None,
                        help="val fold IDs (예: 8)")
    parser.add_argument("--test_folds", type=str, default=None,
                        help="test fold IDs (예: 9)")

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

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

    # Save dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or str(
        SCRIPT_DIR / "results" / f"{args.task}_{eval_mode}_{timestamp}"
    )

    logging.info(f"Task: {args.task} | Mode: {eval_mode} | Classes: {num_classes}")

    # ── Encoder ──
    if args.dummy:
        n_leads = 12
        encoder = DummyEncoder(n_leads=n_leads, feature_dim=256)
        feature_dim = 256
        logging.info("Using DummyEncoder (feature_dim=256)")
    elif args.encoder_cls:
        encoder, feature_dim = load_encoder(args.encoder_cls, args.encoder_ckpt)
        logging.info(f"Encoder: {args.encoder_cls} (feature_dim={feature_dim})")
    else:
        parser.error("--encoder_cls or --dummy required")

    # ── Model Wrapper ──
    model = DownstreamWrapper(
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=num_classes,
        eval_mode=eval_mode,
        head_kwargs=head_cfg,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # ── Data ──
    # Fold 설정
    if args.train_folds:
        data_cfg["train_folds"] = [int(x) for x in args.train_folds.split(",")]
    if args.val_folds:
        data_cfg["val_folds"] = [int(x) for x in args.val_folds.split(",")]
    if args.test_folds:
        data_cfg["test_folds"] = [int(x) for x in args.test_folds.split(",")]

    train_ds, train_loader = build_dataloaders(data_cfg, "train")
    val_ds, val_loader = build_dataloaders(data_cfg, "val")

    test_loader = None
    if data_cfg.get("test_folds"):
        _, test_loader = build_dataloaders(data_cfg, "test")

    logging.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}"
                 + (f" | Test: {len(test_loader.dataset):,}" if test_loader else ""))

    # ── Train ──
    trainer_cfg = {
        **train_cfg,
        "save_dir": save_dir,
        "label_names": data_cfg.get("label_cols"),
    }
    trainer = DownstreamTrainer(model, train_loader, val_loader, test_loader, trainer_cfg)
    results = trainer.train()

    logging.info(f"Results saved to: {save_dir}")
    return results


if __name__ == "__main__":
    main()
