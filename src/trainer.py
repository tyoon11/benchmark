"""
Trainer
========
다운스트림 태스크 학습/평가 루프.
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from .metrics import evaluate_all

logger = logging.getLogger(__name__)


class DownstreamTrainer:
    """
    다운스트림 태스크 Trainer.

    Args:
        model:          DownstreamWrapper
        train_loader:   DataLoader
        val_loader:     DataLoader
        test_loader:    DataLoader (optional)
        cfg:            config dict
    """

    def __init__(self, model, train_loader, val_loader, test_loader=None, cfg=None):
        self.cfg = cfg or {}
        self.device = torch.device(self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)

        # 옵티마이저
        lr = self.cfg.get("lr", 1e-3)
        disc_lr = self.cfg.get("discriminative_lr_factor", 0.1)
        param_groups = self.model.get_param_groups(lr, disc_lr)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.get("weight_decay", 0.01),
        )

        # 스케줄러
        self.epochs = self.cfg.get("epochs", 50)
        warmup = self.cfg.get("warmup_epochs", 5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - warmup, eta_min=self.cfg.get("lr_min", 1e-6)
        )
        self.warmup_epochs = warmup

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 저장
        self.save_dir = Path(self.cfg.get("save_dir", "./results"))
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -float("inf")
        self.best_epoch = -1

        self.label_names = self.cfg.get("label_names")

    def train(self):
        """전체 학습 루프"""
        logger.info(f"Training for {self.epochs} epochs on {self.device}")
        logger.info(f"  eval_mode: {self.model.eval_mode}")
        logger.info(f"  train: {len(self.train_loader.dataset):,} | "
                     f"val: {len(self.val_loader.dataset):,}")

        for epoch in range(self.epochs):
            # Warmup LR
            if epoch < self.warmup_epochs:
                warmup_lr = self.cfg.get("lr", 1e-3) * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr * pg.get("_lr_ratio", 1.0)
                    if "_lr_ratio" not in pg:
                        pg["_lr_ratio"] = pg["lr"] / warmup_lr

            train_loss = self._train_epoch(epoch)
            val_metrics = self._eval_epoch(self.val_loader, "val")

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Logging
            auroc = val_metrics.get("auroc_macro", 0)
            lr_now = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"loss={train_loss:.4f} | val_auroc={auroc:.4f} | lr={lr_now:.2e}"
            )

            # Best model 저장
            if auroc > self.best_metric:
                self.best_metric = auroc
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.save_dir / "best.pt")

        logger.info(f"Best val AUROC: {self.best_metric:.4f} at epoch {self.best_epoch}")

        # Test
        if self.test_loader:
            self.model.load_state_dict(torch.load(self.save_dir / "best.pt", weights_only=True))
            test_metrics = self._eval_epoch(self.test_loader, "test")
            logger.info(f"Test AUROC: {test_metrics.get('auroc_macro', 0):.4f}")
            return test_metrics

        return {"best_val_auroc": self.best_metric, "best_epoch": self.best_epoch}

    def _train_epoch(self, epoch):
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Train {epoch+1}", leave=False):
            signal = batch["signal"].to(self.device)   # (B, C, T)
            label = batch["label"].to(self.device)     # (B, num_classes)

            logits = self.model(signal)
            loss = self.criterion(logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader, prefix="val"):
        """평가"""
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in tqdm(loader, desc=f"Eval {prefix}", leave=False):
            signal = batch["signal"].to(self.device)
            label = batch["label"]

            logits = self.model(signal)
            preds = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(label.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = evaluate_all(all_targets, all_preds, self.label_names)

        # 결과 저장
        result_path = self.save_dir / f"{prefix}_metrics.txt"
        with open(result_path, "w") as f:
            for k, v in sorted(metrics.items()):
                f.write(f"{k}: {v}\n")

        return metrics
