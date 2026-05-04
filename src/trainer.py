"""
Trainer
========
다운스트림 태스크 학습/평가 루프. DDP 지원.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from .metrics import evaluate_all

logger = logging.getLogger(__name__)


class DownstreamTrainer:
    """
    다운스트림 태스크 Trainer (Single GPU + DDP 지원).

    Args:
        model:          DownstreamWrapper (또는 DDP-wrapped)
        train_loader:   DataLoader
        val_loader:     DataLoader
        test_loader:    DataLoader (optional)
        cfg:            config dict
    """

    def __init__(self, model, train_loader, val_loader, test_loader=None, cfg=None):
        self.cfg = cfg or {}
        self.use_ddp = self.cfg.get("use_ddp", False)
        self.rank = self.cfg.get("rank", 0)
        self.world_size = self.cfg.get("world_size", 1)
        self.is_main = (self.rank == 0)

        self.device = torch.device(self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model  # 이미 device에 올라가 있음

        # DDP wrapper에서 원본 모듈 접근
        self.model_unwrapped = model.module if hasattr(model, "module") else model

        # 옵티마이저 (원본 모듈의 param_groups 사용)
        lr = float(self.cfg.get("lr", 1e-3))
        disc_lr = float(self.cfg.get("discriminative_lr_factor", 0.1))
        param_groups = self.model_unwrapped.get_param_groups(lr, disc_lr)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.cfg.get("weight_decay", 0.01)),
        )

        # 스케줄러 (paper: const by default)
        self.epochs = int(self.cfg.get("epochs", 100))
        warmup = int(self.cfg.get("warmup_epochs", 0))
        schedule = str(self.cfg.get("lr_schedule", "const"))
        self.lr_schedule = schedule
        if schedule == "const":
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.epochs - warmup, 1),
                eta_min=float(self.cfg.get("lr_min", 1e-6)),
            )
        self.warmup_epochs = warmup
        self.lr = lr

        # Loss — task_type 분기 (paper main_lite_ecg.py:92-139 재현)
        self.task_type = str(self.cfg.get("task_type", "binary"))
        if self.task_type == "regression":
            # paper: F.l1_loss (MAE) — MSE 아님
            self._loss_fn = self._regression_loss
        elif self.task_type == "multi-label-binary":
            # NaN 마스킹 BCE — mds_ed의 missing label 처리
            self._loss_fn = self._masked_bce_loss
        else:
            # 'binary' 기본 — multi-label BCE (전체 라벨 0/1)
            self._loss_fn = nn.BCEWithLogitsLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 저장 (rank 0만)
        self.save_dir = Path(self.cfg.get("save_dir", "./results"))
        if self.is_main:
            os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -float("inf")
        self.best_epoch = -1

        self.label_names = self.cfg.get("label_names")

    def train(self):
        """전체 학습 루프"""
        if self.is_main:
            logger.info(f"Training for {self.epochs} epochs on {self.device}")
            logger.info(f"  eval_mode: {self.model_unwrapped.eval_mode}")
            logger.info(f"  world_size: {self.world_size}")
            logger.info(f"  train: {len(self.train_loader.dataset):,} | "
                         f"val: {len(self.val_loader.dataset):,}")

        for epoch in range(self.epochs):
            # DDP: epoch별 sampler 시드 설정
            if self.use_ddp and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # Warmup LR (only if warmup_epochs > 0)
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                warmup_lr = self.lr * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    if "_lr_ratio" not in pg:
                        pg["_lr_ratio"] = pg["lr"] / self.lr if self.lr > 0 else 1.0
                    pg["lr"] = warmup_lr * pg["_lr_ratio"]

            train_loss = self._train_epoch(epoch)
            val_metrics = self._eval_epoch(self.val_loader, "val")

            if self.scheduler is not None and epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Logging (rank 0만) — task_type별 primary metric
            if self.is_main:
                if self.task_type == "regression":
                    primary_key = "neg_mae_macro"   # neg_MAE (높을수록 좋음, MAE↓)
                    log_label = "neg_mae"
                else:
                    primary_key = "auroc_macro"
                    log_label = "auroc"
                primary = val_metrics.get(primary_key, 0)
                lr_now = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"loss={train_loss:.4f} | val_{log_label}={primary:.4f} | lr={lr_now:.2e}"
                )

                if primary > self.best_metric:
                    self.best_metric = primary
                    self.best_epoch = epoch + 1
                    torch.save(self.model_unwrapped.state_dict(), self.save_dir / "best.pt")

        if self.is_main:
            log_label = "neg_mae" if self.task_type == "regression" else "AUROC"
            logger.info(f"Best val {log_label}: {self.best_metric:.4f} at epoch {self.best_epoch}")

        # Test
        if self.test_loader and self.is_main:
            self.model_unwrapped.load_state_dict(
                torch.load(self.save_dir / "best.pt", weights_only=True))
            test_metrics = self._eval_epoch(self.test_loader, "test")
            primary_key = "neg_mae_macro" if self.task_type == "regression" else "auroc_macro"
            log_label = "neg_mae" if self.task_type == "regression" else "AUROC"
            logger.info(f"Test {log_label}: {test_metrics.get(primary_key, 0):.4f}")
            return test_metrics

        return {"best_val": self.best_metric, "best_epoch": self.best_epoch}

    def _train_epoch(self, epoch):
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch+1}",
                     leave=False, disable=not self.is_main)
        for batch in pbar:
            signal = batch["signal"].to(self.device)
            label = batch["label"].to(self.device)

            logits = self.model(signal)
            loss = self._loss_fn(logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # DDP: loss를 all-reduce로 평균
        avg_loss = total_loss / max(n_batches, 1)
        if self.use_ddp:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    @torch.no_grad()
    def _eval_epoch(self, loader, prefix="val"):
        """평가 (rank 0에서만 전체 데이터로 평가).

        Multi-window mode: 각 ECG가 N개 chunk로 쪼개져 있으면 batch에 ecg_id가
        들어 있다. 모든 chunk 예측을 ecg_id별로 평균집계한 뒤 metric 계산
        (paper §3.3 aggregate_predictions, aggregate_fn=np.mean).
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_ids = []

        pbar = tqdm(loader, desc=f"Eval {prefix}",
                     leave=False, disable=not self.is_main)
        for batch in pbar:
            signal = batch["signal"].to(self.device)
            label = batch["label"]

            logits = self.model(signal)
            # regression: raw logits, classification: sigmoid 확률
            if self.task_type == "regression":
                preds = logits.cpu().numpy()
            else:
                preds = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(label.numpy())
            if "ecg_id" in batch:
                ids = batch["ecg_id"]
                if isinstance(ids, torch.Tensor):
                    ids = ids.numpy()
                else:
                    ids = np.asarray(ids)
                all_ids.append(ids)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_ids = np.concatenate(all_ids, axis=0) if all_ids else None

        # DDP: 모든 rank의 예측을 gather (rank 0에서 메트릭 계산)
        if self.use_ddp:
            all_preds, all_targets, all_ids = self._gather_predictions(
                all_preds, all_targets, all_ids
            )

        metrics = {}
        if self.is_main:
            # ── Aggregate by ecg_id (mean over non-overlapping chunks) ──
            if all_ids is not None and len(all_ids) > 0 and len(all_ids) != len(np.unique(all_ids)):
                unique_ids = np.unique(all_ids)
                agg_preds = np.empty((len(unique_ids), all_preds.shape[1]),
                                     dtype=all_preds.dtype)
                agg_targets = np.empty((len(unique_ids), all_targets.shape[1]),
                                       dtype=all_targets.dtype)
                for i, uid in enumerate(unique_ids):
                    mask = (all_ids == uid)
                    agg_preds[i] = all_preds[mask].mean(axis=0)
                    agg_targets[i] = all_targets[mask][0]
                all_preds = agg_preds
                all_targets = agg_targets

            metrics = evaluate_all(all_targets, all_preds, self.label_names,
                                    task_type=self.task_type)
            result_path = self.save_dir / f"{prefix}_metrics.txt"
            with open(result_path, "w") as f:
                for k, v in sorted(metrics.items()):
                    f.write(f"{k}: {v}\n")

        # DDP: 메트릭을 broadcast — task_type별 primary metric
        if self.use_ddp:
            primary_key = "neg_mae_macro" if self.task_type == "regression" else "auroc_macro"
            primary = torch.tensor(metrics.get(primary_key, 0.0), device=self.device)
            dist.broadcast(primary, src=0)
            if not self.is_main:
                metrics = {primary_key: primary.item()}

        return metrics

    def _gather_predictions(self, preds, targets, ids=None):
        """모든 rank의 예측 (+ optional ecg_ids) 을 rank 0으로 gather"""
        preds_t = torch.from_numpy(preds).to(self.device)
        targets_t = torch.from_numpy(targets).to(self.device)
        ids_t = (torch.from_numpy(ids).to(self.device).long()
                 if ids is not None else None)

        # 각 rank의 데이터 크기가 다를 수 있으므로 크기 먼저 수집
        local_size = torch.tensor(preds_t.shape[0], device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, local_size)

        max_size = max(s.item() for s in all_sizes)

        # 패딩해서 같은 크기로 맞추기
        if preds_t.shape[0] < max_size:
            pad_size = max_size - preds_t.shape[0]
            preds_t = torch.cat([preds_t, torch.zeros(pad_size, *preds_t.shape[1:], device=self.device)])
            targets_t = torch.cat([targets_t, torch.zeros(pad_size, *targets_t.shape[1:], device=self.device)])
            if ids_t is not None:
                ids_t = torch.cat([ids_t, torch.full((pad_size,), -1, device=self.device, dtype=ids_t.dtype)])

        gathered_preds = [torch.zeros_like(preds_t) for _ in range(self.world_size)]
        gathered_targets = [torch.zeros_like(targets_t) for _ in range(self.world_size)]
        dist.all_gather(gathered_preds, preds_t)
        dist.all_gather(gathered_targets, targets_t)
        if ids_t is not None:
            gathered_ids = [torch.zeros_like(ids_t) for _ in range(self.world_size)]
            dist.all_gather(gathered_ids, ids_t)

        if self.is_main:
            # 패딩 제거
            all_p, all_t, all_i = [], [], []
            for i, size in enumerate(all_sizes):
                n = size.item()
                all_p.append(gathered_preds[i][:n])
                all_t.append(gathered_targets[i][:n])
                if ids_t is not None:
                    all_i.append(gathered_ids[i][:n])
            preds_out = torch.cat(all_p).cpu().numpy()
            targets_out = torch.cat(all_t).cpu().numpy()
            ids_out = torch.cat(all_i).cpu().numpy() if all_i else None
            return preds_out, targets_out, ids_out

        return preds, targets, ids

    # ── Loss helpers (paper main_lite_ecg.py:92-139 재현) ──────────────
    @staticmethod
    def _masked_bce_loss(logits, targets):
        """NaN을 가진 multi-label binary용 — NaN 위치는 loss에 미반영.
        paper main_lite_ecg.py:114-118 재현 (per-position mask).
        """
        import torch.nn.functional as F
        valid = ~torch.isnan(targets)
        if not valid.any():
            return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
        # NaN 위치는 0으로 대체하여 BCE 입력 가능, mask로 제거
        targets_safe = torch.where(valid, targets, torch.zeros_like(targets))
        loss_per_elem = F.binary_cross_entropy_with_logits(
            logits, targets_safe, reduction="none"
        )
        return (loss_per_elem * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    @staticmethod
    def _regression_loss(logits, targets):
        """L1 loss (MAE) with NaN masking. paper main_lite_ecg.py:99,128-133 재현."""
        import torch.nn.functional as F
        valid = ~torch.isnan(targets)
        if not valid.any():
            return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
        targets_safe = torch.where(valid, targets, torch.zeros_like(targets))
        loss_per_elem = F.l1_loss(logits, targets_safe, reduction="none")
        return (loss_per_elem * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
