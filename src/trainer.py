"""
Trainer
========
downstream task training/ . DDP support.
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
    downstream task Trainer (Single GPU + DDP support).

    Args:
        model:          DownstreamWrapper (or DDP-wrapped)
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
        self.model = model  # already device in  

        # DDP wrapper from original module 
        self.model_unwrapped = model.module if hasattr(model, "module") else model

        #  (original module's param_groups use)
        lr = float(self.cfg.get("lr", 1e-3))
        disc_lr = float(self.cfg.get("discriminative_lr_factor", 0.1))
        param_groups = self.model_unwrapped.get_param_groups(lr, disc_lr)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.cfg.get("weight_decay", 1e-3)),
        )

        #  (paper: const by default)
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

        # Loss — task_type  (reproduces paper main_lite_ecg.py:92-139)
        self.task_type = str(self.cfg.get("task_type", "binary"))
        self.num_cls = int(self.cfg.get("num_cls", 0))
        self.num_reg = int(self.cfg.get("num_reg", 0))
        self.report_groups = self.cfg.get("report_groups")  # dict for joint task per-group reporting
        self.cls_cols = self.cfg.get("cls_cols")
        self.reg_cols = self.cfg.get("reg_cols")
        if self.task_type == "regression":
            # paper: F.l1_loss (MAE) — MSE
            self._loss_fn = self._regression_loss
        elif self.task_type == "multi-label-binary":
            # NaN masking BCE — mds_ed's missing label handling
            self._loss_fn = self._masked_bce_loss
        elif self.task_type == "classification_and_regression":
            # paper main_lite_ecg.py:100-139: BCE(cls) + L1(reg), per-element NaN mask
            if self.num_cls <= 0 or self.num_reg <= 0:
                raise ValueError(
                    "classification_and_regression requires num_cls and num_reg in trainer cfg")
            self._loss_fn = self._joint_loss
        else:
            # 'binary' default — multi-label BCE (all label 0/1)
            self._loss_fn = nn.BCEWithLogitsLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # save (rank 0 only)
        self.save_dir = Path(self.cfg.get("save_dir", "./results"))
        if self.is_main:
            os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -float("inf")
        self.best_epoch = -1

        self.label_names = self.cfg.get("label_names")

    def train(self):
        """all training """
        if self.is_main:
            logger.info(f"Training for {self.epochs} epochs on {self.device}")
            logger.info(f"  eval_mode: {self.model_unwrapped.eval_mode}")
            logger.info(f"  world_size: {self.world_size}")
            logger.info(f"  train: {len(self.train_loader.dataset):,} | "
                         f"val: {len(self.val_loader.dataset):,}")

        for epoch in range(self.epochs):
            # DDP: epoch per sampler  config
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

            # Logging (rank 0 only) — task_type per primary metric
            if self.is_main:
                if self.task_type == "regression":
                    primary_key = "neg_mae_macro"   # neg_MAE ( , MAE↓)
                    log_label = "neg_mae"
                elif self.task_type == "classification_and_regression":
                    primary_key = "neg_composite_score"  # paper: (1-macro_AUROC)+mae, lower=better → maximize neg
                    log_label = "neg_composite"
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
            if self.task_type == "regression":
                log_label = "neg_mae"
            elif self.task_type == "classification_and_regression":
                log_label = "neg_composite"
            else:
                log_label = "AUROC"
            logger.info(f"Best val {log_label}: {self.best_metric:.4f} at epoch {self.best_epoch}")

        # Test
        if self.test_loader and self.is_main:
            self.model_unwrapped.load_state_dict(
                torch.load(self.save_dir / "best.pt", weights_only=True))
            test_metrics = self._eval_epoch(self.test_loader, "test")
            if self.task_type == "regression":
                primary_key, log_label = "neg_mae_macro", "neg_mae"
            elif self.task_type == "classification_and_regression":
                primary_key, log_label = "neg_composite_score", "neg_composite"
            else:
                primary_key, log_label = "auroc_macro", "AUROC"
            logger.info(f"Test {log_label}: {test_metrics.get(primary_key, 0):.4f}")
            return test_metrics

        return {"best_val": self.best_metric, "best_epoch": self.best_epoch}

    def _train_epoch(self, epoch):
        """1 epoch training"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch+1}",
                     leave=False, disable=not self.is_main)
        for batch in pbar:
            signal = batch["signal"].to(self.device)
            label = batch["label"].to(self.device)

            # Forward moryecg cache keys when present; non-moryecg encoders
            # accept **_unused and ignore these.
            extra = {}
            if "ecg_filepath" in batch:
                extra["ecg_filepath"] = batch["ecg_filepath"]
            if "ecg_seg_idx" in batch:
                extra["ecg_seg_idx"] = batch["ecg_seg_idx"]
            logits = self.model(signal, **extra)
            loss = self._loss_fn(logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # DDP: loss all-reduce by 
        avg_loss = total_loss / max(n_batches, 1)
        if self.use_ddp:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    @torch.no_grad()
    def _eval_epoch(self, loader, prefix="val"):
        """ (rank 0 from only all data by ).

        Multi-window mode: each ECG N chunk by  if present, batch in ecg_id
         . all chunk example ecg_id per by   metric compute
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

            extra = {}
            if "ecg_filepath" in batch:
                extra["ecg_filepath"] = batch["ecg_filepath"]
            if "ecg_seg_idx" in batch:
                extra["ecg_seg_idx"] = batch["ecg_seg_idx"]
            logits = self.model(signal, **extra)
            # regression: raw logits, classification: sigmoid,
            # joint (cls+reg): sigmoid on first num_cls, raw on the rest
            if self.task_type == "regression":
                preds = logits.cpu().numpy()
            elif self.task_type == "classification_and_regression":
                cls = torch.sigmoid(logits[:, :self.num_cls])
                preds = torch.cat([cls, logits[:, self.num_cls:]], dim=1).cpu().numpy()
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

        # DDP: all rank's example gather (rank 0 from  compute)
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

            metrics = evaluate_all(
                all_targets, all_preds, self.label_names,
                task_type=self.task_type,
                num_cls=self.num_cls if self.task_type == "classification_and_regression" else None,
                cls_cols=self.cls_cols,
                reg_cols=self.reg_cols,
                report_groups=self.report_groups,
            )
            result_path = self.save_dir / f"{prefix}_metrics.txt"
            with open(result_path, "w") as f:
                for k, v in sorted(metrics.items()):
                    f.write(f"{k}: {v}\n")

        # DDP:  broadcast — task_type per primary metric
        if self.use_ddp:
            if self.task_type == "regression":
                primary_key = "neg_mae_macro"
            elif self.task_type == "classification_and_regression":
                primary_key = "neg_composite_score"
            else:
                primary_key = "auroc_macro"
            primary = torch.tensor(metrics.get(primary_key, 0.0), device=self.device)
            dist.broadcast(primary, src=0)
            if not self.is_main:
                metrics = {primary_key: primary.item()}

        return metrics

    def _gather_predictions(self, preds, targets, ids=None):
        """all rank's example (+ optional ecg_ids) rank 0 as gather"""
        preds_t = torch.from_numpy(preds).to(self.device)
        targets_t = torch.from_numpy(targets).to(self.device)
        ids_t = (torch.from_numpy(ids).to(self.device).long()
                 if ids is not None else None)

        # each rank's data size   size first, 
        local_size = torch.tensor(preds_t.shape[0], device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, local_size)

        max_size = max(s.item() for s in all_sizes)

        #  to  same size by 
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
            #  remove
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

    # ── Loss helpers (reproduces paper main_lite_ecg.py:92-139) ──────────────
    @staticmethod
    def _masked_bce_loss(logits, targets):
        """NaN  multi-label binary (for) — NaN location loss in .
        reproduces paper main_lite_ecg.py:114-118 (per-position mask).
        """
        import torch.nn.functional as F
        valid = ~torch.isnan(targets)
        if not valid.any():
            return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
        # NaN location 0 as  BCE input available, mask by remove
        targets_safe = torch.where(valid, targets, torch.zeros_like(targets))
        loss_per_elem = F.binary_cross_entropy_with_logits(
            logits, targets_safe, reduction="none"
        )
        return (loss_per_elem * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    @staticmethod
    def _regression_loss(logits, targets):
        """L1 loss (MAE) with NaN masking. reproduces paper main_lite_ecg.py:99,128-133."""
        import torch.nn.functional as F
        valid = ~torch.isnan(targets)
        if not valid.any():
            return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
        targets_safe = torch.where(valid, targets, torch.zeros_like(targets))
        loss_per_elem = F.l1_loss(logits, targets_safe, reduction="none")
        return (loss_per_elem * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    def _joint_loss(self, logits, targets):
        """Composite BCE(cls) + L1(reg) with per-element NaN masking.
        Reproduces paper main_lite_ecg.py:100-139 (classification_and_regression_loss).
        Output split: logits[:, :num_cls] = cls logits, logits[:, num_cls:] = reg.
        """
        import torch.nn.functional as F
        nc = self.num_cls
        cls_logits = logits[:, :nc]
        reg_logits = logits[:, nc:]
        cls_targs = targets[:, :nc]
        reg_targs = targets[:, nc:]

        cls_valid = ~torch.isnan(cls_targs)
        if cls_valid.any():
            cls_safe = torch.where(cls_valid, cls_targs, torch.zeros_like(cls_targs))
            cls_per = F.binary_cross_entropy_with_logits(
                cls_logits, cls_safe, reduction="none")
            cls_loss = (cls_per * cls_valid.float()).sum() / cls_valid.float().sum().clamp(min=1.0)
        else:
            cls_loss = torch.zeros((), device=logits.device)

        reg_valid = ~torch.isnan(reg_targs)
        if reg_valid.any():
            reg_safe = torch.where(reg_valid, reg_targs, torch.zeros_like(reg_targs))
            reg_per = F.l1_loss(reg_logits, reg_safe, reduction="none")
            reg_loss = (reg_per * reg_valid.float()).sum() / reg_valid.float().sum().clamp(min=1.0)
        else:
            reg_loss = torch.zeros((), device=logits.device)

        return cls_loss + reg_loss
