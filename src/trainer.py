"""
Trainer
=======
Downstream train/eval loop, aligned with the original ``ecg-fm-benchmarking``
Lightning module (``main_lite_base.Main_Lite``). DDP-aware.

Parity notes
------------
* **Optimiser**: AdamW, constant LR by default, discriminative LR per layer
  group (head / late / early = ``lr``, ``lr*f``, ``lr*f^2``).
* **Precision**: the original trains at ``16-mixed`` (``32`` for the S4/CPC
  models). ``cfg["precision"]`` selects ``32`` | ``16-mixed`` | ``bf16-mixed``.
* **Evaluation**: metrics are reported twice per split — ``noagg`` over every
  window and ``agg`` after averaging the windows of each record
  (``aggregate_fn=np.mean``). Checkpoint selection uses the **agg** metric, as
  in the original ``ModelCheckpoint(monitor="macro_auc_agg_val0")``.
* **Test**: run once on the best checkpoint, with 1000-iteration empirical
  bootstrap CIs and optional ``.npz`` prediction export.
"""

import logging
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from .metrics import bootstrap_metrics, evaluate_all

logger = logging.getLogger(__name__)


_AMP_DTYPE = {
    "16-mixed": torch.float16,
    "16": torch.float16,
    "fp16": torch.float16,
    "bf16-mixed": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class DownstreamTrainer:
    """Downstream trainer (single GPU + DDP).

    Args:
        model:        DownstreamWrapper (possibly DDP-wrapped)
        train_loader / val_loader / test_loader: DataLoaders
        cfg:          merged ``train`` config plus runtime fields injected by run.py
    """

    def __init__(self, model, train_loader, val_loader, test_loader=None, cfg=None):
        self.cfg = cfg or {}
        self.use_ddp = self.cfg.get("use_ddp", False)
        self.rank = self.cfg.get("rank", 0)
        self.world_size = self.cfg.get("world_size", 1)
        self.is_main = (self.rank == 0)

        self.device = torch.device(self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model
        self.model_unwrapped = model.module if hasattr(model, "module") else model

        # ── optimiser (discriminative LR groups come from the wrapper) ──
        lr = float(self.cfg.get("lr", 1e-3))
        disc_lr = float(self.cfg.get("discriminative_lr_factor", 0.1))
        param_groups = self.model_unwrapped.get_param_groups(lr, disc_lr)
        param_groups = [g for g in param_groups if len(list(g["params"])) > 0]
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(self.cfg.get("weight_decay", 1e-3)),
        )

        # ── LR schedule (original default: const) ──
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

        # ── mixed precision ──
        precision = str(self.cfg.get("precision", "32"))
        self.amp_dtype = _AMP_DTYPE.get(precision)
        self.use_amp = self.amp_dtype is not None and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype is torch.float16)
        if self.is_main:
            logger.info("  precision: %s (amp=%s)", precision, self.use_amp)

        # ── loss (mirrors main_lite_ecg.py:92-139) ──
        self.task_type = str(self.cfg.get("task_type", "binary"))
        self.num_cls = int(self.cfg.get("num_cls", 0))
        self.num_reg = int(self.cfg.get("num_reg", 0))
        self.report_groups = self.cfg.get("report_groups")
        self.cls_cols = self.cfg.get("cls_cols")
        self.reg_cols = self.cfg.get("reg_cols")
        if self.task_type == "regression":
            self._loss_fn = self._regression_loss           # F.l1_loss
        elif self.task_type == "multi-label-binary":
            self._loss_fn = self._masked_bce_loss           # NaN-masked BCE
        elif self.task_type == "classification_and_regression":
            if self.num_cls <= 0 or self.num_reg <= 0:
                raise ValueError(
                    "classification_and_regression requires num_cls and num_reg in trainer cfg")
            self._loss_fn = self._joint_loss
        else:
            self._loss_fn = nn.BCEWithLogitsLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.save_dir = Path(self.cfg.get("save_dir", "./results"))
        if self.is_main:
            os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -float("inf")
        self.best_epoch = -1

        self.label_names = self.cfg.get("label_names")
        self.bootstrap_iterations = int(self.cfg.get("bootstrap_iterations", 1000))
        self.export_predictions = bool(self.cfg.get("export_predictions", False))

    # ------------------------------------------------------------------
    @property
    def _primary_key(self):
        if self.task_type == "regression":
            return "neg_mae_macro"
        if self.task_type == "classification_and_regression":
            return "neg_composite_score"
        return "auroc_macro"

    @property
    def _primary_label(self):
        return {"neg_mae_macro": "neg_mae",
                "neg_composite_score": "neg_composite"}.get(self._primary_key, "auroc")

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=self.amp_dtype)

    # ------------------------------------------------------------------
    def train(self):
        if self.is_main:
            logger.info(f"Training for {self.epochs} epochs on {self.device}")
            logger.info(f"  eval_mode: {self.model_unwrapped.eval_mode}")
            logger.info(f"  world_size: {self.world_size}")
            logger.info(f"  train: {len(self.train_loader.dataset):,} | "
                        f"val: {len(self.val_loader.dataset):,}")

        for epoch in range(self.epochs):
            if self.use_ddp and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

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

            if self.is_main:
                primary = val_metrics.get(self._primary_key, 0)
                lr_now = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | loss={train_loss:.4f} | "
                    f"val_{self._primary_label}={primary:.4f} | lr={lr_now:.2e}")

                if primary > self.best_metric:
                    self.best_metric = primary
                    self.best_epoch = epoch + 1
                    torch.save(self.model_unwrapped.state_dict(), self.save_dir / "best.pt")
                    # snapshot the val metrics of the *selected* epoch; val_metrics.txt
                    # keeps being overwritten and ends up holding the last epoch.
                    self._write_metrics(val_metrics, "val_best")

        if self.is_main:
            logger.info(f"Best val {self._primary_label}: {self.best_metric:.4f} "
                        f"at epoch {self.best_epoch}")

        # ── Test on the best checkpoint (original: trainer.test(ckpt_path="best")) ──
        if self.test_loader is not None:
            ckpt = self.save_dir / "best.pt"
            if ckpt.exists():
                state = torch.load(ckpt, weights_only=True, map_location="cpu")
                self.model_unwrapped.load_state_dict(state)
                self.model_unwrapped.to(self.device)
            test_metrics = self._eval_epoch(self.test_loader, "test", bootstrap=True)
            if self.is_main:
                logger.info(f"Test {self._primary_label}: "
                            f"{test_metrics.get(self._primary_key, 0):.4f}")
                test_metrics = dict(test_metrics)
                test_metrics.setdefault("best_val", self.best_metric)
                test_metrics.setdefault("best_epoch", self.best_epoch)
                return test_metrics

        return {"best_val": self.best_metric, "best_epoch": self.best_epoch}

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, n_batches = 0.0, 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch+1}",
                    leave=False, disable=not self.is_main)
        for batch in pbar:
            signal = batch["signal"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            # moryecg adapters consume these cache keys; other encoders declare
            # **_unused and the wrapper filters them out.
            extra = {k: batch[k] for k in ("ecg_filepath", "ecg_seg_idx") if k in batch}

            with self._autocast():
                logits = self.model(signal, **extra)
                loss = self._loss_fn(logits.float(), label)

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if self.use_ddp:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        return avg_loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_epoch(self, loader, prefix="val", bootstrap=False):
        """Evaluate a split, reporting both non-aggregated and per-record metrics.

        Windows of the same record share an ``ecg_id``; predictions are averaged
        over them (``aggregate_fn=np.mean``), reproducing
        ``TimeSeriesDataset.aggregate_predictions``.
        """
        self.model.eval()
        all_preds, all_targets, all_ids = [], [], []

        pbar = tqdm(loader, desc=f"Eval {prefix}", leave=False, disable=not self.is_main)
        for batch in pbar:
            signal = batch["signal"].to(self.device, non_blocking=True)
            label = batch["label"]
            extra = {k: batch[k] for k in ("ecg_filepath", "ecg_seg_idx") if k in batch}

            with self._autocast():
                logits = self.model(signal, **extra)
            logits = logits.float()

            if self.task_type == "regression":
                preds = logits.cpu().numpy()
            elif self.task_type == "classification_and_regression":
                cls = torch.sigmoid(logits[:, :self.num_cls])
                preds = torch.cat([cls, logits[:, self.num_cls:]], dim=1).cpu().numpy()
            else:
                preds = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(label.numpy())
            ids = batch.get("ecg_id")
            if ids is not None:
                all_ids.append(ids.numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids))

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_ids = np.concatenate(all_ids, axis=0) if all_ids else None

        if self.use_ddp:
            all_preds, all_targets, all_ids = self._gather_predictions(
                all_preds, all_targets, all_ids)

        metrics = {}
        if self.is_main:
            metrics = self._score(all_preds, all_targets, all_ids, prefix, bootstrap)

        if self.use_ddp:
            primary = torch.tensor(float(metrics.get(self._primary_key, 0.0)), device=self.device)
            dist.broadcast(primary, src=0)
            if not self.is_main:
                metrics = {self._primary_key: primary.item()}

        return metrics

    def _score(self, preds, targets, ids, prefix, bootstrap):
        """Compute noagg + agg metrics, persist them, optionally export predictions."""
        kwargs = dict(
            label_names=self.label_names,
            task_type=self.task_type,
            num_cls=self.num_cls if self.task_type == "classification_and_regression" else None,
            cls_cols=self.cls_cols,
            reg_cols=self.reg_cols,
            report_groups=self.report_groups,
        )

        noagg = evaluate_all(targets, preds, **kwargs)

        preds_agg, targets_agg = _aggregate_predictions(preds, targets, ids)
        agg = evaluate_all(targets_agg, preds_agg, **kwargs)

        # Model selection and all headline numbers use the aggregated metrics,
        # matching ModelCheckpoint(monitor="macro_auc_agg_val0") in the original.
        metrics = dict(agg)
        metrics.update({f"noagg_{k}": v for k, v in noagg.items()})
        metrics["n_windows"] = int(len(preds))
        metrics["n_records"] = int(len(preds_agg))

        if bootstrap and self.bootstrap_iterations > 0:
            logger.info("  bootstrap CIs (%d iterations) on %s ...",
                        self.bootstrap_iterations, prefix)
            cis = bootstrap_metrics(
                targets_agg, preds_agg,
                label_names=self.label_names, task_type=self.task_type,
                num_cls=self.num_cls if self.task_type == "classification_and_regression" else None,
                cls_cols=self.cls_cols, reg_cols=self.reg_cols,
                n_iterations=self.bootstrap_iterations)
            for key, (point, low, high) in cis.items():
                metrics[f"{key}_low"] = low
                metrics[f"{key}_high"] = high
                metrics[f"{key}_boot"] = point

        self._write_metrics(metrics, prefix)

        if self.export_predictions and prefix == "test":
            out = Path(self.cfg.get("prediction_dir") or self.save_dir) / "predictions"
            out.mkdir(parents=True, exist_ok=True)
            names = np.array(self.label_names if self.label_names is not None else [])
            np.savez(out / "test_noagg.npz", preds=preds, targs=targets,
                     ids=ids if ids is not None else np.arange(len(preds)), lbl_itos=names)
            np.savez(out / "test_agg.npz", preds=preds_agg, targs=targets_agg, lbl_itos=names)
            logger.info("  predictions exported to %s", out)

        return metrics

    def _write_metrics(self, metrics, prefix):
        with open(self.save_dir / f"{prefix}_metrics.txt", "w") as f:
            for k, v in sorted(metrics.items()):
                f.write(f"{k}: {v}\n")

    # ------------------------------------------------------------------
    def _gather_predictions(self, preds, targets, ids=None):
        """Gather every rank's predictions (+ ecg_ids) onto rank 0."""
        preds_t = torch.from_numpy(preds).to(self.device)
        targets_t = torch.from_numpy(targets).to(self.device)
        ids_t = torch.from_numpy(ids).to(self.device).long() if ids is not None else None

        local_size = torch.tensor(preds_t.shape[0], device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)

        if preds_t.shape[0] < max_size:
            pad = max_size - preds_t.shape[0]
            preds_t = torch.cat([preds_t, torch.zeros(pad, *preds_t.shape[1:], device=self.device)])
            targets_t = torch.cat([targets_t, torch.zeros(pad, *targets_t.shape[1:], device=self.device)])
            if ids_t is not None:
                ids_t = torch.cat([ids_t, torch.full((pad,), -1, device=self.device, dtype=ids_t.dtype)])

        gathered_preds = [torch.zeros_like(preds_t) for _ in range(self.world_size)]
        gathered_targets = [torch.zeros_like(targets_t) for _ in range(self.world_size)]
        dist.all_gather(gathered_preds, preds_t)
        dist.all_gather(gathered_targets, targets_t)
        if ids_t is not None:
            gathered_ids = [torch.zeros_like(ids_t) for _ in range(self.world_size)]
            dist.all_gather(gathered_ids, ids_t)

        if self.is_main:
            all_p, all_t, all_i = [], [], []
            for i, size in enumerate(all_sizes):
                n = size.item()
                all_p.append(gathered_preds[i][:n])
                all_t.append(gathered_targets[i][:n])
                if ids_t is not None:
                    all_i.append(gathered_ids[i][:n])
            return (torch.cat(all_p).cpu().numpy(),
                    torch.cat(all_t).cpu().numpy(),
                    torch.cat(all_i).cpu().numpy() if all_i else None)

        return preds, targets, ids

    # ── Loss helpers (reproduce main_lite_ecg.py:92-139) ─────────────────
    @staticmethod
    def _masked_bce_loss(logits, targets):
        """NaN-masked multi-label BCE (mds_ed missing-label handling)."""
        import torch.nn.functional as F

        valid = ~torch.isnan(targets)
        if not valid.any():
            return torch.zeros((), device=logits.device, requires_grad=True)
        targets_safe = torch.where(valid, targets, torch.zeros_like(targets))
        loss = F.binary_cross_entropy_with_logits(logits, targets_safe, reduction="none")
        return (loss * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    @staticmethod
    def _regression_loss(logits, targets):
        """NaN-masked L1 (the original uses F.l1_loss)."""
        import torch.nn.functional as F

        valid = ~torch.isnan(targets)
        if not valid.any():
            return torch.zeros((), device=logits.device, requires_grad=True)
        targets_safe = torch.where(valid, targets, torch.zeros_like(targets))
        loss = F.l1_loss(logits, targets_safe, reduction="none")
        return (loss * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    def _joint_loss(self, logits, targets):
        """BCE(cls) + L1(reg) with per-element NaN masking.

        The original averages *per-column* losses (each column's mean over its
        own valid entries, then a mean over columns); this reproduces that
        rather than pooling all valid elements together.
        """
        import torch.nn.functional as F

        nc = self.num_cls
        cls_logits, reg_logits = logits[:, :nc], logits[:, nc:]
        cls_targs, reg_targs = targets[:, :nc], targets[:, nc:]

        def per_column(pred, targ, fn):
            valid = ~torch.isnan(targ)
            targ_safe = torch.where(valid, targ, torch.zeros_like(targ))
            elem = fn(pred, targ_safe, reduction="none") * valid.float()
            counts = valid.float().sum(dim=0)
            keep = counts > 0
            if not bool(keep.any()):
                return torch.zeros((), device=logits.device)
            col_loss = elem.sum(dim=0)[keep] / counts[keep]
            return col_loss.mean()

        cls_loss = per_column(cls_logits, cls_targs, F.binary_cross_entropy_with_logits)
        reg_loss = per_column(reg_logits, reg_targs, F.l1_loss)
        return cls_loss + reg_loss


def _aggregate_predictions(preds, targets, ids, aggregate_fn=np.mean):
    """Average predictions per record id (``TimeSeriesDataset.aggregate_predictions``)."""
    if ids is None or len(ids) == 0 or len(ids) == len(np.unique(ids)):
        return preds, targets

    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    preds_sorted = preds[order]
    targets_sorted = targets[order]

    unique, starts = np.unique(ids_sorted, return_index=True)
    bounds = np.append(starts, len(ids_sorted))

    agg_preds = np.empty((len(unique), preds.shape[1]), dtype=preds.dtype)
    agg_targets = np.empty((len(unique), targets.shape[1]), dtype=targets.dtype)
    for i in range(len(unique)):
        s, e = bounds[i], bounds[i + 1]
        agg_preds[i] = aggregate_fn(preds_sorted[s:e], axis=0)
        agg_targets[i] = targets_sorted[s]      # all windows of a record share the label
    return agg_preds, agg_targets
