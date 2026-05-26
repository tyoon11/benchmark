"""
 
===========
AUROC, AUPRC, F1, Accuracy (binary/multi-label) +
MAE, MSE, R² (regression). NaN masking support.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


def compute_auroc(targets, preds, average="macro"):
    """
    Multi-label AUROC.
    positive/negative all re- do class only as compute.

    Args:
        targets: (N, C) binary
        preds:   (N, C) logits or probabilities
        average: 'macro', 'micro', 'weighted', None(=per-class)
    """
    try:
        n_classes = targets.shape[1]
        per_class = []
        for i in range(n_classes):
            pos = targets[:, i].sum()
            if 0 < pos < len(targets):
                per_class.append(roc_auc_score(targets[:, i], preds[:, i]))
            else:
                per_class.append(float("nan"))

        if average is None:
            return np.array(per_class)

        valid = [v for v in per_class if not np.isnan(v)]
        if not valid:
            return float("nan")

        if average == "macro":
            return float(np.mean(valid))
        elif average == "micro":
            valid_idx = [i for i in range(n_classes)
                         if 0 < targets[:, i].sum() < len(targets)]
            if not valid_idx:
                return float("nan")
            return roc_auc_score(
                targets[:, valid_idx].ravel(), preds[:, valid_idx].ravel()
            )
        return float(np.mean(valid))
    except ValueError:
        return float("nan")


def compute_auprc(targets, preds, average="macro"):
    """Multi-label AUPRC."""
    try:
        if average is None:
            n_classes = targets.shape[1]
            aucs = []
            for i in range(n_classes):
                if targets[:, i].sum() > 0:
                    aucs.append(average_precision_score(targets[:, i], preds[:, i]))
                else:
                    aucs.append(float("nan"))
            return np.array(aucs)
        return average_precision_score(targets, preds, average=average)
    except ValueError:
        return float("nan")


def compute_f1(targets, preds, threshold=0.5, average="macro"):
    """Multi-label F1 (threshold based)."""
    preds_bin = (preds >= threshold).astype(int)
    try:
        return f1_score(targets, preds_bin, average=average, zero_division=0)
    except ValueError:
        return float("nan")


def compute_accuracy(targets, preds):
    """Multi-class accuracy (argmax based)."""
    targ_idx = np.argmax(targets, axis=-1)
    pred_idx = np.argmax(preds, axis=-1)
    return accuracy_score(targ_idx, pred_idx)


def compute_regression_metrics(targets, preds, label_names=None):
    """multivariate regression metrics — NaN masking (per-target).

    Returns:
        dict with: mae_macro, mse_macro, rmse_macro, r2_macro,
                   neg_mae_macro (best_metric (for) —  ),
                   per-target MAE/MSE/R²
    """
    n_targets = targets.shape[1] if targets.ndim > 1 else 1
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
        preds = preds.reshape(-1, 1)

    per_mae, per_mse, per_r2 = [], [], []
    for i in range(n_targets):
        valid = ~np.isnan(targets[:, i])
        if valid.sum() < 2:
            per_mae.append(float("nan"))
            per_mse.append(float("nan"))
            per_r2.append(float("nan"))
            continue
        t = targets[valid, i]
        p = preds[valid, i]
        per_mae.append(float(mean_absolute_error(t, p)))
        per_mse.append(float(mean_squared_error(t, p)))
        try:
            per_r2.append(float(r2_score(t, p)))
        except ValueError:
            per_r2.append(float("nan"))

    def _macro(values):
        valid = [v for v in values if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    mae_macro = _macro(per_mae)
    mse_macro = _macro(per_mse)
    rmse_macro = float(np.sqrt(mse_macro)) if not np.isnan(mse_macro) else float("nan")

    results = {
        "mae_macro":     mae_macro,
        "mse_macro":     mse_macro,
        "rmse_macro":    rmse_macro,
        "r2_macro":      _macro(per_r2),
        "neg_mae_macro": -mae_macro if not np.isnan(mae_macro) else -float("inf"),
    }
    if label_names is not None:
        for name, mae, mse, r2 in zip(label_names, per_mae, per_mse, per_r2):
            results[f"mae_{name}"] = mae
            results[f"mse_{name}"] = mse
            results[f"r2_{name}"]  = r2
    return results


def compute_joint_metrics(targets, preds, num_cls, cls_cols, reg_cols,
                           report_groups=None):
    """Paper-style metrics for the joint MIMIC task (cls + reg).

    Args:
        targets: (N, num_cls + num_reg) — cls part binary/NaN, reg part z-normed/NaN
        preds:   (N, num_cls + num_reg) — cls part sigmoid'd, reg part raw
        num_cls: number of leading classification columns
        cls_cols: list of cls column names (len = num_cls)
        reg_cols: list of reg column names (len = num_reg)
        report_groups: dict {group_name: {"kind": "cls"|"reg", "cols": [...]}} for
                       per-MIMIC-task sub-metrics (paper Table 3/4/7/8 rows).

    Returns dict with:
        auroc_macro_cls, auprc_macro_cls, f1_macro_cls   (over all cls cols)
        mae_macro_reg, mae_global_reg                    (over all reg cols, z-norm units)
        composite_score = (1 - auroc_macro_cls) + mae_global_reg  (paper, lower=better)
        neg_composite_score                              (for best-checkpoint maximize)
        per-group: <group>_auroc_macro / <group>_mae_macro / <group>_mae_global
        per-class AUROC for cls cols, per-target MAE for reg cols
    """
    cls_targets = targets[:, :num_cls]
    cls_preds = preds[:, :num_cls]
    reg_targets = targets[:, num_cls:]
    reg_preds = preds[:, num_cls:]

    # ── cls metrics (per-column AUROC, ignore NaN rows per-column) ──
    cls_targets_safe = cls_targets.copy()
    # Drop rows that are entirely NaN across cls (no information)
    all_nan_rows = np.isnan(cls_targets_safe).all(axis=1)
    if all_nan_rows.any():
        cls_targets_safe = cls_targets_safe[~all_nan_rows]
        cls_preds_safe = cls_preds[~all_nan_rows]
    else:
        cls_preds_safe = cls_preds
    # AUROC computed per column with NaN→0 (paper behavior for multi-label-binary)
    cls_targets_safe = np.nan_to_num(cls_targets_safe, nan=0.0)

    per_class_auroc = compute_auroc(cls_targets_safe, cls_preds_safe, average=None)
    if isinstance(per_class_auroc, float):  # ValueError fallback
        per_class_auroc = np.full(num_cls, float("nan"))
    valid_auroc = per_class_auroc[~np.isnan(per_class_auroc)]
    auroc_macro = float(valid_auroc.mean()) if len(valid_auroc) else float("nan")

    auprc_macro = compute_auprc(cls_targets_safe, cls_preds_safe, "macro")
    f1_macro = compute_f1(cls_targets_safe, cls_preds_safe, average="macro")

    # ── reg metrics (NaN-masked per column) ──
    reg_metrics = compute_regression_metrics(reg_targets, reg_preds, reg_cols)
    mae_macro_reg = reg_metrics.get("mae_macro", float("nan"))

    # Paper main_lite_base.py uses ravel-then-mean_absolute_error as global MAE
    flat_targs = reg_targets.ravel()
    flat_preds = reg_preds.ravel()
    valid = ~np.isnan(flat_targs)
    if valid.any():
        from sklearn.metrics import mean_absolute_error
        mae_global = float(mean_absolute_error(flat_targs[valid], flat_preds[valid]))
    else:
        mae_global = float("nan")

    composite = (1.0 - auroc_macro) + mae_global if not (
        np.isnan(auroc_macro) or np.isnan(mae_global)) else float("nan")

    out = {
        "auroc_macro_cls": auroc_macro,
        "auprc_macro_cls": auprc_macro,
        "f1_macro_cls":    f1_macro,
        "mae_macro_reg":   mae_macro_reg,
        "mae_global_reg":  mae_global,
        "composite_score": composite,
        "neg_composite_score": -composite if not np.isnan(composite) else -float("inf"),
        # convenience aliases so trainer/run.py shared paths still work
        "auroc_macro":  auroc_macro,
        "mae_macro":    mae_macro_reg,
        "neg_mae_macro": -mae_macro_reg if not np.isnan(mae_macro_reg) else -float("inf"),
    }

    # per-class AUROC + per-target MAE (full breakdown)
    if cls_cols is not None:
        for name, auc in zip(cls_cols, per_class_auroc):
            out[f"auroc_{name}"] = float(auc) if not np.isnan(auc) else float("nan")
    for name in reg_cols or []:
        for prefix in ("mae", "mse", "r2"):
            k = f"{prefix}_{name}"
            if k in reg_metrics:
                out[k] = reg_metrics[k]

    # ── per-MIMIC-task sub-group metrics (paper Tables 3/4/7/8) ──
    if report_groups:
        cls_idx = {c: i for i, c in enumerate(cls_cols or [])}
        reg_idx = {c: i for i, c in enumerate(reg_cols or [])}
        for gname, gdef in report_groups.items():
            kind = gdef.get("kind")
            cols = gdef.get("cols", [])
            if kind == "cls":
                idxs = [cls_idx[c] for c in cols if c in cls_idx]
                if not idxs:
                    continue
                sub_t = cls_targets_safe[:, idxs]
                sub_p = cls_preds_safe[:, idxs]
                aucs = compute_auroc(sub_t, sub_p, average=None)
                if isinstance(aucs, float):
                    aucs = np.full(len(idxs), float("nan"))
                valid = aucs[~np.isnan(aucs)]
                out[f"{gname}_auroc_macro"] = float(valid.mean()) if len(valid) else float("nan")
                out[f"{gname}_auprc_macro"] = compute_auprc(sub_t, sub_p, "macro")
                out[f"{gname}_f1_macro"]    = compute_f1(sub_t, sub_p, average="macro")
            elif kind == "reg":
                idxs = [reg_idx[c] for c in cols if c in reg_idx]
                if not idxs:
                    continue
                sub_t = reg_targets[:, idxs]
                sub_p = reg_preds[:, idxs]
                sub_m = compute_regression_metrics(sub_t, sub_p, [reg_cols[i] for i in idxs])
                out[f"{gname}_mae_macro"]  = sub_m.get("mae_macro", float("nan"))
                out[f"{gname}_mse_macro"]  = sub_m.get("mse_macro", float("nan"))
                out[f"{gname}_rmse_macro"] = sub_m.get("rmse_macro", float("nan"))
                out[f"{gname}_r2_macro"]   = sub_m.get("r2_macro", float("nan"))
                # global MAE (ravel) for this group — matches paper's per-task reporting
                flat_t = sub_t.ravel()
                flat_p = sub_p.ravel()
                v = ~np.isnan(flat_t)
                if v.any():
                    from sklearn.metrics import mean_absolute_error
                    out[f"{gname}_mae_global"] = float(mean_absolute_error(
                        flat_t[v], flat_p[v]))
                else:
                    out[f"{gname}_mae_global"] = float("nan")

    return out


def evaluate_all(targets, preds, label_names=None, task_type="binary",
                  num_cls=None, cls_cols=None, reg_cols=None, report_groups=None):
    """
    all  compute. task_type per .

    task_type:
        'binary'                       → AUROC/AUPRC/F1 (multi-label binary)
        'multi-label-binary'           → AUROC/AUPRC/F1 + NaN masking
        'regression'                   → MAE/MSE/RMSE/R² + neg_MAE (best_metric (for))
        'classification_and_regression'→ BCE+L1 joint (paper): macro_AUROC + global_MAE +
                                          composite_score + per-group metrics
    """
    if task_type == "classification_and_regression":
        return compute_joint_metrics(targets, preds, num_cls=num_cls,
                                       cls_cols=cls_cols, reg_cols=reg_cols,
                                       report_groups=report_groups)
    if task_type == "regression":
        return compute_regression_metrics(targets, preds, label_names)

    # binary / multi-label-binary
    if task_type == "multi-label-binary":
        # NaN location  compute from exclude (per-class)
        # compute_auroc already positive=0  NaN handling,
        # Here: NaN target 0(negative) as   masking
        targets = targets.copy()
        # NaN rows exclude — justorder above all NaN rows only remove
        all_nan = np.isnan(targets).all(axis=1)
        if all_nan.any():
            targets = targets[~all_nan]
            preds = preds[~all_nan]
        #  NaN 0 as (per-class AUROC positive  0 if so, automatic NaN handling)
        targets = np.nan_to_num(targets, nan=0.0)

    results = {
        "auroc_macro": compute_auroc(targets, preds, "macro"),
        "auroc_micro": compute_auroc(targets, preds, "micro"),
        "auprc_macro": compute_auprc(targets, preds, "macro"),
        "f1_macro":    compute_f1(targets, preds, average="macro"),
    }

    if label_names is not None:
        per_class = compute_auroc(targets, preds, average=None)
        for name, auc in zip(label_names, per_class):
            results[f"auroc_{name}"] = auc

    return results
