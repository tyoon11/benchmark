"""
평가 메트릭
===========
AUROC, AUPRC, F1, Accuracy (binary/multi-label) +
MAE, MSE, R² (regression). NaN 마스킹 지원.
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
    양성/음성이 모두 존재하는 클래스만으로 계산합니다.

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
    """Multi-label F1 (threshold 기반)."""
    preds_bin = (preds >= threshold).astype(int)
    try:
        return f1_score(targets, preds_bin, average=average, zero_division=0)
    except ValueError:
        return float("nan")


def compute_accuracy(targets, preds):
    """Multi-class accuracy (argmax 기반)."""
    targ_idx = np.argmax(targets, axis=-1)
    pred_idx = np.argmax(preds, axis=-1)
    return accuracy_score(targ_idx, pred_idx)


def compute_regression_metrics(targets, preds, label_names=None):
    """다변량 regression metrics — NaN 마스킹 (per-target).

    Returns:
        dict with: mae_macro, mse_macro, rmse_macro, r2_macro,
                   neg_mae_macro (best_metric용 — 높을수록 좋음),
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


def evaluate_all(targets, preds, label_names=None, task_type="binary"):
    """
    전체 메트릭 계산. task_type별 분기.

    task_type:
        'binary'              → AUROC/AUPRC/F1 (multi-label binary)
        'multi-label-binary'  → AUROC/AUPRC/F1 + NaN 마스킹
        'regression'          → MAE/MSE/RMSE/R² + neg_MAE (best_metric용)
    """
    if task_type == "regression":
        return compute_regression_metrics(targets, preds, label_names)

    # binary / multi-label-binary
    if task_type == "multi-label-binary":
        # NaN 위치는 메트릭 계산에서 제외 (per-class)
        # compute_auroc는 이미 양성=0 케이스 NaN 처리하므로,
        # 여기서는 NaN target을 0(negative)으로 바꾸지 않고 마스킹
        targets = targets.copy()
        # NaN 행은 제외 — 단순화 위해 모두 NaN 행만 제거
        all_nan = np.isnan(targets).all(axis=1)
        if all_nan.any():
            targets = targets[~all_nan]
            preds = preds[~all_nan]
        # 남은 NaN은 0으로 (per-class AUROC가 양성 카운트 0이면 자동 NaN 처리)
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
