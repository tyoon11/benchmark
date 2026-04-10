"""
평가 메트릭
===========
AUROC, AUPRC, F1, Accuracy 등 다운스트림 태스크 평가 메트릭.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score,
)


def compute_auroc(targets, preds, average="macro"):
    """
    Multi-label AUROC.

    Args:
        targets: (N, C) binary
        preds:   (N, C) logits or probabilities
        average: 'macro', 'micro', 'weighted', None(=per-class)
    """
    try:
        if average is None:
            # Per-class AUROC
            n_classes = targets.shape[1]
            aucs = []
            for i in range(n_classes):
                if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                    aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
                else:
                    aucs.append(float("nan"))
            return np.array(aucs)
        return roc_auc_score(targets, preds, average=average)
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


def evaluate_all(targets, preds, label_names=None):
    """
    전체 메트릭 계산.

    Returns:
        dict with: auroc_macro, auroc_micro, auprc_macro, f1_macro,
                   per-class AUROC (if label_names provided)
    """
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
