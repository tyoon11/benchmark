"""
Metrics
=======
Ports the original ``ecg-fm-benchmarking`` scoring functions verbatim so the
numbers are directly comparable to the published tables:

* :func:`multiclass_roc_curve` — ``clinical_ts/utils/eval_utils_cafa.py``.
  Note the macro definition: it is the mean of the per-class AUCs over **all**
  classes, with classes that cannot be scored (no positives or no negatives in
  the split) counted as **0.5**, not skipped. NaN targets are masked per column.
* :func:`regression_metrics` — ``clinical_ts/utils/eval_utils_regression.py``.
  ``mae`` is the *global* MAE over the raveled, NaN-masked matrix; per-target
  MAEs are reported alongside it.
* :func:`empirical_bootstrap` — ``clinical_ts/utils/bootstrap_utils.py``,
  used for the 95% CIs on the test split.

The previous sklearn-based ``auroc_macro`` (mean over scorable classes only)
is kept as ``auroc_macro_skipnan`` for comparison; ``auroc_macro`` is now the
paper definition, which is what every downstream script reads.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import (auc, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score,
                             roc_curve)
from sklearn.utils import resample


# ═══════════════════════════════════════════════════════════════════════════
# Paper-exact scoring
# ═══════════════════════════════════════════════════════════════════════════
def multiclass_roc_curve(y_true, y_pred, classes=None, precision_recall=False):
    """Per-class / micro / macro ROC-AUC, matching ``eval_utils_cafa``.

    Returns ``(fpr, tpr, roc_auc)`` dictionaries keyed by class name plus
    ``"micro"`` and ``"macro"``.
    """
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = len(y_pred[0])
    if classes is None:
        classes = [str(i) for i in range(n_classes)]

    for i, c in enumerate(classes):
        y_truei = y_true[:, i]
        y_predi = y_pred[:, i]

        maski = ~np.isnan(y_truei)          # mask nan targets if present
        y_truei = y_truei[maski]
        y_predi = y_predi[maski]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if precision_recall:
                tpr[c], fpr[c], _ = precision_recall_curve(y_truei, y_predi)
                roc_auc[c] = -np.sum(np.diff(fpr[c]) * np.array(tpr[c])[:-1])
            else:
                fpr[c], tpr[c], _ = roc_curve(y_truei, y_predi)
                roc_auc[c] = auc(fpr[c], tpr[c])

    # micro
    y_true_micro = y_true.ravel()
    y_pred_micro = y_pred.ravel()
    mask_micro = ~np.isnan(y_true_micro)
    y_true_micro = y_true_micro[mask_micro]
    y_pred_micro = y_pred_micro[mask_micro]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if precision_recall:
            tpr["micro"], fpr["micro"], _ = precision_recall_curve(y_true_micro, y_pred_micro)
            roc_auc["micro"] = -np.sum(np.diff(fpr["micro"]) * np.array(tpr["micro"])[:-1])
        else:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_micro, y_pred_micro)
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # macro ROC *curve* (only meaningful for ROC, not PR)
        if precision_recall is False:
            all_fpr = np.unique(np.concatenate([fpr[c] for c in classes]))
            mean_tpr = None
            for c in classes:
                if len(fpr[c]) < 2:
                    continue
                f = interp1d(fpr[c], tpr[c], bounds_error=False,
                             fill_value=(tpr[c][0], tpr[c][-1]))
                mean_tpr = f(all_fpr) if mean_tpr is None else mean_tpr + f(all_fpr)
            if mean_tpr is not None:
                mean_tpr = mean_tpr / n_classes
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr

    # macro AUC by direct averaging; unscoreable classes count as 0.5
    # (conservative choice, verbatim from the original)
    roc_auc_macro = 0.0
    for c in classes:
        roc_auc_macro += 0.5 if np.isnan(roc_auc[c]) else roc_auc[c]
    roc_auc["macro"] = roc_auc_macro / n_classes

    return fpr, tpr, roc_auc


def regression_metrics(y_true, y_pred, metrics=("mae",), target_names=None):
    """Per-target and global regression metrics, matching ``eval_utils_regression``."""
    metrics = list(metrics) if metrics is not None else ["mae", "r2"]
    n_targets = y_true.shape[1]
    results = {}

    if target_names is None:
        target_names = [str(i) for i in range(n_targets)]
    elif len(target_names) != n_targets:
        raise ValueError(f"target_names length {len(target_names)} != {n_targets} targets")

    for i in range(n_targets):
        mask = ~np.isnan(y_true[:, i])
        t, p = y_true[mask, i], y_pred[mask, i]
        for m in metrics:
            key = f"{target_names[i]}_{m}"
            if len(t) == 0:
                results[key] = np.nan
            elif m == "mae":
                results[key] = mean_absolute_error(t, p)
            elif m == "mse":
                results[key] = mean_squared_error(t, p)
            elif m == "r2":
                results[key] = r2_score(t, p) if len(t) > 1 else np.nan

    flat_t = y_true.ravel()
    flat_p = y_pred.ravel()
    mask = ~np.isnan(flat_t)
    t, p = flat_t[mask], flat_p[mask]
    for m in metrics:
        if len(t) == 0:
            results[m] = np.nan
        elif m == "mae":
            results[m] = mean_absolute_error(t, p)
        elif m == "mse":
            results[m] = mean_squared_error(t, p)
        elif m == "r2":
            results[m] = r2_score(t, p) if len(t) > 1 else np.nan
    return results


def mcrc_flat(targs, preds, classes):
    """Bootstrap-friendly wrapper: ordered array of the roc_auc dict values."""
    _, _, res = multiclass_roc_curve(targs, preds, classes=classes)
    return np.array(list(res.values()))


def regression_flat(targs, preds, metrics=("mae",), target_names=None):
    """Bootstrap-friendly wrapper mirroring ``main_lite_base.regression_flat``."""
    metrics = list(metrics)
    res = regression_metrics(targs, preds, metrics=metrics, target_names=target_names)
    if target_names is None:
        target_names = [str(i) for i in range(targs.shape[1])]
    ordered = [res[m] for m in metrics]
    for name in target_names:
        ordered.extend(res[f"{name}_{m}"] for m in metrics)
    return np.array(ordered)


def empirical_bootstrap(input_tuple, score_fn, ids=None, n_iterations=1000,
                        alpha=0.95, score_fn_kwargs=None, threads=0,
                        input_tuple2=None, ignore_nans=False):
    """Empirical bootstrap CIs, matching ``clinical_ts/utils/bootstrap_utils``.

    ``threads=0`` (single process) is the default here: the original spawns a
    multiprocessing Pool, which deadlocks when called from inside a training
    process that already holds DataLoader workers.
    """
    score_fn_kwargs = score_fn_kwargs or {}
    if not isinstance(input_tuple, tuple):
        input_tuple = (input_tuple,)
    if input_tuple2 is not None and not isinstance(input_tuple2, tuple):
        input_tuple2 = (input_tuple2,)

    def _score(tup):
        return np.asarray(score_fn(*tup, **score_fn_kwargs), dtype=np.float64)

    score_point = _score(input_tuple)
    if input_tuple2 is not None:
        score_point = score_point - _score(input_tuple2)

    if n_iterations == 0:
        zeros = np.zeros_like(score_point)
        return score_point, zeros, zeros, []

    n = len(input_tuple[0])
    if ids is None:
        rng = np.random.RandomState(0)
        ids = np.array([resample(range(n), n_samples=n, random_state=rng.randint(0, 2 ** 31 - 1))
                        for _ in range(n_iterations)])

    results = []
    for sample_ids in ids:
        value = _score(tuple(t[sample_ids] for t in input_tuple))
        if input_tuple2 is not None:
            value = value - _score(tuple(t[sample_ids] for t in input_tuple2))
        results.append(value)
    results = np.asarray(results, dtype=np.float64)

    percentile_fn = np.nanpercentile if ignore_nans else np.percentile
    score_diff = results - score_point
    score_low = score_point + percentile_fn(score_diff, ((1.0 - alpha) / 2.0) * 100, axis=0)
    score_high = score_point + percentile_fn(score_diff, (alpha + ((1.0 - alpha) / 2.0)) * 100, axis=0)

    if ignore_nans:
        return score_point, score_low, score_high, np.sum(np.isnan(score_diff), axis=0)
    return score_point, score_low, score_high, ids


# ═══════════════════════════════════════════════════════════════════════════
# sklearn-based helpers (kept for the *_skipnan diagnostics)
# ═══════════════════════════════════════════════════════════════════════════
def compute_auroc(targets, preds, average="macro"):
    """Per-class AUROC skipping unscoreable classes (diagnostic, not the paper macro)."""
    try:
        n_classes = targets.shape[1]
        per_class = []
        for i in range(n_classes):
            mask = ~np.isnan(targets[:, i])
            t = targets[mask, i]
            pos = t.sum()
            if 0 < pos < len(t):
                per_class.append(roc_auc_score(t, preds[mask, i]))
            else:
                per_class.append(float("nan"))

        if average is None:
            return np.array(per_class)

        valid = [v for v in per_class if not np.isnan(v)]
        if not valid:
            return float("nan")
        if average == "micro":
            valid_idx = [i for i in range(n_classes) if not np.isnan(per_class[i])]
            t = targets[:, valid_idx].ravel()
            p = preds[:, valid_idx].ravel()
            mask = ~np.isnan(t)
            return float(roc_auc_score(t[mask], p[mask]))
        return float(np.mean(valid))
    except ValueError:
        return float("nan")


def compute_auprc(targets, preds, average="macro"):
    """Multi-label AUPRC (NaN targets treated as negatives by sklearn)."""
    try:
        safe = np.nan_to_num(targets, nan=0.0)
        if average is None:
            aucs = []
            for i in range(safe.shape[1]):
                aucs.append(average_precision_score(safe[:, i], preds[:, i])
                            if safe[:, i].sum() > 0 else float("nan"))
            return np.array(aucs)
        return float(average_precision_score(safe, preds, average=average))
    except ValueError:
        return float("nan")


def compute_f1(targets, preds, threshold=0.5, average="macro"):
    """Multi-label F1 at a fixed threshold."""
    try:
        safe = np.nan_to_num(targets, nan=0.0)
        return float(f1_score(safe, (preds >= threshold).astype(int),
                              average=average, zero_division=0))
    except ValueError:
        return float("nan")


def compute_regression_metrics(targets, preds, label_names=None):
    """Regression summary in this repo's key naming, built on :func:`regression_metrics`."""
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
    names = list(label_names) if label_names is not None else [str(i) for i in range(targets.shape[1])]

    res = regression_metrics(targets, preds, metrics=("mae", "mse", "r2"), target_names=names)

    def _macro(suffix):
        vals = [res[f"{n}_{suffix}"] for n in names]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    mae_macro = _macro("mae")
    mse_macro = _macro("mse")
    out = {
        "mae_macro": mae_macro,
        "mae_global": float(res["mae"]),          # paper's "mae"
        "mse_macro": mse_macro,
        "rmse_macro": float(np.sqrt(mse_macro)) if not np.isnan(mse_macro) else float("nan"),
        "r2_macro": _macro("r2"),
        "r2_global": float(res["r2"]),
        "neg_mae_macro": -mae_macro if not np.isnan(mae_macro) else -float("inf"),
        "neg_mae_global": -float(res["mae"]) if not np.isnan(res["mae"]) else -float("inf"),
    }
    if label_names is not None:
        for n in names:
            out[f"mae_{n}"] = res[f"{n}_mae"]
            out[f"mse_{n}"] = res[f"{n}_mse"]
            out[f"r2_{n}"] = res[f"{n}_r2"]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Task-level entry points
# ═══════════════════════════════════════════════════════════════════════════
def _classification_metrics(targets, preds, label_names):
    """Paper macro/micro/per-class AUC plus sklearn diagnostics."""
    n_classes = targets.shape[1]
    classes = list(label_names) if label_names is not None and len(label_names) == n_classes \
        else [str(i) for i in range(n_classes)]

    _, _, roc = multiclass_roc_curve(targets, preds, classes=classes)

    results = {
        "auroc_macro": float(roc["macro"]),          # paper definition
        "auroc_micro": float(roc["micro"]),
        "auroc_macro_skipnan": compute_auroc(targets, preds, "macro"),
        "auprc_macro": compute_auprc(targets, preds, "macro"),
        "f1_macro": compute_f1(targets, preds, average="macro"),
        "n_classes_unscoreable": int(sum(np.isnan(roc[c]) for c in classes)),
    }
    for c in classes:
        results[f"auroc_{c}"] = float(roc[c]) if not np.isnan(roc[c]) else float("nan")
    return results, classes


def compute_joint_metrics(targets, preds, num_cls, cls_cols, reg_cols,
                          report_groups=None):
    """Joint MIMIC task (cls + reg), matching ``main_lite_base.eval_scores``.

    ``composite_score = (1 - macro_AUROC_cls) + global_MAE_reg`` — lower is
    better, so ``neg_composite_score`` is what the trainer maximises.
    """
    cls_targets, cls_preds = targets[:, :num_cls], preds[:, :num_cls]
    reg_targets, reg_preds = targets[:, num_cls:], preds[:, num_cls:]

    cls_names = list(cls_cols) if cls_cols else [str(i) for i in range(num_cls)]
    reg_names = list(reg_cols) if reg_cols else [str(i) for i in range(reg_targets.shape[1])]

    _, _, cls_roc = multiclass_roc_curve(cls_targets, cls_preds, classes=cls_names)
    auroc_macro = float(cls_roc["macro"])

    reg_res = regression_metrics(reg_targets, reg_preds, metrics=("mae",), target_names=reg_names)
    mae_global = float(reg_res["mae"])

    composite = (1.0 - auroc_macro) + mae_global
    reg_summary = compute_regression_metrics(reg_targets, reg_preds, reg_names)

    out = {
        "auroc_macro_cls": auroc_macro,
        "auroc_micro_cls": float(cls_roc["micro"]),
        "auprc_macro_cls": compute_auprc(cls_targets, cls_preds, "macro"),
        "f1_macro_cls": compute_f1(cls_targets, cls_preds, average="macro"),
        "mae_macro_reg": reg_summary["mae_macro"],
        "mae_global_reg": mae_global,
        "composite_score": composite,
        "neg_composite_score": -composite,
        # aliases so the shared trainer/summary paths keep working
        "auroc_macro": auroc_macro,
        "mae_macro": reg_summary["mae_macro"],
        "neg_mae_macro": reg_summary["neg_mae_macro"],
    }
    for name in cls_names:
        out[f"auroc_{name}"] = float(cls_roc[name]) if not np.isnan(cls_roc[name]) else float("nan")
    for name in reg_names:
        for prefix in ("mae", "mse", "r2"):
            out[f"{prefix}_{name}"] = reg_summary.get(f"{prefix}_{name}", float("nan"))

    if report_groups:
        cls_idx = {c: i for i, c in enumerate(cls_names)}
        reg_idx = {c: i for i, c in enumerate(reg_names)}
        for gname, gdef in report_groups.items():
            kind, cols = gdef.get("kind"), gdef.get("cols", [])
            if kind == "cls":
                idxs = [cls_idx[c] for c in cols if c in cls_idx]
                if not idxs:
                    continue
                sub_t, sub_p = cls_targets[:, idxs], cls_preds[:, idxs]
                sub_names = [cls_names[i] for i in idxs]
                _, _, sub_roc = multiclass_roc_curve(sub_t, sub_p, classes=sub_names)
                out[f"{gname}_auroc_macro"] = float(sub_roc["macro"])
                out[f"{gname}_auprc_macro"] = compute_auprc(sub_t, sub_p, "macro")
                out[f"{gname}_f1_macro"] = compute_f1(sub_t, sub_p, average="macro")
            elif kind == "reg":
                idxs = [reg_idx[c] for c in cols if c in reg_idx]
                if not idxs:
                    continue
                sub_t, sub_p = reg_targets[:, idxs], reg_preds[:, idxs]
                sub = compute_regression_metrics(sub_t, sub_p, [reg_names[i] for i in idxs])
                out[f"{gname}_mae_macro"] = sub["mae_macro"]
                out[f"{gname}_mae_global"] = sub["mae_global"]
                out[f"{gname}_mse_macro"] = sub["mse_macro"]
                out[f"{gname}_rmse_macro"] = sub["rmse_macro"]
                out[f"{gname}_r2_macro"] = sub["r2_macro"]
    return out


def evaluate_all(targets, preds, label_names=None, task_type="binary",
                 num_cls=None, cls_cols=None, reg_cols=None, report_groups=None):
    """Dispatch to the right metric set for ``task_type``.

    ``binary`` / ``multi-label-binary`` -> AUROC (paper macro/micro) + AUPRC + F1
    ``regression``                      -> MAE/MSE/RMSE/R2 (+ neg_MAE for selection)
    ``classification_and_regression``   -> composite score, see :func:`compute_joint_metrics`
    """
    if task_type == "classification_and_regression":
        return compute_joint_metrics(targets, preds, num_cls=num_cls,
                                     cls_cols=cls_cols, reg_cols=reg_cols,
                                     report_groups=report_groups)
    if task_type == "regression":
        return compute_regression_metrics(targets, preds, label_names)

    results, _ = _classification_metrics(targets, preds, label_names)
    return results


def bootstrap_metrics(targets, preds, label_names=None, task_type="binary",
                      num_cls=None, cls_cols=None, reg_cols=None,
                      n_iterations=1000):
    """95% empirical-bootstrap CIs for the headline metrics (test split only).

    Returns ``{metric: (point, low, high)}``; empty when ``n_iterations <= 0``.
    """
    if n_iterations <= 0:
        return {}

    if task_type == "regression":
        names = list(label_names) if label_names is not None else \
            [str(i) for i in range(targets.shape[1])]
        point, low, high, _ = empirical_bootstrap(
            (targets, preds), regression_flat, n_iterations=n_iterations,
            score_fn_kwargs={"metrics": ["mae"], "target_names": names})
        keys = ["mae_global"] + [f"mae_{n}" for n in names]
        return {k: (float(point[i]), float(low[i]), float(high[i]))
                for i, k in enumerate(keys)}

    if task_type == "classification_and_regression":
        cls_names = list(cls_cols)
        reg_names = list(reg_cols)
        cls_point, cls_low, cls_high, _ = empirical_bootstrap(
            (targets[:, :num_cls], preds[:, :num_cls]), mcrc_flat,
            n_iterations=n_iterations, score_fn_kwargs={"classes": cls_names})
        reg_point, reg_low, reg_high, _ = empirical_bootstrap(
            (targets[:, num_cls:], preds[:, num_cls:]), regression_flat,
            n_iterations=n_iterations,
            score_fn_kwargs={"metrics": ["mae"], "target_names": reg_names})
        cls_keys = [f"auroc_{n}" for n in cls_names] + ["auroc_micro_cls", "auroc_macro_cls"]
        reg_keys = ["mae_global_reg"] + [f"mae_{n}" for n in reg_names]
        out = {k: (float(cls_point[i]), float(cls_low[i]), float(cls_high[i]))
               for i, k in enumerate(cls_keys)}
        out.update({k: (float(reg_point[i]), float(reg_low[i]), float(reg_high[i]))
                    for i, k in enumerate(reg_keys)})
        composite = ((1.0 - out["auroc_macro_cls"][0]) + out["mae_global_reg"][0],
                     (1.0 - out["auroc_macro_cls"][1]) + out["mae_global_reg"][1],
                     (1.0 - out["auroc_macro_cls"][2]) + out["mae_global_reg"][2])
        out["composite_score"] = composite
        return out

    n_classes = targets.shape[1]
    classes = list(label_names) if label_names is not None and len(label_names) == n_classes \
        else [str(i) for i in range(n_classes)]
    point, low, high, _ = empirical_bootstrap(
        (targets, preds), mcrc_flat, n_iterations=n_iterations,
        score_fn_kwargs={"classes": classes})
    keys = [f"auroc_{c}" for c in classes] + ["auroc_micro", "auroc_macro"]
    return {k: (float(point[i]), float(low[i]), float(high[i]))
            for i, k in enumerate(keys)}


def compute_accuracy(targets, preds):
    """Multi-class accuracy (argmax based)."""
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(np.argmax(targets, axis=-1), np.argmax(preds, axis=-1)))
