"""
Pipeline parity check
=====================
Verifies that the tensor reaching an encoder matches what the original
``ecg-fm-benchmarking`` pipeline would have produced, without needing the
original's memmap store.

What it checks, per (task, model):

1. **Window contract** — the batch is ``input_size x model_fs`` samples long,
   i.e. the adapter never has to resize internally.
2. **Lead order** — the permutation applied to the HEEDB channel order, plus a
   physiological sanity test: with standard ordering the limb leads must satisfy
   Einthoven (``II = I + III``) and the augmented leads
   ``aVR = -(I+II)/2``, ``aVL = (I-III)/2``, ``aVF = (II+III)/2``. Feeding the
   raw HEEDB order fails these, which is exactly what the pre-fix pipeline did.
3. **Window count** — number of val/test windows per record equals
   ``floor(record_seconds / input_size)``, matching the original's strided
   chunking, and train yields one window per record.
4. **Resampling** — the band-limited path is active (resampy present) and the
   spectrum of a decimated window has no energy above the new Nyquist, which
   linear interpolation would leave behind.

Run:
    python scripts/verify_parity.py --task ptbxl_super
    python scripts/verify_parity.py --task ptbxl_super --model ecg_founder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run import encoder_contract, load_config  # noqa: E402
from src.dataset import build_dataset  # noqa: E402
from src.leads import STANDARD_12  # noqa: E402
from src import signal_utils  # noqa: E402

# (input_size seconds, model_fs Hz) from the original run.sh
MODEL_CONTRACTS = {
    "ecg_founder": (2.5, 500),
    "ecg_jepa": (10.0, 250),
    "st_mem": (2.4, 250),
    "merl": (2.5, 500),
    "ecgfm_ked": (10.0, 500),
    "hubert_ecg": (5.0, 100),
    "cpc": (2.5, 240),
    "ecg_fm": (5.0, 500),
}

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def _check(results, name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))


def einthoven_report(sig, leads):
    """Max absolute residual of the standard 12-lead identities, in mV."""
    idx = {name: i for i, name in enumerate(leads)}
    need = ["I", "II", "III", "aVR", "aVL", "aVF"]
    if any(n not in idx for n in need):
        return None
    I, II, III = sig[idx["I"]], sig[idx["II"]], sig[idx["III"]]
    aVR, aVL, aVF = sig[idx["aVR"]], sig[idx["aVL"]], sig[idx["aVF"]]
    return {
        "II = I + III": float(np.abs(II - (I + III)).max()),
        "aVR = -(I+II)/2": float(np.abs(aVR + (I + II) / 2).max()),
        "aVL = (I-III)/2": float(np.abs(aVL - (I - III) / 2).max()),
        "aVF = (II+III)/2": float(np.abs(aVF - (II + III) / 2).max()),
    }


def run_model(task: str, model: str, n_samples: int) -> list:
    input_size, fs_model = MODEL_CONTRACTS[model]
    results = []

    cfg = load_config(task)
    data_cfg = cfg["data"]
    data_cfg["input_size"] = input_size
    data_cfg["fs_model"] = fs_model
    data_cfg["lead_order"] = "standard"
    data_cfg["fold_col"] = cfg.get("fold", {}).get("col", "strat_fold")

    # use the last two folds so the check runs on a small slice
    import pandas as pd

    table = pd.read_csv(data_cfg["table_csv"], usecols=[data_cfg["fold_col"]])
    mx = int(table[data_cfg["fold_col"]].max())
    data_cfg["test_folds"] = [mx]
    data_cfg["train_folds"] = [mx]

    expected_len = int(round(input_size * fs_model))

    ds_test = build_dataset(data_cfg, "test")
    ds_train = build_dataset(data_cfg, "train")

    # 1. window contract
    item = ds_test[0]
    sig = item["signal"].numpy()
    _check(results, "window length",
           sig.shape[-1] == expected_len,
           f"got {sig.shape[-1]}, expected {expected_len} ({input_size}s x {fs_model}Hz)")
    _check(results, "lead count", sig.shape[0] == 12, f"got {sig.shape[0]}")

    # 2. lead order / physiological identities
    residuals = einthoven_report(sig, STANDARD_12)
    if residuals is None:
        results.append((WARN, "lead identities", "fewer than 12 leads"))
    else:
        scale = float(np.abs(sig).max()) or 1.0
        worst = max(residuals.values())
        _check(results, "12-lead identities (standard order)",
               worst < 0.05 * scale,
               " | ".join(f"{k}: {v:.4f}" for k, v in residuals.items())
               + f"  (signal max |x| = {scale:.3f} mV)")

        # the same test on the *unpermuted* HEEDB order must fail — proof the
        # permutation is doing real work rather than being a no-op
        if ds_test.lead_perm is not None:
            inverse = np.argsort(ds_test.lead_perm)
            raw = sig[inverse]              # back to HEEDB order
            raw_res = einthoven_report(raw, STANDARD_12)
            raw_worst = max(raw_res.values())
            _check(results, "HEEDB order fails the identities (control)",
                   raw_worst > 0.05 * scale,
                   f"worst residual without the permutation: {raw_worst:.4f}")

    # 3. window counts
    ids = ds_test.get_id_mapping()
    per_record = np.bincount(ids)[np.unique(ids)]
    fs_native = float(ds_test._fs[0])
    window_native = int(round(input_size * fs_native))

    # ds_train holds one entry per record whose candidate-segment lengths give the
    # record length, so the expected val/test window count can be predicted exactly
    off, lens = ds_train._cand_offsets, ds_train._cand_lens
    predicted = np.array([sum(int(l) // int(ds_train._output_size[i])
                              for l in lens[off[i]:off[i + 1]])
                          for i in range(len(off) - 1)])
    _check(results, "val/test windows per record",
           int(np.median(per_record)) == int(np.median(predicted)),
           f"median {int(np.median(per_record))} windows/record; "
           f"predicted {int(np.median(predicted))} from the record lengths "
           f"(window = {window_native} native samples)")

    train_ids = ds_train.get_id_mapping()
    _check(results, "train: one window per record",
           len(train_ids) == len(np.unique(train_ids)),
           f"{len(train_ids)} windows / {len(np.unique(train_ids))} records "
           f"(chunkify_train=False)")

    # 4. resampling quality
    _check(results, "band-limited resampler available",
           signal_utils._HAS_RESAMPY,
           "resampy" if signal_utils._HAS_RESAMPY else "falling back to resample_poly")

    if fs_model < fs_native:
        spec = np.abs(np.fft.rfft(sig[1]))
        freqs = np.fft.rfftfreq(sig.shape[-1], d=1.0 / fs_model)
        nyq = fs_model / 2
        top = spec[freqs > 0.9 * nyq].sum()
        total = spec.sum() or 1.0
        _check(results, f"anti-aliasing ({fs_native:g}->{fs_model:g}Hz)",
               top / total < 0.02,
               f"{100 * top / total:.2f}% of spectral mass above 0.9*Nyquist")

    # 5. spot-check more samples for shape stability
    for i in np.linspace(0, len(ds_test) - 1, min(n_samples, len(ds_test)), dtype=int)[1:]:
        s = ds_test[int(i)]["signal"].numpy()
        if s.shape != (12, expected_len):
            _check(results, "shape stability", False, f"item {i}: {s.shape}")
            break
    else:
        _check(results, "shape stability", True, f"{min(n_samples, len(ds_test))} items")

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="ptbxl_super")
    ap.add_argument("--model", nargs="*", default=list(MODEL_CONTRACTS))
    ap.add_argument("--n-samples", type=int, default=8)
    args = ap.parse_args()

    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    failures = 0
    for model in args.model:
        print(f"\n=== {args.task} / {model} "
              f"({MODEL_CONTRACTS[model][0]}s @ {MODEL_CONTRACTS[model][1]}Hz) ===")
        try:
            for status, name, detail in run_model(args.task, model, args.n_samples):
                mark = {PASS: "  ok ", FAIL: "FAIL", WARN: "warn"}[status]
                print(f"  [{mark}] {name}" + (f"\n         {detail}" if detail else ""))
                failures += (status == FAIL)
        except Exception as exc:
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            failures += 1

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
