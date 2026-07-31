"""
Lead ordering
=============
The HEEDB H5 store keeps 12-lead ECGs in a non-standard channel order:

    ['I','II','III','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR']

while every published ECG foundation model (and the original
``ecg-fm-benchmarking`` pipeline, which reads WFDB records directly) assumes the
standard order:

    ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

Feeding the HEEDB order to a checkpoint pretrained on the standard order permutes
9 of 12 leads and destroys the pretrained representation. Rather than rewriting
the H5 store, the permutation is applied here, in the benchmark data loader, so
each encoder receives leads in whatever order it declares via ``lead_order``.

Encoders declare their expectation with a class attribute::

    lead_order = "standard"   # default when the attribute is absent
    lead_order = "heedb"      # MoRyECG adapters: pretrained on the H5 order

Two-lead stores (e.g. cpsc2021, ``['I','II']``) are passed through untouched.
"""

from __future__ import annotations

import ast
import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# Canonical orders ---------------------------------------------------------
STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]

HEEDB_12 = ["I", "II", "III", "V1", "V2", "V3",
            "V4", "V5", "V6", "aVF", "aVL", "aVR"]

LEAD_ORDERS = {
    "standard": STANDARD_12,
    "wfdb": STANDARD_12,      # alias — WFDB/PhysioNet records use the standard order
    "heedb": HEEDB_12,
    "native": None,           # sentinel: keep whatever the store provides
}


def canonical_lead_name(name: str) -> str:
    """Normalise a lead name so 'avr', 'AVR', ' aVR ' all compare equal."""
    key = str(name).strip().replace("-", "").replace("_", "").upper()
    aliases = {
        "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
        "I": "I", "II": "II", "III": "III",
        "V1": "V1", "V2": "V2", "V3": "V3",
        "V4": "V4", "V5": "V5", "V6": "V6",
        # occasionally seen spellings
        "MLI": "I", "MLII": "II", "MLIII": "III",
    }
    return aliases.get(key, key)


def parse_channel_names(value) -> Optional[list]:
    """Parse the ``channel_name`` column of a *_table.csv.

    The column is stored as the ``repr`` of a python list, e.g.
    ``"['I', 'II', 'III', 'V1', ...]"``. Returns None when it cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [canonical_lead_name(v) for v in value]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # fall back to a permissive split
        parsed = [t for t in text.strip("[]").replace("'", "").replace('"', "").split(",") if t.strip()]
    if not isinstance(parsed, (list, tuple)):
        return None
    return [canonical_lead_name(v) for v in parsed]


def resolve_target_order(lead_order, n_leads: int = 12) -> Optional[list]:
    """Map a ``lead_order`` spec to an explicit list of lead names.

    Accepts a key of :data:`LEAD_ORDERS`, an explicit list of names, or None.
    Returns None when no reordering should happen.
    """
    if lead_order is None:
        return None
    if isinstance(lead_order, (list, tuple)):
        return [canonical_lead_name(v) for v in lead_order]
    key = str(lead_order).strip().lower()
    if key not in LEAD_ORDERS:
        raise ValueError(
            f"unknown lead_order {lead_order!r}; expected one of "
            f"{sorted(LEAD_ORDERS)} or an explicit list of lead names")
    return LEAD_ORDERS[key]


def build_lead_permutation(source_names: Optional[Sequence[str]],
                           target_names: Optional[Sequence[str]]):
    """Index array ``perm`` such that ``sig[perm]`` is in ``target_names`` order.

    Returns None when no permutation is needed or possible:
      * either side unknown,
      * the orders already agree,
      * the source does not contain every target lead (e.g. a 2-lead store).
    """
    if not source_names or not target_names:
        return None

    src = [canonical_lead_name(v) for v in source_names]
    dst = [canonical_lead_name(v) for v in target_names]

    if src == dst:
        return None

    if len(src) != len(dst) or set(src) != set(dst):
        # Not a pure permutation (different lead sets / lead counts). The store
        # is left untouched; single-/two-lead datasets take this path.
        logger.info(
            "Lead reordering skipped: source %s is not a permutation of target %s",
            src, dst)
        return None

    index = {name: i for i, name in enumerate(src)}
    return np.asarray([index[name] for name in dst], dtype=np.int64)


def describe_permutation(source_names, target_names, perm) -> str:
    """Human-readable one-liner for logs."""
    if perm is None:
        return f"lead order kept as-is ({list(source_names) if source_names else 'unknown'})"
    return (f"lead reorder {list(source_names)} -> {list(target_names)} "
            f"(perm={perm.tolist()})")
