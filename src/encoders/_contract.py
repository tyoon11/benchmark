"""
Encoder contract
================
Every adapter declares the window it was benchmarked with in the original
``ecg-fm-benchmarking`` ``run.sh``::

    input_size  : float   # seconds           (--input-size)
    model_fs    : float   # Hz                (--fs-model)
    lead_order  : str     # 'standard' | 'heedb' | explicit list of lead names

``run.py`` reads these and configures the dataset accordingly, so the crop is
taken at the dataset's native rate and band-limit resampled to ``model_fs``
*before* it reaches the encoder — exactly the order the original uses
(``TimeSeriesDataset`` crop -> ``Resample`` transform).

Adapters must therefore **not** resample internally. :func:`ensure_length`
stays as a loud safety net for ad-hoc/legacy configs; if it ever fires during a
benchmark run, the task config and the encoder contract disagree.

Original run.sh values, for reference:

    ecg_founder   2.5 s @ 500 Hz     merl_resnet   2.5 s @ 500 Hz
    ecg_jepa     10.0 s @ 250 Hz     ecgfm_ked    10.0 s @ 500 Hz
    st_mem        2.4 s @ 250 Hz     cpc           2.5 s @ 240 Hz
    hubert_ecg    5.0 s @ 100 Hz     ecg_fm        5.0 s @ 500 Hz
"""

import logging

import torch.nn.functional as F

logger = logging.getLogger(__name__)

_WARNED = set()


def model_seq_len(encoder) -> int:
    """Samples the encoder expects: ``round(input_size * model_fs)``."""
    return int(round(float(encoder.input_size) * float(encoder.model_fs)))


def ensure_length(x, expected: int, name: str):
    """Return ``x`` with time dimension ``expected``, warning if resizing is needed.

    A correctly configured run never resizes here — the dataset already produced
    the right window. Resizing the *whole* window (rather than cropping at the
    native rate and resampling) is what the pre-parity implementation did, and
    it silently changed both the duration and the effective sampling rate seen
    by the model.
    """
    if x.shape[-1] == expected:
        return x
    if name not in _WARNED:
        _WARNED.add(name)
        logger.warning(
            "%s received %d samples but expects %d. The dataset should already "
            "deliver the encoder's window (input_size x model_fs); falling back to "
            "linear interpolation, which does NOT match the original pipeline. "
            "Check that run.py passed the encoder contract into the data config.",
            name, x.shape[-1], expected)
    return F.interpolate(x, size=expected, mode="linear", align_corners=False)
