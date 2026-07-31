# ECG Foundation Model Benchmark

Downstream evaluation framework for 12-lead ECG foundation models across **28 paper-canonical clinical tasks**.
Reproduces the evaluation protocol of [Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](https://arxiv.org/abs/2509.25095) and extends it with **MoRyECG** as the ninth model.

> **Quick orientation:** `run.py` trains one (model, task, mode) triple.
> `run_full_benchmark.sh` runs all models × tasks × modes across GPUs.
> `run_bootstrap.sh` computes 95% CI and produces the paper-style markdown table.

---

## Requirements

**Hardware:** ≥1 NVIDIA GPU with ≥16 GB VRAM (A100/H100 recommended for full benchmark).

**Software:**
- Python 3.10
- PyTorch ≥ 2.1, CUDA 12.1
- See `requirements.txt` for full dependency list

---

## Installation

```bash
git clone https://github.com/tyoon11/benchmark.git
cd benchmark
pip install -r requirements.txt
```

Verify the pipeline with a dummy encoder (no checkpoint or dataset needed):

```bash
python run.py --task ptbxl_super --eval_mode linear_probe --dummy --epochs 1
```

---

## Data setup

Set two environment variables that all scripts read:

```bash
export ECG_DATA_ROOT=/your/data/root   # H5 datasets + raw MIMIC CSV files
export ECG_CKPT_ROOT=/your/ckpt/root   # baseline FM checkpoint files
```

### Step 1 — Obtain H5 datasets

Datasets must first be converted to the standardized H5 format using the preprocessing
pipeline in [tyoon11/MoryECG](https://github.com/tyoon11/MoryECG) (`preprocessing/`).
See [MoRyECG preprocessing README](https://github.com/tyoon11/MoryECG/tree/main/preprocessing) for per-dataset instructions.

Place the resulting H5 files under `$ECG_DATA_ROOT/`:

```
$ECG_DATA_ROOT/
├── h5/
│   ├── physionet/v2.0/    # PTB, PTB-XL, Chapman/Ningbo, CPSC2018, CPSC-Extra, Georgia
│   ├── code15/v2.0/       # CODE-15%
│   ├── sph/v2.0/          # SPH
│   ├── ZZU-pECG/v2.0/     # ZZU pECG (pediatric)
│   ├── mimic4/v2.0/       # MIMIC-IV-ECG  (credentialed PhysioNet access required)
│   └── cpsc2021/v2.0/     # CPSC2021 (variant only)
└── raw/physionet.org/files/
    └── echonext/1.1.0/    # EchoNext (NumPy files, loaded directly)
```

Public datasets (PTB-XL, Chapman, CPSC2018, Georgia, etc.) are available on
[PhysioNet](https://physionet.org) without credentialing.
ZZU pECG is available from [Nature Scientific Data](https://www.nature.com/articles/s41597-025-05225-z).
SPH is available from [Figshare](https://doi.org/10.6084/m9.figshare.c.5779802).
CODE-15% is on [Zenodo](https://zenodo.org/records/4916206).

### Step 2 — Build label CSVs (non-MIMIC, ~2 min)

```bash
python scripts/build_labels_paper.py
```

Generates paper-canonical label CSVs and stratified fold columns for all
non-MIMIC tasks. Run once per machine (CSVs are excluded from git).

### Step 3 — Build MIMIC labels (~1 hour, credentialed access required)

Download the following datasets from PhysioNet and place under
`$ECG_DATA_ROOT/raw/physionet.org/files/<dataset>/`:

| Dataset | PhysioNet page | Required files |
|---|---|---|
| MIMIC-IV-ECG 1.0 | [link](https://physionet.org/content/mimic-iv-ecg/1.0/) | `machine_measurements.csv`, `record_list.csv` |
| MIMIC-IV-ECG-ICD 1.0.1 | [link](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/) | `records_w_diag_icd10.csv` |
| MIMIC-IV-ED 2.2 | [link](https://physionet.org/content/mimic-iv-ed/2.2/) | `ed/edstays.csv.gz`, `ed/vitalsign.csv.gz` |
| MIMIC-IV 3.1 (hosp) | [link](https://physionet.org/content/mimiciv/3.1/) | `admissions.csv.gz`, `omr.csv.gz`, `labevents.csv.gz`, `d_labitems.csv.gz` |
| MIMIC-IV 3.1 (icu) | [link](https://physionet.org/content/mimiciv/3.1/) | `chartevents.csv.gz`, `d_items.csv.gz`, `icustays.csv.gz` |
| MDS-ED 1.0.0 | [link](https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/) | `mds_ed.csv` |

Then build the joint label file:

```bash
./run_build_mimic_labels.sh
# Produces: labels/mimic_paper_labels.csv  (~116k rows × 1127 outputs)
#           labels/mimic_paper_labels.json (schema: cls_cols / reg_cols / report_groups)
```

Skip this step if you only need non-MIMIC tasks.

---

## Download baseline FM checkpoints

Place checkpoints under `$ECG_CKPT_ROOT/` with this layout:

```
$ECG_CKPT_ROOT/
├── ecg_founder/12_lead_ECGFounder.pth
├── ecg_jepa/multiblock_epoch100.pth
├── st_mem/st_mem_vit_base_full.pth
├── merl/res18_best_encoder.pth
├── ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt
├── hubert_ecg/hubert_ecg_base.safetensors
├── ecg_fm/mimic_iv_ecg_physionet_pretrained.pt
└── cpc/last_11597276.ckpt
```

| Model | Download source |
|---|---|
| ECGFounder | [HuggingFace — PKUDigitalHealth/ECGFounder](https://huggingface.co/PKUDigitalHealth/ECGFounder) |
| ECG-JEPA, ST-MEM, MERL, HuBERT-ECG, ECG-FM, ECG-CPC | [AI4HealthUOL/ECG-FM-Benchmarking](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) (see their README) |
| ECGFM-KED | [Zenodo 14881564](https://zenodo.org/records/14881564) |

**MoRyECG:** checkpoint is auto-located from `MORYECG_REPO` (defaults to the
parent of this `benchmark/` directory if cloned inside the MoRyECG repo).
See [tyoon11/MoryECG](https://github.com/tyoon11/MoryECG) for pretrained checkpoints.

---

## Parity with the original benchmark

The evaluation protocol is aligned to
[`ecg-fm-benchmarking`](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)
(`code/main_lite.py` + `code/run.sh`). Verify it before trusting any number:

```bash
python scripts/verify_parity.py --task ptbxl_super   # window, leads, chunking, resampling
python scripts/audit_cohorts.py                      # record counts, label vocab, folds
```

### The encoder contract

The input window comes from the **model**, not the task config. Each adapter declares
`input_size` (seconds) and `model_fs` (Hz) — the equivalents of `--input-size` /
`--fs-model` in the original `run.sh` — and `run.py` feeds them to the dataset:

| model | input_size | model_fs | samples |
|---|:-:|:-:|:-:|
| ECGFounder / MERL | 2.5 s | 500 Hz | 1250 |
| ST-MEM | 2.4 s | 250 Hz | 600 |
| ECG-JEPA | 10 s | 250 Hz | 2500 |
| ECGFM-KED | 10 s | 500 Hz | 5000 |
| HuBERT-ECG | 5 s | 100 Hz | 500 |
| ECG-FM | 5 s | 500 Hz | 2500 |
| ECG-CPC | 2.5 s | 240 Hz | 600 |

The dataset crops at the record's **native** rate, then band-limit resamples
(`resampy`, as the original's `Resample` transform does) — so adapters must not
resample internally. Task configs no longer carry `target_fs` / `target_length`.

### Lead order

The HEEDB H5 store keeps channels as
`I, II, III, V1…V6, aVF, aVL, aVR`, while every published checkpoint (and the
original pipeline, which reads WFDB directly) assumes
`I, II, III, aVR, aVL, aVF, V1…V6`. The H5 files are left untouched; the loader
permutes leads to whatever the encoder declares via `lead_order`:

* published baselines → `"standard"` (permutation `[0,1,2,11,10,9,3,4,5,6,7,8]`)
* MoRyECG adapters → `"heedb"` (pass-through; the model and its preprocessing
  cache were built on that layout)

`verify_parity.py` proves this physiologically: with standard ordering the
Einthoven/Goldberger identities (`II = I + III`, `aVR = -(I+II)/2`, …) hold to
~0.001 mV, and fails them at ~1.7 mV without the permutation.

### Reproduced quirks

Two behaviours of the original are bugs, but reproducing them is what makes the
numbers comparable. Both are switchable:

| behaviour | flag to disable |
|---|---|
| Frozen encoders stay in **train** mode while probing (dropout on, BatchNorm running stats still updating) because the original never overrides `train()` | `--no_paper_frozen` |
| `StMemWrapper.get_params()` matches nothing, so ST-MEM `finetune_linear` trains the head only | `ECGFM_PAPER_QUIRKS=0` |

`auroc_macro` follows the original `multiclass_roc_curve`: the mean of per-class
AUCs over **all** classes, scoring unscoreable classes as **0.5** rather than
skipping them. The skip-variant is reported alongside as `auroc_macro_skipnan`.
Every split reports both `noagg_*` (per window) and unprefixed (per record,
windows averaged) metrics; checkpoint selection uses the aggregated one, as in
`ModelCheckpoint(monitor="macro_auc_agg_val0")`.

### Known residual gaps

These are ingest-side and cannot be closed from the benchmark layer:

* The H5 store holds fewer records than the source datasets (chapman −414,
  ningbo −1480, georgia −61, cpsc2018 −28, cpsc_extra −20). That shifts
  borderline labels across the `min_cnt=10` cut, so **chapman (41 vs 42),
  cpsc_extra (32 vs 33) and ningbo (67 vs 68)** have one class fewer than the
  original. `run.py` warns; absolute numbers for those three tasks are not
  directly comparable to the published table.
* Folds are regenerated by `scripts/build_folds.py` rather than inherited from
  the original's preprocessing. `stratify()` is seeded (`random_seed=0`), so the
  assignment is deterministic, but it depends on record order and cohort, which
  differ by the counts above. Only PTB-XL reuses the original `strat_fold`.

### MoRyECG preprocessing cache

`$MORYECG_CACHE` holds one R-peak/beat/STFT bundle per `(filepath, seg_idx)`,
computed from the **whole segment** straight out of the H5 at 500 Hz in the
store's own lead order. On a cache hit the encoder consumes the bundle and
ignores the windowed tensor entirely, so entry content does **not** depend on
`input_size` — the parity change did not invalidate it (verified by recomputing
entries and diffing: equal to fp16 storage precision, identical `n_valid`).

What did change is coverage. Task configs used to leave `seg_idx` unset, so
`precompute_moryecg_cache.py` only ever cached `seg 0`; the dataset now defaults
to `seg_mode: all`. Segments past the first miss and fall back to live
preprocessing — correct, but slow. Top the cache up (existing entries are
skipped; `--force` to recompute everything):

```bash
MORYECG_CACHE=/path/to/cache python scripts/precompute_moryecg_cache.py \
    --tasks ptb ningbo cpsc2018 ... --workers 96
```

The cache root carries a `_cache_stamp.json` recording the pipeline version,
`model_fs`, lead order and `seg_mode`; `run.py` warns when it is missing or
disagrees.

---

## Running the benchmark

### Single experiment

```bash
# Linear probe — ECGFounder on PTB-XL super-category
python run.py \
    --task ptbxl_super \
    --eval_mode linear_probe \
    --encoder_cls src.encoders.ecg_founder.ECGFounderEncoder \
    --encoder_ckpt $ECG_CKPT_ROOT/ecg_founder/12_lead_ECGFounder.pth

# Fine-tune — ECG-JEPA on CODE-15%
python run.py \
    --task code15 \
    --eval_mode finetune_linear \
    --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
    --encoder_ckpt $ECG_CKPT_ROOT/ecg_jepa/multiblock_epoch100.pth \
    --lr 5e-4 --epochs 30

# MoRyECG (cb1024) — run from inside the MoRyECG repo, or set MORYECG_REPO
MORYECG_REPO=/path/to/MoryECG \
python run.py \
    --task ptbxl_super --eval_mode linear_probe \
    --encoder_cls src.encoders.moryecg.MoRyECGEncoder \
    --encoder_ckpt /path/to/MoryECG/checkpoints/pretrain_heedb_cb1024_v4/best.pt
```

Results are saved under `results/<timestamp>/<model>_<task>_<mode>/`.

### Full benchmark (all models × all tasks × all modes)

```bash
# All 9 models, use GPUs 0-3
GPU_IDS="0 1 2 3" bash run_full_benchmark.sh

# Subset of models
MODELS_OVERRIDE="ecg_founder moryecg_cb1024" bash run_full_benchmark.sh

# Subset of tasks
TASKS_OVERRIDE="ptbxl_super code15 mimic" bash run_full_benchmark.sh
```

### Resume interrupted runs

```bash
# Fill any missing (model, task, mode) triples in an existing timestamp dir
TIMESTAMP=20260511_172035 GPU_IDS_OVERRIDE="0 1 2 3" \
    bash run_task_parallel_benchmark.sh

# Dry run — show what's missing without running anything
DRY_RUN=1 TIMESTAMP=20260511_172035 bash run_task_parallel_benchmark.sh
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 run.py \
    --task code15 --eval_mode finetune_linear \
    --encoder_cls src.encoders.ecg_founder.ECGFounderEncoder \
    --encoder_ckpt $ECG_CKPT_ROOT/ecg_founder/12_lead_ECGFounder.pth
```

---

## Statistical evaluation (bootstrap CI + paper tables)

```bash
# All 4 stages: extract predictions → CI → pairwise → tables
bash run_bootstrap.sh results/<timestamp>

# Use 4 GPUs for extraction, 32 CPU workers for CI computation
WORKERS=32 bash run_bootstrap.sh results/<timestamp> "0,1,2,3"

# Skip re-extraction if predictions already exist
SKIP_EXTRACT=1 bash run_bootstrap.sh results/<timestamp>
```

Outputs under `results/<timestamp>/pairwise/`:

| File | Description |
|---|---|
| `summary_linear_probe.md` | Paper-style markdown table (linear probe) |
| `summary_attention_probe.md` | Paper-style markdown table (attention probe) |
| `summary_finetune_linear.md` | Paper-style markdown table (finetune) |
| `pairwise_summary.csv` | Ranks + significance for all model pairs |
| `summary_ci_long.csv` | Point estimate + 95% CI, long format |

**Bold** = unique best (point estimate). __Underline__ = tied with best (95% paired bootstrap CI of difference includes 0).

---

## Tasks

| Category | Tasks (n) | Primary metric |
|---|---|---|
| Adult ECG interpretation | PTB, Ningbo, CPSC2018, CPSC-Extra, Georgia, Chapman, Chapman (rhythm), CODE-15%, SPH, PTB-XL ×6 (15) | Macro-AUROC ↑ |
| Pediatric ECG (1) | ZZU pECG | Macro-AUROC ↑ |
| Cardiac structure (1) | EchoNext | Macro-AUROC ↑ |
| MIMIC — classification (6) | Cardiac, Non-cardiac, Deterioration, Mortality, ICU, Sex | Macro-AUROC ↑ |
| MIMIC — regression (5) | Age, Biometrics, ECG Features, Lab Values, Vital Signs | Z-norm MAE ↓ |

All 11 MIMIC sub-tasks come from a **single joint model** (1127 outputs: 1092 classification + 35 regression) trained with NaN-masked BCE + L1 composite loss. Per-sub-task metrics are read from one `test_metrics.txt`.

### Evaluation modes

| Mode | Encoder weights | Head | Description |
|---|---|---|---|
| `linear_probe` | Frozen | Linear | Representation quality baseline |
| `attention_probe` | Frozen | V-JEPA learnable-query attention pool | Sequence-level representation |
| `finetune_linear` | Trainable | Linear | End-to-end fine-tuning |

---

## Results snapshot (Finetune linear, macro-AUROC ↑ / z-norm MAE ↓)

| Task | ECGFounder | ECG-JEPA | ST-MEM | MERL | ECGFM-KED | HuBERT-ECG | ECG-FM | ECG-CPC | **MoRyECG** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| PTB-XL (all) ↑ | 0.933 | 0.930 | **0.947** | 0.932 | 0.910 | 0.928 | 0.925 | **0.952** | 0.921 |
| CODE-15% ↑ | 0.979 | 0.981 | **0.982** | 0.976 | 0.976 | 0.980 | 0.981 | 0.980 | 0.955 |
| CPSC-Extra ↑ | 0.852 | 0.862 | 0.856 | 0.825 | 0.808 | 0.829 | 0.798 | 0.861 | **0.866** |
| ZZU pECG ↑ | 0.910 | 0.914 | 0.914 | 0.908 | 0.880 | 0.897 | 0.896 | **0.925** | 0.884 |
| EchoNext ↑ | 0.808 | 0.817 | 0.833 | 0.823 | 0.812 | 0.786 | 0.819 | **0.834** | 0.786 |
| MIMIC (Cardiac) ↑ | 0.773 | 0.788 | **0.803** | 0.790 | 0.782 | 0.770 | 0.802 | 0.795 | 0.769 |
| MIMIC (Age) ↓ | 0.532 | 0.536 | 0.504 | 0.534 | 0.548 | 0.589 | **0.499** | 0.512 | 0.598 |

Full 28-task × 3-mode tables are generated by `run_bootstrap.sh`.

---

## Adding your own encoder

### 1. Write an adapter (`src/encoders/my_model.py`)

```python
import torch, torch.nn as nn

class MyModelEncoder(nn.Module):
    # Encoder contract — read by run.py, see src/encoders/_contract.py
    input_size    = 10.0       # window the model was pretrained on (seconds)
    model_fs      = 500        # expected sampling rate (Hz)
    model_seq_len = 5000       # = input_size × model_fs
    lead_order    = "standard" # I,II,III,aVR,aVL,aVF,V1..V6 ("heedb" for the H5 order)
    feature_dim   = 768        # output embedding size

    def __init__(self, checkpoint=None):
        super().__init__()
        self.model = ...   # instantiate your backbone
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state.get("model", state), strict=False)

    def forward(self, x):
        """x: (B, 12, T)  →  (seq_feat (B, T', D),  pooled (B, D))"""
        x = torch.nan_to_num(x)
        seq_feat = self.model(x)
        pooled   = seq_feat.mean(dim=1)
        return seq_feat, pooled
```

`DownstreamWrapper` accepts: `(seq, pooled)` tuple · `{"seq":…, "pooled":…}` dict ·
`(B, D)` tensor (pooled only) · `(B, T, D)` tensor (GAP pooled automatically).

### 2. Register (3 lines total)

```python
# src/encoders/__init__.py
from .my_model import MyModelEncoder
```

```bash
# configs/models.sh
MODEL_CLS_MAP[my_model]="src.encoders.my_model.MyModelEncoder"
MODEL_CKPT_MAP[my_model]="/path/to/ckpt.pt"
```

### 3. Run

```bash
MODELS_OVERRIDE="my_model" bash run_full_benchmark.sh
```

### Common pitfalls

| Symptom | Root cause | Fix |
|---|---|---|
| Scores well below paper across every task | Wrong `lead_order` — a pretrained checkpoint is seeing permuted leads | Set `lead_order` correctly and confirm with `scripts/verify_parity.py` |
| `ensure_length` warning in the log | `input_size`/`model_fs` disagree with what the dataset produced | Fix the class attributes; never resample inside `forward` |
| Scores a few pts below paper | `input_size` not set → wrong window and no multi-window aggregation | Declare `input_size` and `model_fs` as class attributes |
| IndexError or garbage features | Model expects fewer than 12 leads | Select leads in `forward`: `x = x[:, lead_idx, :]` (indices are in *standard* order) |
| Scores near chance on frozen eval | Wrong wrapper — importing raw backbone only, missing input projection | Import the full wrapper class from the original repo |

---

## Project layout

```
benchmark/
├── run.py                          # single-experiment entrypoint
├── run_full_benchmark.sh           # all models × tasks × modes (parallel)
├── run_task_parallel_benchmark.sh  # task-level GPU scheduler (fill missing jobs)
├── run_bootstrap.sh                # 4-stage bootstrap CI pipeline
├── run_build_mimic_labels.sh       # MIMIC joint label build (~1 hour)
├── configs/
│   ├── default.yaml                # lr, epochs, head architecture defaults
│   ├── models.sh                   # model class + checkpoint path registry
│   └── tasks/                      # one YAML per task (28 canonical + 7 variants)
├── src/
│   ├── dataset.py                  # H5ECGDataset — NaN-preserving, multi-window
│   ├── wrapper.py                  # DownstreamWrapper — encoder-agnostic
│   ├── trainer.py                  # BCE / masked BCE / masked L1 dispatch
│   ├── metrics.py                  # AUROC, AUPRC, F1, MAE, RMSE, R²
│   ├── heads.py                    # Linear / attention / MLP heads
│   └── encoders/                   # 9 encoder adapters (8 reference FMs + MoRyECG)
├── labels/
│   └── mimic_paper_labels.json     # MIMIC schema (cls_cols / reg_cols / groups)
└── scripts/
    ├── build_labels_paper.py       # non-MIMIC paper-canonical label CSVs
    ├── build_mimic_labels.py       # MIMIC per-source intermediate labels
    ├── merge_mimic_joint.py        # merge per-source → mimic_paper_labels.csv
    ├── precompute_moryecg_cache.py # offline R-peak/beat cache for MoRyECG
    ├── extract_predictions.py      # bootstrap stage 1: best.pt → preds.npy
    ├── bootstrap_ci.py             # bootstrap stage 2: single-model 95% CI
    ├── bootstrap_pairwise.py       # bootstrap stage 3: pairwise + tied rank
    └── make_summary_table.py       # bootstrap stage 4: paper-style tables
```

---

## References

- Paper: *Benchmarking ECG FMs: A Reality Check Across Clinical Tasks* · [arXiv 2509.25095](https://arxiv.org/abs/2509.25095)
- Original benchmark: [AI4HealthUOL/ECG-FM-Benchmarking](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking)
- MoRyECG model repo: [tyoon11/MoryECG](https://github.com/tyoon11/MoryECG)
