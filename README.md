# Benchmark — ECG Foundation Model Downstream Evaluation

Self-contained framework that plugs an ECG encoder into the
**paper-canonical clinical tasks** (17 ECG interpretation + 1 joint MIMIC-IV-ECG
multi-task that internally reports 8 paper-Table sub-results) and runs
Linear Probe / Attention Probe / Full Fine-tune.

The training and evaluation procedure of
[*Benchmarking ECG FMs: A Reality Check Across Clinical Tasks*](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking)
is reproduced verbatim — per-encoder input window, random-crop augmentation at
train, multi-window mean aggregation at val/test, layer-dependent LR, and the
three task families (binary / multi-label-binary / regression) with NaN
masking — all identical to the reference paper.

The MoRyECG main and ablation results in paper Tables 1, 2, 3, 4, 7, 8 are
produced by this package.

---

## Quick start

```bash
pip install -r requirements.txt

# Smoke test — dummy encoder, 1 epoch
python run.py --task ptbxl_super --eval_mode linear_probe --dummy --epochs 1
```

Paper baseline code (the `clinical_ts` subset) is bundled in
[`src/external/`](src/external/) — no external clone needed. To actually
train you need (1) a pretrained encoder checkpoint and (2) the ECG datasets.

---

## What's included

### 8 reference encoders

| Model | Input | Params | Adapter |
|---|:-:|:-:|---|
| ECGFounder | 2.5 s @ 500 Hz | 30.7 M | `src.encoders.ecg_founder.ECGFounderEncoder` |
| ECG-JEPA | 10.0 s @ 250 Hz | 85.4 M | `src.encoders.ecg_jepa.ECGJEPAEncoder` |
| ST-MEM | 2.4 s @ 250 Hz | 88.5 M | `src.encoders.st_mem.StMemEncoder` |
| CPC | 2.5 s @ 240 Hz | 3.2 M | `src.encoders.cpc.CPCEncoder` |
| MERL ResNet | 2.5 s @ 500 Hz | 3.8 M | `src.encoders.merl.MerlResNetEncoder` |
| ECGFM-KED | 10.0 s @ 500 Hz | 7.9 M | `src.encoders.ecgfm_ked.EcgFmKEDEncoder` |
| HuBERT-ECG | 5.0 s @ 100 Hz | 93.1 M | `src.encoders.hubert_ecg.HuBERTECGEncoder` |
| ECG-FM | 5.0 s @ 500 Hz | 90.4 M | `src.encoders.ecg_fm.ECGFMEncoder` |

**MoRyECG** itself is also wired up — adapter at
[`src/encoders/moryecg.py`](src/encoders/moryecg.py), registered in
`configs/models.sh` under the keys `moryecg_cb{128,256,512,1024,2048}`.
It loads the pretrained checkpoint produced by `../training/pretrain/train.py`
and runs the full R-peak → beat-tokenize → STFT pipeline inside `forward()`
(matches the pretrain preprocessing exactly). Run with:

```bash
python run.py --task ptbxl_super --eval_mode linear_probe \
    --encoder_cls src.encoders.moryecg.MoRyECGEncoder \
    --encoder_ckpt /path/to/pretrain_heedb_cb1024_v4/best.pt
```

The companion `tokenizer_heedb_full_cb{K}_v4/best.pt` is auto-located in the
sibling directory of the pretrain checkpoint. Set `MORYECG_CACHE=<dir>` to
enable an opt-in preprocessing cache (R-peak detection is the bottleneck;
~10× speedup after warm-up).

### 28 paper-canonical tasks + 7 variants

```
Adult ECG interpretation:    ptb, ningbo, cpsc2018, cpsc_extra, georgia,
                             chapman, chapman_rhythm, code15, sph_diag,
                             ptbxl_{all, super, sub, diag, form, rhythm}
Pediatric ECG interp:        zzu_pecg
Cardiac structure & func:    echonext              (NumPy loader)

MIMIC-IV-ECG (1 joint task, paper-faithful):
  mimic                      classification_and_regression — 158 diag + 6 det +
                             1 sex + 35 metadata-regression = 200 outputs,
                             single model trained with BCE+L1 composite loss,
                             per-sub-task metrics reported (cardiac, deterioration,
                             sex, age, biometrics, ecg_features, labvalues, vitals)
                             from the same trained model (paper Tables 3/4/7/8).

Variants:                    code15_diag, code15_diag_jepa, cpsc2021_af,
                             physionet_all, ptbxl_super_jepa
```

### Task types (paper `main_lite_ecg.py:92-139`)

| `task_type` | Loss | Eval metric | NaN handling | Examples |
|---|---|---|---|---|
| `binary` (default) | BCE-with-logits | AUROC / AUPRC / F1 | NaN → 0 (negative) | ptbxl_*, chapman, ... |
| `multi-label-binary` | masked BCE | AUROC / AUPRC / F1 | NaN masked (paper L114) | — |
| `regression` | masked L1 (MAE) | MAE / MSE / RMSE / R² / neg-MAE | NaN masked (paper L128) | — |
| `classification_and_regression` | masked BCE(cls) + masked L1(reg) | composite_score = (1−AUROC) + global_MAE; per-sub-task AUROC/MAE | NaN masked (paper L100-139) | `mimic` |

Set `task.task_type` in the task YAML; default is `binary`.

### 4 evaluation modes

| Mode | Encoder | Head | Purpose |
|---|---|---|---|
| `linear_probe` | Frozen | Linear | representation quality (default) |
| `attention_probe` | Frozen | V-JEPA learnable-query attention pool | sequence-level representation |
| `finetune_linear` | Trainable | Linear | end-to-end fine-tune |
| `finetune_attention` | Trainable | V-JEPA attention pool | end-to-end + attention head |

Fine-tune modes apply layer-dependent LR automatically:
`head = lr`, `late = lr × 0.1`, `early = lr × 0.01`.

### Multi-window training + test-time aggregation (paper §3.3)

| Split | Behavior |
|---|---|
| **Train** | 1 sample per ECG; `__getitem__` picks a random offset chunk. ~100 view augmentations across 100 epochs |
| **Val / Test** | 1 ECG → ⌊target_length / chunk_length⌋ deterministic non-overlapping chunks; predictions are mean-aggregated by ECG ID |

`run.py` reads `encoder.chunk_seconds` and switches between random and
deterministic chunking by split — no extra config required.

---

## Usage

### Single experiment

```bash
# Linear probe
python run.py --task ptbxl_super --eval_mode linear_probe \
    --encoder_cls src.encoders.ecg_founder.ECGFounderEncoder \
    --encoder_ckpt /path/to/ckpt.pth

# Full fine-tune (lower LR)
python run.py --task code15 --eval_mode finetune_linear \
    --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
    --encoder_ckpt /path/to/jepa.pth --lr 5e-4 --epochs 30

# Dummy encoder — exercises the pipeline without external dependencies
python run.py --task ptbxl_super --eval_mode linear_probe --dummy --epochs 1
```

CLI overrides: `--epochs`, `--lr`, `--batch_size`, `--device`, `--save_dir`,
`--train_folds`, ...

### Full benchmark

```bash
bash run_full_benchmark.sh all
MODELS_OVERRIDE="ecg_founder ecg_jepa" bash run_full_benchmark.sh
TASKS_OVERRIDE="ptbxl_super echonext" bash run_full_benchmark.sh
```

Outputs land under `results/<timestamp>/`.

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 run.py --task ptbxl_super --eval_mode finetune_linear \
    --encoder_cls src.encoders.ecg_founder.ECGFounderEncoder \
    --encoder_ckpt /path/to/ckpt.pth
```

---

## Statistical evaluation (bootstrap CI + tied ranking)

Paper §3.5 evaluation protocol, reproduced exactly:

* **Classification primary metric**: macro-averaged AUROC (↑)
* **Regression primary metric**: z-normalised MAE (↓)
* **Significance**: empirical bootstrap on the test set (`n = 1000`); a pair
  is significant if the 95% CI of the per-bootstrap diff excludes 0
* **Tied rank**: paired-diff CI containing 0 → statistically tied

```bash
bash run_bootstrap.sh /path/to/results/<timestamp>
WORKERS=32 bash run_bootstrap.sh /path/to/results/<timestamp> "0,1,2,3"
FILTER=cpc bash run_bootstrap.sh /path/to/results/<timestamp> "0"
SKIP_EXTRACT=1 bash run_bootstrap.sh /path/to/results/<timestamp>
```

### 4-stage pipeline

| Stage | Script | Role |
|---|---|---|
| 1. **Extract** | [`extract_predictions.py`](scripts/extract_predictions.py) | best.pt → preds.npy, targets.npy, ids.npy |
| 2. **Single-model CI** | [`bootstrap_ci.py`](scripts/bootstrap_ci.py) | n=1000 empirical bootstrap (matches paper) |
| 3. **Pairwise + tied rank** | [`bootstrap_pairwise.py`](scripts/bootstrap_pairwise.py) | paired-diff CI with shared bootstrap index + union-find rank groups |
| 4. **Paper-style table** | [`make_summary_table.py`](scripts/make_summary_table.py) | `summary_<mode>.{csv,md}` with **bold** = unique best, __underline__ = tied with best |

### Output layout

```
results/<timestamp>/
├── <model>_<task>_<mode>/
│   ├── best.pt, test_metrics.txt        (training stage)
│   ├── preds.npy        (N, C)          # stage 1
│   ├── targets.npy      (N, C)
│   ├── ids.npy          (N,)            # pairwise alignment key
│   ├── preds_meta.json
│   └── bootstrap.json                   # stage 2
│
├── bootstrap_summary.csv
└── pairwise/                            # stages 3-4
    ├── pairwise_diff_<task>_<mode>.csv
    ├── tied_groups_<task>_<mode>.txt
    ├── pairwise_summary.csv
    ├── summary_<mode>.csv
    ├── summary_<mode>_marked.csv
    ├── summary_<mode>.md
    └── summary_ci_long.csv
```

---

## Adding a new model (e.g., MoRyECG)

Create `src/encoders/my_model.py`:

```python
import sys, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "external"))


class MyModelEncoder(nn.Module):
    chunk_seconds = 2.5     # paper run.sh --input-size
    model_fs      = 500     # paper run.sh --fs-model
    model_seq_len = 1250    # = chunk_seconds * model_fs

    feature_dim = 768

    def __init__(self, checkpoint=None):
        super().__init__()
        self.model = ...
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)

    def forward(self, x):
        """x: (B, 12, T) -> (sequence_features (B, T', D), pooled (B, D))"""
        x = torch.nan_to_num(x)
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len,
                              mode="linear", align_corners=False)
        seq_feat = self.model(x)
        pooled   = seq_feat.mean(dim=1)
        return seq_feat, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, p in self.named_parameters():
            if name.startswith(("stem", "block0", "block1")):
                early.append(p)
            else:
                late.append(p)
        return {"early": early, "late": late}
```

`DownstreamWrapper` accepts four `forward` output shapes:

1. tuple `(seq_feat, pooled)` — recommended
2. dict `{"seq": ..., "pooled": ...}`
3. tensor `(B, D)` — pooled only
4. tensor `(B, T, D)` — seq only (GAP applied automatically)

### Common pitfalls

* `chunk_seconds` missing → multi-window aggregation disabled, paper numbers unreachable.
* 8-lead models must select channels inside `forward` (`x = x[:, lead_idx, :]`).
* BatchNorm — `DownstreamWrapper` freezes BN stats automatically during frozen eval.
* Pretrained-backbone wrapper extras — if the paper wrapper class
  (`S4Predictor`, `RNNEncoder`, ...) adds an input projection around the raw
  backbone, importing only the raw backbone yields a random-init layer and
  corrupts features (measured: CPC 0.78 vs paper 0.88, a one-line diff).

### Register (3 lines)

`src/encoders/__init__.py`:
```python
from .my_model import MyModelEncoder
```

`configs/models.sh`:
```bash
MODEL_CLS_MAP[my_model]="src.encoders.my_model.MyModelEncoder"
MODEL_CKPT_MAP[my_model]="/path/to/ckpt.pt"
MODEL_NAMES_DEFAULT+=(my_model)
```

---

## Data preparation

Two environment variables drive everything:

```bash
export ECG_DATA_ROOT=/your/data/root
export ECG_CKPT_ROOT=/your/ckpt/root
```

### Directory layout (env-var-relative)

```
$ECG_DATA_ROOT/
├── h5/
│   ├── physionet/v2.0/      # PTB-XL, Chapman, CPSC2018, CPSC-Extra, Georgia, PTB
│   ├── code15/v2.0/         # CODE-15%
│   ├── sph/v2.0/            # SPH
│   ├── ZZU-pECG/v2.0/       # ZZU pECG
│   ├── mimic4/v2.0/         # MIMIC-IV-ECG (~800k records)
│   └── cpsc2021/v2.0/       # CPSC2021 (variant only)
└── raw/physionet.org/files/
    ├── echonext/1.1.0/                                    # EchoNext NumPy
    ├── mimic-iv-ecg/1.0/                                  # machine_measurements.csv
    ├── mimic-iv-ecg-ext-icd-labels/1.0.1/                 # records_w_diag_icd10.csv
    ├── mimic-iv-ed/2.2/ed/                                # vitalsign, edstays
    ├── mimiciv/3.1/{hosp,icu}/                            # omr, labevents, chartevents, ...
    └── multimodal-emergency-benchmark/1.0.0/              # mds_ed.csv (MDS-ED)

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

### Pretrained baseline checkpoints (download URLs)

| Model | URL |
|---|---|
| ECGFounder | <https://huggingface.co/PKUDigitalHealth/ECGFounder> |
| ECG-JEPA | from [`AI4HealthUOL/ECG-FM-Benchmarking`](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) |
| ST-MEM | from [`AI4HealthUOL/ECG-FM-Benchmarking`](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) |
| MERL ResNet | from [`AI4HealthUOL/ECG-FM-Benchmarking`](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) |
| ECGFM-KED | <https://zenodo.org/records/14881564> |
| HuBERT-ECG / ECG-FM / CPC | from [`AI4HealthUOL/ECG-FM-Benchmarking`](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) |

---

## MIMIC label build

The single joint MIMIC task is generated from raw credentialed PhysioNet data
in two phases:

1. **Per-source intermediate labels** ([`scripts/build_mimic_labels.py`](scripts/build_mimic_labels.py)):
   reproduces the original
   [`mimic_preprocessing.py`](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking/blob/main/mimic_preprocessing.py)
   1-to-1 to produce per-source label CSVs (cardiac/sex/age/ecg_features/
   deterioration/biometrics/vitals/labvalues).
2. **Joint merge** ([`scripts/merge_mimic_joint.py`](scripts/merge_mimic_joint.py)):
   merges the per-source CSVs onto the `is_diagnostic==1` cohort, producing
   `labels/mimic_paper_labels.csv` (~116k rows × 165 cls + 35 reg columns)
   and `labels/mimic_paper_labels.json` (schema with per-MIMIC-sub-task
   report groups). This is the file consumed by `run.py --task mimic`.

### Required raw files

| Dataset | Page | Files |
|---|---|---|
| MIMIC-IV-ECG (1.0) | <https://physionet.org/content/mimic-iv-ecg/1.0/> | `machine_measurements.csv`, `record_list.csv` |
| MIMIC-IV-ECG-ICD (1.0.1) | <https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/> | `records_w_diag_icd10.csv` |
| MIMIC-IV-ED (2.2) | <https://physionet.org/content/mimic-iv-ed/> | `ed/edstays.csv.gz`, `ed/vitalsign.csv.gz` |
| MIMIC-IV (3.1) `hosp/` | <https://physionet.org/content/mimiciv/3.1/> | `admissions.csv.gz`, `omr.csv.gz`, `labevents.csv.gz`, `d_labitems.csv.gz` |
| MIMIC-IV (3.1) `icu/` | <https://physionet.org/content/mimiciv/3.1/> | `chartevents.csv.gz`, `d_items.csv.gz`, `icustays.csv.gz` |
| MDS-ED (1.0.0) | <https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/> | `mds_ed.csv` |

Place them under
`$ECG_DATA_ROOT/raw/physionet.org/files/<dataset>/...` (see top of
`build_mimic_labels.py` for exact paths).

### Parallel build (4 stages)

```bash
./run_build_mimic_labels.sh
```

* Stage 1 (parallel, ~2 min): diagnostic, sex, ecg_features, deterioration,
  mortality, icu_admission
* Stage 2 (serial, ~40 min): biometrics — chunked filter of
  `chartevents.csv.gz` (~30 GB) + cache build
* Stage 3 (parallel, ~15 min): vitals + labvalues (reuses cache)
* Stage 4 (serial, <30 s): merge per-source CSVs → `mimic_paper_labels.csv`

Total wall time ~1 h. Per-task logs land in `labels/_logs/build_<task>.log`.

### Result

```
labels/
├── mimic_paper_labels.csv                     (116k rows × 165 cls + 35 reg — joint task)
├── mimic_paper_labels.json                    (schema: cls_cols / reg_cols / report_groups)
└── (intermediates consumed by the merge step)
    ├── mimic_cardiac_paper_labels.csv         (158 diag, is_diagnostic==1 cohort)
    ├── mimic_sex_paper_labels.csv             (1)
    ├── mimic_age_paper_labels.csv             (1 reg)
    ├── mimic_ecg_features_paper_labels.csv    (7 reg)
    ├── mimic_deterioration_paper_labels.csv   (6 multi-label-binary)
    ├── mimic_biometrics_paper_labels.csv      (3 reg)
    ├── mimic_vitals_paper_labels.csv          (6 reg)
    └── mimic_labvalues_paper_labels.csv       (18 reg)
```

---

## Project layout

```
benchmark/
├── run.py                          # single-experiment entrypoint
├── run_full_benchmark.sh           # all models × all tasks × all modes (parallel)
├── run_parallel_tasks.sh           # one model × all tasks
├── run_build_mimic_labels.sh       # MIMIC 11-task labels — 3-stage parallel
├── run_bootstrap.sh                # bootstrap pipeline orchestrator (4 stages)
├── run_benchmark.sh                # single (model, task, mode) wrapper
├── configs/
│   ├── default.yaml                # base training config (lr, epochs, head)
│   ├── models.sh                   # model registry
│   └── tasks/                      # task YAMLs (paper 28 + 7 variants)
├── src/
│   ├── dataset.py                  # H5ECGDataset (task_type dispatch, NaN-preserving)
│   ├── dataset_numpy.py            # EchoNextDataset
│   ├── wrapper.py                  # DownstreamWrapper (encoder-agnostic)
│   ├── heads.py                    # Linear / V-JEPA attention / MLP heads
│   ├── trainer.py                  # BCE / masked BCE / masked L1 dispatch
│   ├── metrics.py                  # AUROC / AUPRC / F1 + MAE / MSE / RMSE / R²
│   ├── encoders/                   # 8 reference encoder adapters
│   └── external/clinical_ts/       # bundled paper backbone subset
├── labels/                         # paper-canonical label JSON defs (CSV is .gitignored)
├── scripts/
│   ├── build_labels_paper.py       # paper-canonical label build
│   ├── build_mimic_labels.py       # MIMIC 11-task labels
│   ├── build_folds.py              # strat_fold column generator
│   ├── summarize_results.py        # training-result CSV summary
│   ├── extract_predictions.py      # [bootstrap stage 1]
│   ├── bootstrap_ci.py             # [bootstrap stage 2]
│   ├── bootstrap_pairwise.py       # [bootstrap stage 3]
│   └── make_summary_table.py       # [bootstrap stage 4]
└── results/                        # experiment outputs (gitignored)
```

---

## References

* Paper: *Benchmarking ECG FMs: A Reality Check Across Clinical Tasks*
  (ICLR 2026); <https://arxiv.org/abs/2509.25095>
* Bundled paper code: [AI4HealthUOL/ECG-FM-Benchmarking](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking)
* MDS-ED: <https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/>
* MIMIC-IV-ECG-ICD labels: <https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/>
