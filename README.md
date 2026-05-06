# ECG Downstream Benchmark

ECG encoder를 paper-canonical **28개 임상 task** (17 ECG interpretation + 11 MIMIC-IV-ECG)에 plug-in해서 Linear Probe / Attention Probe / Full Finetune을 돌리는 self-contained 프레임워크.

논문 [*Benchmarking ECG FMs: A Reality Check Across Clinical Tasks*](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) 의 학습/평가 절차를 그대로 구현 — 인코더별 input window, train시 random crop augmentation, val/test시 multi-window mean aggregation, layer-dependent LR, **multi-task type (binary / multi-label-binary / regression) + NaN masking** 모두 paper와 동일.

---

## Quick start

```bash
git clone https://github.com/tyoon11/benchmark.git
cd benchmark
pip install -r requirements.txt

# Smoke test — 더미 인코더로 1 epoch
python run.py --task ptbxl_super --eval_mode linear_probe --dummy --epochs 1
```

paper 코드(`clinical_ts` subset)는 [`src/external/`](src/external/) 안에 bundled — **외부 repo clone 불필요**. 실제 학습은 (1) 사전학습 checkpoint와 (2) ECG 데이터만 환경별로 준비.

---

## What's included

### 8 encoders (paper-aligned)

| Model | input | params | Adapter |
|---|:-:|:-:|---|
| ECGFounder | 2.5s @ 500Hz | 30.7M | `src.encoders.ecg_founder.ECGFounderEncoder` |
| ECG-JEPA | 10.0s @ 250Hz | 85.4M | `src.encoders.ecg_jepa.ECGJEPAEncoder` |
| ST-MEM | 2.4s @ 250Hz | 88.5M | `src.encoders.st_mem.StMemEncoder` |
| CPC | 2.5s @ 240Hz | 3.2M | `src.encoders.cpc.CPCEncoder` |
| MERL ResNet | 2.5s @ 500Hz | 3.8M | `src.encoders.merl.MerlResNetEncoder` |
| ECGFM-KED | 10.0s @ 500Hz | 7.9M | `src.encoders.ecgfm_ked.EcgFmKEDEncoder` |
| HuBERT-ECG | 5.0s @ 100Hz | 93.1M | `src.encoders.hubert_ecg.HuBERTECGEncoder` |
| ECG-FM | 5.0s @ 500Hz | 90.4M | `src.encoders.ecg_fm.ECGFMEncoder` |

### 28 paper-canonical tasks + 7 variants

```
Adult ECG interpretation:    ptb, ningbo, cpsc2018, cpsc_extra, georgia,
                             chapman, chapman_rhythm, code15, sph_diag,
                             ptbxl_{all, super, sub, diag, form, rhythm}
Pediatric ECG interp:        zzu_pecg
Cardiac structure & func:    echonext              (NumPy loader)

MIMIC-IV-ECG (11 tasks):
  Discharge diagnoses:       mimic_cardiac, mimic_noncardiac
  Patient characteristics:   mimic_sex, mimic_age
  ECG features:              mimic_ecg_features    (regression × 7)
  Acute care (MDS-ED):       mimic_deterioration, mimic_mortality, mimic_icu_admission
  Biometrics/Vitals/Labs:    mimic_biometrics, mimic_vitals, mimic_labvalues  (regression)

Variants:                    code15_diag, code15_diag_jepa, cpsc2021_af,
                             physionet_all, ptbxl_super_jepa
```

Task 정의는 [`configs/tasks/*.yaml`](configs/tasks/). MIMIC 라벨 생성은 아래 ["MIMIC label build"](#mimic-label-build) 참조.

### Task types (paper main_lite_ecg.py:92-139 재현)

| `task_type` | Loss | Eval metric | NaN 처리 | 사용 task 예 |
|---|---|---|---|---|
| `binary` (default) | BCEWithLogits | AUROC / AUPRC / F1 | NaN→0 (negative) | ptbxl_*, chapman, sex, cardiac, … |
| `multi-label-binary` | masked BCE | AUROC / AUPRC / F1 | NaN 마스킹 (paper:114) | mortality, deterioration, icu_admission |
| `regression` | masked L1 (MAE) | MAE / MSE / RMSE / R² / neg_MAE | NaN 마스킹 (paper:128) | age, ecg_features, biometrics, vitals, labvalues |

`task.task_type` 을 task yaml에 명시. 미지정 시 `binary` 적용.

### 4 eval modes

| Mode | Encoder | Head | 용도 |
|---|---|---|---|
| `linear_probe` | Frozen | Linear | 표현 품질 평가 (기본) |
| `attention_probe` | Frozen | V-JEPA Learnable Query Attention Pool | Sequence-level 표현 평가 |
| `finetune_linear` | Trainable | Linear | End-to-end finetune |
| `finetune_attention` | Trainable | V-JEPA Attention Pool | E2E + attention head |

Finetune 모드는 layer-dependent LR 자동 적용: `head=lr`, `late=lr × 0.1`, `early=lr × 0.01`.

### Multi-window train + test-time aggregation (paper §3.3)

ECG는 보통 10초인데 모델은 짧은 window(2.5–5초)만 받음. 어느 구간을 보여줄지 자동 결정:

| Split | 동작 |
|---|---|
| **Train** | ECG 1개당 1 sample, `__getitem__`마다 random offset에서 chunk 추출. 100 epoch ≈ 100가지 view augmentation |
| **Val/Test** | ECG 1개를 ⌊target_length / chunk_length⌋ 개 deterministic non-overlapping chunk로 확장. ECG ID로 mean aggregate → multi-view 평균 |

`run.py`가 `encoder.chunk_seconds`를 읽어 자동 wiring (`split=='train'`이면 random crop, 아니면 deterministic chunks). 별도 설정 불필요.

---

## 사용법

### 단일 실험

```bash
# Linear probe
python run.py --task ptbxl_super --eval_mode linear_probe \
    --encoder_cls src.encoders.ecg_founder.ECGFounderEncoder \
    --encoder_ckpt /path/to/ckpt.pth

# Full finetune (lower LR)
python run.py --task code15 --eval_mode finetune_linear \
    --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
    --encoder_ckpt /path/to/jepa.pth --lr 5e-4 --epochs 30

# 더미 인코더 — 외부 의존성 없이 파이프라인만 검증
python run.py --task ptbxl_super --eval_mode linear_probe --dummy --epochs 1
```

CLI override: `--epochs`, `--lr`, `--batch_size`, `--device`, `--save_dir`, `--train_folds` 등.

### 전체 벤치마크

```bash
# 전 모델 × 전 태스크 × 전 모드 (병렬 GPU 자동 분배)
bash run_full_benchmark.sh all

# 특정 모델만
MODELS_OVERRIDE="ecg_founder ecg_jepa" bash run_full_benchmark.sh

# 특정 태스크만
TASKS_OVERRIDE="ptbxl_super echonext" bash run_full_benchmark.sh
```

결과는 `results/<timestamp>/` 아래 task별 디렉토리 + `results_all.csv` 누적.

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 run.py --task ptbxl_super --eval_mode finetune_linear \
    --encoder_cls src.encoders.ecg_founder.ECGFounderEncoder \
    --encoder_ckpt /path/to/ckpt.pth
```

---

## 통계 평가 (Bootstrap CI + 동순위 ranking)

Paper §3.5 평가 절차 그대로:
- **분류 1차 지표**: macro-averaged AUROC (↑)
- **회귀 1차 지표**: z-normalized MAE (↓)
- **유의차**: 테스트 세트 경험적 부트스트랩 (n=1000), 95% 신뢰구간이 0을 포함하지 않으면 유의
- **동순위(tied)**: pairwise diff CI가 0 포함 → 통계적 유의차 없음

학습 단계는 집계 metric만 저장하므로, **`best.pt`를 다시 로드해서 test 추론을 한 번 더** 돌려 sample-level 예측을 추출한 뒤 부트스트랩.

### 한 번에 실행

```bash
# default — N_ITERS=1000, WORKERS=$(nproc), GPU 0 만 사용
bash run_bootstrap.sh /path/to/results/<timestamp>

# 4 GPU 병렬 + 32 CPU worker
WORKERS=32 bash run_bootstrap.sh /path/to/results/<timestamp> "0,1,2,3"

# 일부만
FILTER=cpc bash run_bootstrap.sh /path/to/results/<timestamp> "0"

# 추출 끝났으면 부트스트랩만 (CPU only)
SKIP_EXTRACT=1 bash run_bootstrap.sh /path/to/results/<timestamp>
```

### 4-stage 파이프라인

각 단계는 개별 스크립트로도 실행 가능 (`scripts/`).

| 단계 | 스크립트 | 역할 |
|---|---|---|
| 1. **추출** | [`extract_predictions.py`](scripts/extract_predictions.py) | 각 `<model>_<task>_<mode>/best.pt` 로드 → test 추론 → `preds.npy`, `targets.npy`, `ids.npy`, `preds_meta.json` 저장. multi-window aggregation, regression z-norm, auto fold split 모두 학습 시와 동일 |
| 2. **단일-모델 CI** | [`bootstrap_ci.py`](scripts/bootstrap_ci.py) | n=1000 경험적 부트스트랩 (paper `clinical_ts.utils.bootstrap_utils.empirical_bootstrap` 동일: `point + percentile(scores - point, 2.5/97.5)`). `bootstrap.json` + 합본 `bootstrap_summary.csv` |
| 3. **Pairwise + 동순위** | [`bootstrap_pairwise.py`](scripts/bootstrap_pairwise.py) | (task, mode)별 모델 pair 모두에 대해 **공유 부트스트랩 인덱스**로 paired diff CI. union-find로 동순위 그룹화. `pairwise_diff_<task>_<mode>.csv`, `tied_groups_<task>_<mode>.txt`, `pairwise_summary.csv` |
| 4. **Paper-style 표** | [`make_summary_table.py`](scripts/make_summary_table.py) | (task × model) pivot — **bold** = 단독 best, __underline__ = best와 통계적 동순위. `summary_<mode>.csv` (raw), `summary_<mode>_marked.csv`, `summary_<mode>.md`, `summary_ci_long.csv` |

### 산출물 구조

```
results/<timestamp>/
├── <model>_<task>_<mode>/
│   ├── best.pt, test_metrics.txt        (학습 단계 산출물)
│   ├── preds.npy        (N, C)          ← 1단계
│   ├── targets.npy      (N, C)
│   ├── ids.npy          (N,)            ← pairwise alignment 키
│   ├── preds_meta.json
│   └── bootstrap.json                   ← 2단계 — point + 95% CI
│
├── bootstrap_summary.csv                ← 모든 (model, task, mode) point + CI 합본
└── pairwise/                            ← 3·4단계
    ├── pairwise_diff_<task>_<mode>.csv  모델 pair별 diff CI + significant flag
    ├── tied_groups_<task>_<mode>.txt    rank 그룹 (Rank 1: A=0.95  B=0.94  C=0.93 …)
    ├── pairwise_summary.csv             task / mode / model / score / rank
    │
    ├── summary_<mode>.csv               논문 Table 1 형식 raw 점수 (Category × Task × Model)
    ├── summary_<mode>_marked.csv        **best** / __tied__ 마킹된 CSV
    ├── summary_<mode>.md                Markdown 표 (그대로 viewer 렌더링)
    └── summary_ci_long.csv              "0.946 [0.928, 0.962]" long-format
```

### Markdown 표 예시 (`summary_<mode>.md`)

```
| Category  | Task              | ECGFounder | ECG-JEPA  | ST-MEM    | MERL      | … |
|-----------|-------------------|------------|-----------|-----------|-----------|---|
| Adult ECG | PTB ↑             | 0.780      | 0.816     | __0.862__ | 0.624     | … |
| Adult ECG | Ningbo ↑          | 0.966      | 0.965     | **0.971** | 0.962     | … |
| Adult ECG | PTB-XL (super) ↑  | 0.928      | 0.921     | 0.937     | __0.942__ | … |
| Ped. ECG  | ZZU pECG ↑        | 0.905      | __0.918__ | __0.918__ | 0.896     | … |
```

`↑` macro-AUROC (분류) / `↓` z-norm MAE (회귀). **bold** = 점추정 단독 best. __underline__ = best와 paired bootstrap 95% CI가 0 포함 (통계적 동순위).

### 시간 예상 (475 dirs, n=1000)

| 단계 | 단일 코어 / 1 GPU | 4 GPU + 64 CPU worker |
|---|---|---|
| 1. 추출 (GPU bound) | ~6 시간 | **~1.5 시간** |
| 2. CI | ~40 분 | ~3 분 |
| 3. Pairwise | ~5 시간 | ~10 분 |
| 4. 표 생성 | <10 초 | <10 초 |
| **합계** | ~12 시간 | **~2 시간** |

추출 후엔 raw `preds.npy`/`targets.npy`만 있으면 부트스트랩은 GPU 없이 언제든 다시 돌릴 수 있어 — `n_iters` 늘리거나 다른 metric 추가할 때 1·4단계만 재실행.

---

## 새 모델 추가하기

`src/encoders/my_model.py` 생성. 핵심은 **3개 클래스 속성**으로 paper input window를 선언하는 것:

```python
import sys, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

# bundled clinical_ts backbone을 쓰려면:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "external"))


class MyModelEncoder(nn.Module):
    # ── 1. paper input window (필수: paper run.sh와 동일하게) ──
    chunk_seconds = 2.5     # paper run.sh 의 --input-size
    model_fs      = 500     # paper run.sh 의 --fs-model
    model_seq_len = 1250    # = chunk_seconds × model_fs

    # ── 2. encoder output dim (필수) ──
    feature_dim = 768

    def __init__(self, checkpoint=None):
        super().__init__()
        self.model = ...   # backbone 인스턴스화
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)

    def forward(self, x):
        """x: (B, 12, T)  →  (sequence_features (B,T',D), pooled (B,D))"""
        x = torch.nan_to_num(x)
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len,
                              mode="linear", align_corners=False)
        seq_feat = self.model(x)
        pooled   = seq_feat.mean(dim=1)
        return seq_feat, pooled

    # ── 3. (선택) layer-dependent LR ──
    # finetune 시 head/late/early 그룹별로 lr × {1, 0.1, 0.01}.
    # 미구현 시 head + 전체 encoder 2-그룹 fallback.
    def get_layer_groups(self):
        early, late = [], []
        for name, p in self.named_parameters():
            if name.startswith(("stem", "block0", "block1")):
                early.append(p)
            else:
                late.append(p)
        return {"early": early, "late": late}
```

**forward 출력은 4가지 형식 지원** (`DownstreamWrapper`가 자동 파싱):
1. tuple `(seq_feat, pooled)` — 권장
2. dict `{"seq": ..., "pooled": ...}`
3. tensor `(B, D)` — pooled only
4. tensor `(B, T, D)` — seq only (자동 GAP 적용)

### 자주 빠뜨리는 포인트

- **`chunk_seconds` 안 넣으면** multi-window 비활성 → paper 결과 재현 불가
- **8-lead 모델**은 forward에서 `x = x[:, lead_idx, :]` 채널 select (예: ECG-JEPA)
- **BatchNorm 모델**은 frozen eval시 `DownstreamWrapper`가 BN stats 자동 freeze
- **Fixed pos_embedding** 모델은 zero-pad 필요할 수 있음
- **Pretrained backbone wrapper의 추가 projection 누락 주의** — paper의 wrapper class
  (예: `S4Predictor`, `RNNEncoder`)가 내부에서 raw 모델을 wrap 하면서 input projection
  을 추가/skip 하는 경우, raw 모델만 가져오면 random-init Linear/Conv 1개가 끼어
  feature가 corrupt 됨 (실측: CPC 0.78 vs paper 0.88 — 한 줄 차이 때문). 어댑터
  작성 시 paper wrapper의 forward 흐름 그대로 재현하거나 wrapper class 자체를 import.

### Paper와 공정 비교 contract (새 모델 추가 시)

새 모델이 paper의 8개 모델과 **같은 조건에서 평가**되도록 보장하는 항목:

| 항목 | 자동 보장 | 사용자가 신경쓸 부분 |
|---|---|---|
| **데이터 split** | `strat_fold` 기반 자동 split (paper 동일) | 새 task 만들 땐 `strat_fold` 컬럼 포함 |
| **라벨셋** | `labels/` 안의 paper-canonical 라벨 자동 사용 | task yaml의 `label_csv` 만 잘 지정 |
| **Optimizer/Schedule** | AdamW + lr=1e-3 + const + 100 epoch (paper 동일) | 다른 모델만 다른 lr 쓰면 부정 비교 — default 유지 |
| **Loss** | task_type별 자동: BCE / masked BCE / masked L1 (paper 동일) | 인코더 출력 dim이 task `num_classes`와 일치만 보장 |
| **Multi-window train+agg** | `chunk_seconds` 선언만 하면 자동 (paper §3.3) | **반드시 paper run.sh와 동일한 input_size × fs_model** |
| **Layer-LR (finetune)** | head=lr, late=0.1lr, early=0.01lr (paper 동일) | `get_layer_groups()` 미구현시 head + 전체 2그룹 fallback |
| **Eval modes (4)** | linear_probe / attention_probe / finetune_linear / finetune_attention | head는 framework 가 동일 (V-JEPA learnable query, heads=16) |
| **Test-time aggregation** | non-overlapping chunks 평균집계 (paper 동일) | 자동 |
| **Frozen eval 시 BN** | `DownstreamWrapper`가 자동 freeze (paper 동일) | 자동 |

**공정성 위반 시그널** (이게 보이면 비교가 부정확):
- 새 모델만 다른 epoch / lr / batch_size 쓰는 경우
- Pretrain 데이터셋이 평가 task의 train set과 겹치는데 데이터 leak 안 막은 경우 (paper 모델도 일부 그러므로 새 모델만 별도 corrigendum 필요)
- 새 모델이 normalize=true 를 강제하는데 task yaml 수정 안 한 경우
- `chunk_seconds`를 paper 보다 작게 설정 (input window 줄어 unfair advantage 또는 disadvantage)
- forward 안에서 추가 augmentation/dropout 등 paper에 없는 것 적용

### 등록 (3 줄)

[`src/encoders/__init__.py`](src/encoders/__init__.py):
```python
from .my_model import MyModelEncoder
```

[`configs/models.sh`](configs/models.sh):
```bash
MODEL_CLS_MAP[my_model]="src.encoders.my_model.MyModelEncoder"
MODEL_CKPT_MAP[my_model]="/path/to/ckpt.pt"
MODEL_NAMES_DEFAULT+=(my_model)
```

### Smoke test → 전체 벤치마크

```bash
# 3 epoch 검증
python run.py --task ptbxl_super --eval_mode linear_probe \
    --encoder_cls src.encoders.my_model.MyModelEncoder \
    --encoder_ckpt /path/to/ckpt.pt --epochs 3
```

로그에 `Multi-window enabled: chunk_seconds=2.5 → chunk_length=1250 samples` 가 찍히면 paper-aligned 동작 OK. 정상이면:

```bash
MODELS_OVERRIDE="my_model" bash run_full_benchmark.sh all
```

---

## 새 태스크 추가하기

### H5 task

```yaml
# configs/tasks/my_task.yaml
task:
  name: my_task
  num_classes: 10

data:
  h5_root:    /path/to/h5/dataset
  table_csv:  /path/to/ecg_table.csv     # filepath, pid, fs, strat_fold
  label_csv:  /path/to/labels.csv         # filepath + binary label cols
  label_cols: [label_a, label_b, ...]
  target_fs:     500
  target_length: 5000                    # 보통 10s ECG (encoder가 chunk로 자름)
  normalize:     false
  batch_size:    64
```

`fold.auto_split: true` 면 `strat_fold` 컬럼 max-1=val, max=test 자동 분기. CLI `--{train,val,test}_folds` 로 override 가능.

### NumPy task (EchoNext-style)

```yaml
data:
  loader_type: echonext_numpy            # NumPy 직접 로드 활성
  metadata_csv: /path/to/metadata.csv
  waveforms:
    train: /path/to/<prefix>_train_waveforms.npy
    val:   /path/to/<prefix>_val_waveforms.npy
    test:  /path/to/<prefix>_test_waveforms.npy
  label_cols:    [...]
  split_col:     split
  source_fs:     250
  target_fs:     250
  target_length: 2500
  layout:        NHWC                    # (N, 1, T, C) — or "NCT" for (N, C, T)
  n_leads:       12
  normalize:     false                   # 이미 정규화된 경우

fold:
  auto_split: false                      # split_col이 직접 분기
```

`waveforms[split].npy`의 row-i는 `metadata_csv` split 필터된 i번째 row와 정렬돼야 함. `.npy` 는 mmap 로드 — 수십 GB OK.

---

## 데이터 준비

데이터 / 체크포인트 경로는 task yaml 마다 흩어진 게 아니라 **환경변수 2개**로 한 번에 지정합니다.

### 환경변수

```bash
# 1) ECG 데이터 root  (기본: /home/irteam/ddn-opendata1)
export ECG_DATA_ROOT=/your/data/root

# 2) 사전학습 checkpoint root  (기본: /home/irteam/ddn-opendata1/model/ECGFMs)
export ECG_CKPT_ROOT=/your/ckpt/root
```

설정 안 하면 원래 서버의 절대경로로 default fallback (이 repo가 만들어진 환경 backward-compat). 다른 서버에서 clone한 경우 위 2개만 export하면 됩니다.

### 디렉토리 구조 (env var 기준)

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

각 task yaml의 `h5_root`/`table_csv`/`metadata_csv`/`waveforms`는 `${ECG_DATA_ROOT}/...` 형태로 작성돼 있어 `run.py`가 자동 expand. 체크포인트 경로는 `configs/models.sh` + `run_parallel_tasks.sh`에서 `${ECG_CKPT_ROOT:-...}` 형태로 자동 expand.

> **Note**: 디렉토리 구조 자체가 다르면 (예: H5가 `~/data/h5/...`처럼 다른 위치) `configs/tasks/*.yaml`의 `${ECG_DATA_ROOT}/h5/...` 부분을 본인 환경에 맞게 직접 수정하면 됩니다.

### Pretrained checkpoints (다운로드 URL)

| Model | URL |
|---|---|
| ECGFounder | https://huggingface.co/PKUDigitalHealth/ECGFounder |
| ECG-JEPA (multiblock) | https://drive.google.com/file/d/1mh-XL0XOvvhFbhvuZ9c2KnTHa9B4F3Wx |
| ST-MEM | https://drive.google.com/file/d/1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ |
| MERL ResNet | https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2 |
| ECGFM-KED | https://zenodo.org/records/14881564 |
| HuBERT-ECG / ECG-FM / CPC | paper [`ecg-fm-benchmarking`](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) |

### ECG 데이터셋

- **H5**: paper의 `convert_raw_to_h5` 파이프라인 결과 — `ECG/metadata.fs` + `ECG/segments/<i>/signal` 구조
- **NumPy** (EchoNext): `(N, 1, T, C)` shape `.npy` + metadata CSV
- **라벨**: paper-canonical 라벨 정의가 [`labels/`](labels/) 안에 미리 들어있음 (csv + json)

---

## MIMIC label build

MIMIC-IV-ECG 11개 task는 raw 데이터(PhysioNet credentialed) 로부터 [`scripts/build_mimic_labels.py`](scripts/build_mimic_labels.py)로 생성. 원본 [`mimic_preprocessing.py`](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking/blob/main/mimic_preprocessing.py) 1:1 재현.

### 필요한 raw 파일 (11개)

| 데이터셋 | 페이지 | 파일 |
|---|---|---|
| MIMIC-IV-ECG (1.0) | https://physionet.org/content/mimic-iv-ecg/1.0/ | `machine_measurements.csv`, `record_list.csv` |
| MIMIC-IV-ECG-ICD (1.0.1) | https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/ | `records_w_diag_icd10.csv` |
| MIMIC-IV-ED (2.2) | https://physionet.org/content/mimic-iv-ed/ | `ed/edstays.csv.gz`, `ed/vitalsign.csv.gz` |
| MIMIC-IV (3.1) hosp/ | https://physionet.org/content/mimiciv/3.1/ | `admissions.csv.gz`, `omr.csv.gz`, `labevents.csv.gz`, `d_labitems.csv.gz` |
| MIMIC-IV (3.1) icu/ | https://physionet.org/content/mimiciv/3.1/ | `chartevents.csv.gz`, `d_items.csv.gz`, `icustays.csv.gz` |
| MDS-ED (1.0.0) | https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/ | `mds_ed.csv` |

배치 위치: `$ECG_DATA_ROOT/raw/physionet.org/files/<dataset>/...` (build_mimic_labels.py 상단 경로 참조).

### 병렬 빌드 (3-stage)

```bash
./run_build_mimic_labels.sh
```

Stage 1 (병렬, ~2분): diagnostic, sex, ecg_features, deterioration, mortality, icu_admission  
Stage 2 (단독, ~40분): biometrics — chartevents.csv.gz (~30GB) 청크 필터 + 캐시 생성  
Stage 3 (병렬, ~15분): vitals + labvalues (캐시 재사용)

전체 ~1시간. 각 task 로그 → `labels/_logs/build_<task>.log`.

### 결과

```
labels/
├── mimic_cardiac_paper_labels.csv             (114k rows × 158 labels — paper Table 99.7%↑ match)
├── mimic_noncardiac_paper_labels.csv          (178k × 918)
├── mimic_sex_paper_labels.csv                 (binary)
├── mimic_age_paper_labels.csv                 (regression)
├── mimic_ecg_features_paper_labels.csv        (regression × 7)
├── mimic_deterioration_paper_labels.csv       (multi-label-binary × 6)
├── mimic_mortality_paper_labels.csv           (multi-label-binary × 7)
├── mimic_icu_admission_paper_labels.csv       (multi-label-binary × 2)
├── mimic_biometrics_paper_labels.csv          (regression × 3)
├── mimic_vitals_paper_labels.csv              (regression × 6)
└── mimic_labvalues_paper_labels.csv           (regression × 18)
```

---

## Project layout

```
benchmark/
├── run.py                          # 단일 실험 entrypoint
├── run_full_benchmark.sh           # 전 모델 × 전 태스크 × 전 모드 병렬
├── run_parallel_tasks.sh           # 단일 모델 × 전 태스크
├── run_build_mimic_labels.sh       # MIMIC 11개 task 라벨 3-stage 병렬 빌드
├── run_bootstrap.sh                # 부트스트랩 평가 4-stage 오케스트레이터
├── configs/
│   ├── default.yaml                # 기본 학습 설정 (lr, epochs, head)
│   ├── models.sh                   # 모델 레지스트리
│   └── tasks/                      # 35+개 태스크 yaml (paper 28 + variants 7)
├── src/
│   ├── dataset.py                  # H5ECGDataset (task_type 분기, NaN 보존)
│   ├── dataset_numpy.py            # EchoNextDataset
│   ├── wrapper.py                  # DownstreamWrapper (encoder-agnostic)
│   ├── heads.py                    # Linear / V-JEPA Attention / MLP heads
│   ├── trainer.py                  # BCE / masked-BCE / masked-L1 자동 분기
│   ├── metrics.py                  # AUROC / AUPRC / F1 + MAE / MSE / RMSE / R²
│   ├── encoders/                   # 8 encoder adapters
│   └── external/clinical_ts/       # paper backbone subset (bundled)
├── labels/                         # paper-canonical 라벨 정의 (csv + json)
├── scripts/
│   ├── build_labels_paper.py       # paper-canonical 라벨 빌드
│   ├── build_mimic_labels.py       # MIMIC 11개 task 라벨 빌드
│   ├── build_folds.py              # strat_fold 컬럼 생성
│   ├── summarize_results.py        # 학습 결과 CSV 요약
│   ├── extract_predictions.py      # [bootstrap 1단계] best.pt → test 추론 → preds.npy
│   ├── bootstrap_ci.py             # [bootstrap 2단계] n=1000 단일-모델 95% CI
│   ├── bootstrap_pairwise.py       # [bootstrap 3단계] paired diff CI + tied-rank
│   └── make_summary_table.py       # [bootstrap 4단계] paper-style table (CSV/MD)
└── results/                        # 실험 결과 (gitignore)
```

---

## References

- 논문: *Benchmarking ECG FMs: A Reality Check Across Clinical Tasks* (ICLR 2026)
- Bundled paper code: [AI4HealthUOL/ECG-FM-Benchmarking](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking)
- MDS-ED dataset (Multimodal Decision Support — Emergency Department): https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/
- MIMIC-IV-ECG-ICD labels: https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/
