# ECG Downstream Benchmark

H5 기반 ECG 다운스트림 태스크 벤치마크 프레임워크.
어떤 ECG encoder 모델이든 플러그인하여 Linear Probe, Attention Probe, Full Finetuning을 수행할 수 있습니다.

논문 [*Benchmarking ECG FMs: A Reality Check Across Clinical Tasks*](https://github.com/) (ICLR 2026 submission) 의 학습/평가 절차를 그대로 따릅니다 — 인코더별 input window, train시 random crop, val/test시 multi-window mean aggregation 모두 동일.

## 구조

```
benchmark/
├── run.py                         # 단일 실험 엔트리포인트
├── run_full_benchmark.sh          # 전 모델 × 전 태스크 병렬 벤치마크
├── run_parallel_tasks.sh          # 단일 모델 × 전 태스크
├── configs/
│   ├── default.yaml               # 기본 학습 설정
│   ├── models.sh                  # 모델 레지스트리 (encoder_cls / checkpoint 경로)
│   └── tasks/                     # 태스크별 데이터/라벨 설정
│       ├── ptbxl_super.yaml
│       ├── ptbxl_sub.yaml
│       ├── ptbxl_diag.yaml
│       ├── ptbxl_form.yaml
│       ├── ptbxl_rhythm.yaml
│       ├── chapman.yaml
│       ├── chapman_rhythm.yaml
│       ├── code15.yaml
│       ├── cpsc2018.yaml
│       ├── cpsc_extra.yaml
│       ├── georgia.yaml
│       ├── ningbo.yaml
│       ├── ptb.yaml
│       └── zzu_pecg.yaml
├── labels/                        # 태스크별 라벨 CSV/JSON (논문 동일 라벨셋)
├── scripts/
│   ├── build_benchmark_labels.py  # 라벨 파이프라인
│   ├── build_labels_paper.py      # 논문 동일 라벨 생성
│   └── build_folds.py             # 계층화 fold split
├── src/
│   ├── dataset.py                 # H5ECGDataset (chunk_length + ecg_id + random_crop)
│   ├── dataset_numpy.py           # EchoNextDataset (numpy + metadata.csv)
│   ├── wrapper.py                 # DownstreamWrapper (encoder-agnostic)
│   ├── heads.py                   # LinearHead, AttentionPoolingHead (heads=16), MLPHead
│   ├── trainer.py                 # DownstreamTrainer (eval시 ecg_id 평균집계)
│   ├── metrics.py                 # AUROC, AUPRC, F1
│   └── encoders/                  # 모델별 encoder adapter (chunk_seconds 선언)
└── results/                       # 실험 결과 저장 (gitignore)
```

## 지원 모델 (8개)

각 모델은 paper run.sh와 동일한 input window를 사용합니다.

| Model | input_size | model_fs | seq_len | chunks per 10s ECG |
|---|:-:|:-:|:-:|:-:|
| ECGFounder | 2.5s | 500Hz | 1250 | 4 |
| ECG-JEPA | 10.0s | 250Hz | 2500 | 1 (full) |
| ST-MEM | 2.4s | 250Hz | 600 | 4 |
| CPC | 2.5s | 240Hz | 600 | 4 |
| MERL ResNet | 2.5s | 500Hz | 1250 | 4 |
| ECGFM-KED | 10.0s | 500Hz | 5000 | 1 (full) |
| HuBERT-ECG | 5.0s | 100Hz | 500 | 2 |
| ECG-FM | 5.0s | 500Hz | 2500 | 2 |

## Multi-window 학습/평가 (paper §3.3)

ECG는 보통 10초이지만 모델 대부분이 짧은 window(2.5–5초)만 받습니다. 매 학습/평가 시 어느 구간을 보여줄지 자동으로 결정됩니다:

- **Train** (`random_crop=True`): ECG 1개당 1 sample, `__getitem__` 마다 random offset에서 chunk 추출.
  100 epoch ≈ 100가지 다른 view → augmentation 효과
- **Val/Test** (`random_crop=False`): ECG 1개를 ⌊target_length / chunk_length⌋개의 deterministic non-overlapping chunk로 확장. 각 chunk 예측을 `ecg_id`로 mean aggregate → test-time multi-view 평균
- 자동 wiring: `run.py`가 `encoder.chunk_seconds` 를 읽어 `data_cfg["chunk_length"]` 설정 + `split=='train'` 으로 `random_crop` 분기

ECG-JEPA 처럼 input_size=10s인 모델은 chunk가 1개라 multi-window 효과는 없지만 random_crop offset이 0으로 고정되어 자연스럽게 동일하게 동작합니다.

## Eval Modes

| Mode | Encoder | Head | 용도 |
|---|---|---|---|
| `linear_probe` | Frozen | Linear | 표현 품질 평가 (가장 기본) |
| `attention_probe` | Frozen | Attention Pooling | Sequence-level 표현 평가 |
| `finetune_linear` | Trainable | Linear | End-to-end finetuning |
| `finetune_attention` | Trainable | Attention Pooling | End-to-end + attention head |

Finetune 모드에서는 **discriminative learning rate** 적용:
- Head: `lr`
- Encoder: `lr × discriminative_lr_factor` (기본 0.1)

## Encoder 요구사항

```python
import torch.nn as nn
import torch.nn.functional as F

class MyEncoder(nn.Module):
    # ── 필수: output dimension ──
    feature_dim = 768

    # ── 권장: paper-aligned multi-window 활성화 ──
    chunk_seconds = 2.5     # 모델이 한 번에 보는 window 길이 (초)
    model_fs      = 500     # 모델이 기대하는 sampling rate (Hz)
    model_seq_len = 1250    # = chunk_seconds * model_fs (편의 상수)

    def __init__(self, checkpoint=None):
        super().__init__()
        # 모델 로드 + checkpoint
        ...

    def forward(self, x):
        """
        x: (batch, n_leads, T_in) — T_in 은 dataset이 chunk_length로 잘라준 길이
        encoder 내부에서 model_seq_len 으로 resample 후 처리.

        반환 형식 (택 1):
          1. tuple: (sequence_features, pooled_features)
             - sequence_features: (B, T', D) or None
             - pooled_features:   (B, feature_dim)
          2. dict:  {"seq": ..., "pooled": ...}
          3. tensor: (B, D) — pooled only
          4. tensor: (B, T', D) — GAP 자동 적용
        """
        x = torch.nan_to_num(x)
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len,
                              mode="linear", align_corners=False)
        # ... encoder forward ...
        return seq_feat, pooled

    # ── 권장: discriminative LR을 쓰려면 ──
    def get_layer_groups(self):
        """encoder를 'early' / 'late' 두 그룹으로 나눠서 layer-LR 적용.
        Trainer가 head=lr, late=lr*0.1, early=lr*0.01 로 자동 매핑."""
        early, late = [], []
        for name, p in self.named_parameters():
            if name.startswith("first_half_block"):
                early.append(p)
            else:
                late.append(p)
        return {"early": early, "late": late}
```

**`chunk_seconds` 가 없으면**: dataset chunking 안 됨, multi-window train+aggregation 비활성. 모델은 항상 `target_length` (보통 5000샘플) 전체를 받게 되니 인코더 내부에서 직접 crop/resample 해야 함.

**`chunk_seconds` 가 있으면**: dataset이 자동으로 `chunk_length = chunk_seconds * target_fs` 로 잘라서 보냄 (train: random offset, val/test: 4 chunks 평균). encoder는 잘려서 들어온 chunk를 `model_seq_len` 으로 resample만 하면 됨.

## 사용법

### 더미 인코더로 테스트

```bash
python run.py --task ptbxl_super --eval_mode linear_probe --dummy --epochs 3
```

### 커스텀 인코더

```bash
# Linear Probe
python run.py --task ptbxl_super --eval_mode linear_probe \
    --encoder_cls my_models.ECGViT --encoder_ckpt weights/vit.pt

# Attention Probe  
python run.py --task physionet_all --eval_mode attention_probe \
    --encoder_cls my_models.ECGViT --encoder_ckpt weights/vit.pt

# Full Finetuning
python run.py --task code15_diag --eval_mode finetune_linear \
    --encoder_cls my_models.ECGViT --encoder_ckpt weights/vit.pt \
    --lr 5e-4 --epochs 30

# Attention Head Finetuning
python run.py --task cpsc2021_af --eval_mode finetune_attention \
    --encoder_cls my_models.ECGViT --epochs 20
```

### CLI Override

```bash
python run.py --task ptbxl_super --eval_mode linear_probe --dummy \
    --epochs 10 --lr 3e-4 --batch_size 128 --device cuda:1
```

## 새 모델 추가하기

새 ECG 모델을 벤치마크에 paper-equivalent로 플러그인하는 절차.

### 1) Encoder adapter 작성

`src/encoders/my_model.py` 생성. 핵심은 **chunk_seconds + model_fs + model_seq_len** 3개 클래스 속성으로 paper의 input window를 선언하는 것.

```python
"""My ECG Model Encoder Adapter"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModelEncoder(nn.Module):
    """
    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, T) at data target_fs  (dataset이 chunk로 잘라줌)
      - resample to model_seq_len internally
    """

    # ── 1. paper input window 선언 ──
    chunk_seconds = 2.5     # 모델이 한 번에 보는 윈도우 길이 (초)
    model_fs      = 500     # 모델 native sampling rate (Hz)
    model_seq_len = 1250    # = chunk_seconds × model_fs

    # ── 2. feature_dim 필수 ──
    feature_dim = 768

    def __init__(self, checkpoint: str = None, **kwargs):
        super().__init__()
        # 모델 인스턴스화
        from my_pkg.model import MyModel
        self.model = MyModel(...)

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        missing, _ = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[MyModelEncoder] Missing keys: {missing[:5]}...")
        print(f"[MyModelEncoder] Loaded from {path}")

    def forward(self, x):
        x = torch.nan_to_num(x)
        # dataset이 이미 chunk_length만큼 잘라서 보내지만, 다른 fs일 수 있음
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len,
                              mode="linear", align_corners=False)

        # ... encoder forward ...
        seq_feat = ...   # (B, T', D)
        pooled   = seq_feat.mean(dim=1)
        return seq_feat, pooled

    # ── 3. (선택) Layer-dependent LR ──
    # finetune 시 head/late/early 그룹별로 lr × {1, 0.1, 0.01} 적용됨.
    # 미구현 시 head + 전체 encoder 2-그룹으로 fallback.
    def get_layer_groups(self):
        early, late = [], []
        for name, p in self.named_parameters():
            if name.startswith(("stem", "block0", "block1")):
                early.append(p)
            else:
                late.append(p)
        return {"early": early, "late": late}
```

#### 자주 빠뜨리는 포인트

- **`chunk_seconds` 안 넣으면** multi-window 학습/평가가 자동 비활성화되어 paper 결과 재현 불가
- **`model_seq_len` 와 input shape이 안 맞을 때** `F.interpolate` 로 resample 권장 (특정 모델은 fixed pos_embedding 때문에 zero-pad 필요할 수도 있음 — 그럴 땐 별도 attribute로 선언, 예: ST-MEM의 옛 방식 참고)
- ECG-JEPA 처럼 8-lead만 받는 모델은 forward 시작 부분에서 `x = x[:, lead_idx, :]` 채널 select
- BatchNorm 모델은 frozen eval 시 자동으로 BN stats freeze됨 (`DownstreamWrapper`가 처리)

### 2) `src/encoders/__init__.py`에 export

```python
from .my_model import MyModelEncoder
```

### 3) `configs/models.sh`에 등록 (run_full_benchmark.sh 용)

```bash
MODEL_CLS_MAP[my_model]="src.encoders.my_model.MyModelEncoder"
MODEL_CKPT_MAP[my_model]="/path/to/my_model.pt"
# 전체 벤치마크 기본 실행 순서에 포함하려면:
MODEL_NAMES_DEFAULT+=(my_model)
```

### 4) 단일 실행 검증 (3 epoch smoke test)

```bash
python run.py --task ptbxl_super --eval_mode linear_probe \
    --encoder_cls src.encoders.my_model.MyModelEncoder \
    --encoder_ckpt /path/to/my_model.pt --epochs 3
```

로그에서 확인할 것:
- `Multi-window enabled: chunk_seconds=2.5 → chunk_length=1250 samples` 가 찍히면 OK
- `Encoder feature_dim=768` 이 모델 출력 차원과 일치
- AUROC가 출력되면 수치는 의미 없음 (3 epoch 검증용)

### 5) 전체 벤치마크 실행

```bash
# 추가한 모델만 전 task에 대해 실행
MODELS_OVERRIDE="my_model" bash run_full_benchmark.sh

# 모든 모델 × 모든 태스크
bash run_full_benchmark.sh all
```

### 6) (선택) Paper와 정확히 같은 input spec 결정 방법

`ecg-fm-benchmarking/run.sh` 의 ARGS_MODEL 블록을 참고:

```bash
"my_model")
    "--input-size 2.5"   # → chunk_seconds = 2.5
    "--fs-model 500"     # → model_fs = 500
    "--input-channels 12"
    "--pretrained ..."
    ;;
```

`--input-size × --fs-model = model_seq_len` 으로 계산.

## 태스크 추가

`configs/tasks/my_task.yaml` 파일 생성:

```yaml
task:
  name: my_task
  num_classes: 10

data:
  h5_root: /path/to/h5/dataset
  table_csv: /path/to/ecg_table.csv
  label_csv: /path/to/labels.csv
  label_cols:
    - label_a
    - label_b
  target_fs: 500
  target_length: 5000
  normalize: true
  batch_size: 64
```

## 데이터 형식

- **H5 파일**: `convert_raw_to_h5` 파이프라인으로 생성된 표준 H5
- **Table CSV**: `ecg_table.csv` (filepath, pid, rid, fs 등)
- **Label CSV**: `{dataset}_labels.csv` (filepath + binary 라벨 컬럼)
- 라벨 CSV는 table CSV와 `filepath`로 조인

## NumPy 직접 로드 (EchoNext)

H5 변환 없이 (.npy waveforms + metadata.csv)로 배포된 데이터셋을 위한 경로.
PhysioNet EchoNext 1.1.0 (`/home/irteam/ddn-opendata1/raw/physionet.org/files/echonext/1.1.0`)에 적용.

H5 경로와 동일하게 **chunk_length + ecg_id + random_crop** 모두 지원 — 위의 "Multi-window 학습/평가" 섹션이 EchoNextDataset 에도 그대로 적용됩니다 (paper §3.3 동일). 어떤 인코더든 EchoNext 에서 multi-window train+aggregation 효과를 받습니다.

### 데이터 포맷

- `EchoNext_<split>_waveforms.npy`: shape `(N, 1, 2500, 12)` float64, 250Hz × 10s × 12-lead
  (이미 median-filter + percentile-clip + dataset-wide z-score 처리됨)
- `echonext_metadata_100k.csv`: split 컬럼 + 11개 echo binary flag + composite SHD flag
- 사전 정의된 split: `train` (72,475) / `val` (4,626) / `test` (5,442) / `no_split` (17,457)

### 제공 task

- `echonext.yaml` (= `echonext_shd`) — `shd_moderate_or_greater_flag` 단일 binary (양성률 ≈ 52% train / 43% val·test)
- `echonext_multi.yaml` — 12개 binary 라벨 동시 학습 (각 valve 중등도 이상 + LVEF/LVWT/PASP/TR 등 + composite SHD)
- `echonext_smoke.yaml` — train 자리에 val을 임시 매핑한 동작 검증용 (train .npy download 진행 중일 때)

### 인코더별 EchoNext chunk 동작 (target_length=2500 @ 250Hz)

| Model | chunk_length | n_chunks/ECG |
|---|:-:|:-:|
| ECGFounder | 625 | 4 |
| ECG-JEPA | 2500 | 1 (full) |
| ST-MEM | 600 | 4 |
| CPC | 625 | 4 |
| MERL ResNet | 625 | 4 |
| ECGFM-KED | 2500 | 1 (full) |
| HuBERT-ECG | 1250 | 2 |
| ECG-FM | 1250 | 2 |

### 사용법

```bash
# 1) Linear probe (frozen encoder)
python run.py --task echonext --eval_mode linear_probe --dummy --epochs 1

# 2) Full fine-tune
python run.py --task echonext --eval_mode finetune_linear \
    --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
    --encoder_ckpt weights/jepa.pt --epochs 30

# 3) Multi-label (12 flags)
python run.py --task echonext_multi --eval_mode attention_probe \
    --encoder_cls src.encoders.my_model.MyEncoder
```

### 새 NumPy 데이터셋 추가

`configs/tasks/<name>.yaml`에 다음 구조로 작성:

```yaml
task:
  name: <name>
  num_classes: <K>

data:
  loader_type: echonext_numpy        # NumPy 직접 로드 경로 활성
  metadata_csv: /path/to/metadata.csv
  waveforms:
    train: /path/to/<prefix>_train_waveforms.npy
    val:   /path/to/<prefix>_val_waveforms.npy
    test:  /path/to/<prefix>_test_waveforms.npy
  label_cols: [<col_a>, <col_b>, ...]
  split_col: split
  source_fs: 250                     # waveform 원본 fs
  target_fs: 250                     # 모델 fs (다르면 자동 resample)
  target_length: 2500                # crop/pad 목표 길이
  normalize: false                   # 이미 정규화된 경우
  layout: NHWC                       # (N, 1, T, C) 또는 'NCT' = (N, C, T)
  n_leads: 12
  batch_size: 64
  num_workers: 8

fold:
  auto_split: false                  # 사전 정의 split 사용
```

### 주의사항

- `waveform.npy`는 mmap으로 로드 (수십 GB 가능). row-i가 metadata CSV의 split 필터된 i번째 row와 정렬되어야 함
- DDP 학습은 H5 경로와 동일하게 `torchrun --nproc_per_node=N run.py ...` 사용
