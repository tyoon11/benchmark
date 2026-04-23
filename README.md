# ECG Downstream Benchmark

H5 기반 ECG 다운스트림 태스크 벤치마크 프레임워크.
어떤 ECG encoder 모델이든 플러그인하여 Linear Probe, Attention Probe, Full Finetuning을 수행할 수 있습니다.

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
│   ├── dataset.py                 # H5ECGDataset
│   ├── wrapper.py                 # DownstreamWrapper (encoder-agnostic)
│   ├── heads.py                   # LinearHead, AttentionPoolingHead, MLPHead
│   ├── trainer.py                 # DownstreamTrainer (학습/평가 루프)
│   ├── metrics.py                 # AUROC, AUPRC, F1
│   └── encoders/                  # 모델별 encoder adapter
└── results/                       # 실험 결과 저장 (gitignore)
```

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
class MyEncoder(nn.Module):
    feature_dim = 768  # 필수: output dimension

    def forward(self, x):
        """
        x: (batch, n_leads, seq_len)
        
        반환 형식 (택 1):
          1. tuple: (sequence_features, pooled_features)
             - sequence_features: (batch, seq_len', embed_dim) or None
             - pooled_features:   (batch, feature_dim)
          2. dict:  {"seq": ..., "pooled": ...}
          3. tensor: (batch, feature_dim) — pooled only
          4. tensor: (batch, seq_len', embed_dim) — GAP 자동 적용
        """
```

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

## 모델 추가

새 모델을 벤치마크에 플러그인하는 절차:

### 1) Encoder adapter 작성

`src/encoders/my_model.py` 생성. 요구사항은 위 "Encoder 요구사항" 참조.

```python
import torch.nn as nn

class MyModelEncoder(nn.Module):
    def __init__(self, checkpoint: str = None, **kwargs):
        super().__init__()
        self.feature_dim = 768          # 필수
        # 실제 모델 로드 + checkpoint load
        ...

    def forward(self, x):               # x: (B, 12, seq_len)
        # ...
        return seq_feat, pooled         # (B, T, D), (B, D)
```

### 2) `src/encoders/__init__.py`에 export

```python
from .my_model import MyModelEncoder
```

### 3) `configs/models.sh`에 등록 (2줄)

```bash
MODEL_CLS_MAP[my_model]="src.encoders.my_model.MyModelEncoder"
MODEL_CKPT_MAP[my_model]="/path/to/my_model.pt"
# 전체 벤치마크 기본 실행 순서에 포함하려면:
MODEL_NAMES_DEFAULT+=(my_model)
```

### 4) 단일 실행 확인

```bash
python run.py --task ptbxl_super --eval_mode linear_probe \
    --encoder_cls src.encoders.my_model.MyModelEncoder \
    --encoder_ckpt /path/to/my_model.pt --epochs 3
```

### 5) 전체 벤치마크 실행

```bash
# 추가한 모델만
MODELS_OVERRIDE="my_model" bash run_full_benchmark.sh

# 전 모델 × 전 태스크
bash run_full_benchmark.sh all
```

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
