# ECG Downstream Benchmark

ECG encoder를 paper-canonical 17개 임상 task에 plug-in해서 Linear Probe / Attention Probe / Full Finetune을 돌리는 self-contained 프레임워크.

논문 [*Benchmarking ECG FMs: A Reality Check Across Clinical Tasks*](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) 의 학습/평가 절차를 그대로 구현 — 인코더별 input window, train시 random crop augmentation, val/test시 multi-window mean aggregation, layer-dependent LR 모두 동일.

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

### 17 paper-canonical tasks + 7 variants

```
Adult ECG interpretation:    ptb, ningbo, cpsc2018, cpsc_extra, georgia,
                             chapman, chapman_rhythm, code15, sph_diag,
                             ptbxl_{all, super, sub, diag, form, rhythm}
Pediatric ECG interp:        zzu_pecg
Cardiac structure & func:    echonext              (NumPy loader)

Variants:                    code15_diag, code15_diag_jepa, cpsc2021_af,
                             physionet_all, ptbxl_super_jepa,
                             echonext_multi, echonext_smoke
```

Task 정의는 [`configs/tasks/*.yaml`](configs/tasks/).

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

### Pretrained checkpoints

| Model | URL |
|---|---|
| ECGFounder | https://huggingface.co/PKUDigitalHealth/ECGFounder |
| ECG-JEPA (multiblock) | https://drive.google.com/file/d/1mh-XL0XOvvhFbhvuZ9c2KnTHa9B4F3Wx |
| ST-MEM | https://drive.google.com/file/d/1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ |
| MERL | https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2 |
| ECGFM-KED | https://zenodo.org/records/14881564 |
| HuBERT-ECG | (paper repo) |
| ECG-FM | paper의 `mimic_iv_ecg_physionet_pretrained.pt` |
| CPC | paper [`ecg-fm-benchmarking`](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking) |

다운로드 후 `configs/models.sh` + `run_parallel_tasks.sh` 의 `MODEL_CKPT_MAP` / `MODEL_CKPT` 경로를 본인 환경에 맞게 수정.

### ECG 데이터셋

- **H5 데이터**: paper의 `convert_raw_to_h5` 파이프라인으로 변환된 표준 H5 (PTB-XL, CODE-15, Chapman 등). `ECG/metadata.fs` + `ECG/segments/<i>/signal` 구조.
- **NumPy 데이터** (EchoNext): `(N, 1, T, C)` shape `.npy` + metadata CSV.
- **라벨**: paper-canonical 라벨 정의가 [`labels/`](labels/) 안에 미리 들어있음 (csv + json).

데이터 경로는 [`configs/tasks/<task>.yaml`](configs/tasks/) 의 `h5_root` 또는 `waveforms` 를 본인 환경에 맞게 수정.

---

## Project layout

```
benchmark/
├── run.py                          # 단일 실험 entrypoint
├── run_full_benchmark.sh           # 전 모델 × 전 태스크 × 전 모드 병렬
├── run_parallel_tasks.sh           # 단일 모델 × 전 태스크
├── configs/
│   ├── default.yaml                # 기본 학습 설정 (lr, epochs, head)
│   ├── models.sh                   # 모델 레지스트리
│   └── tasks/                      # 24개 태스크 yaml (paper 17 + variants 7)
├── src/
│   ├── dataset.py                  # H5ECGDataset (chunk + ecg_id + random_crop)
│   ├── dataset_numpy.py            # EchoNextDataset
│   ├── wrapper.py                  # DownstreamWrapper (encoder-agnostic)
│   ├── heads.py                    # Linear / V-JEPA Attention / MLP heads
│   ├── trainer.py                  # eval시 ecg_id 평균집계
│   ├── metrics.py                  # AUROC / AUPRC / F1
│   ├── encoders/                   # 8 encoder adapters
│   └── external/clinical_ts/       # paper backbone subset (bundled)
├── labels/                         # paper-canonical 라벨 정의 (csv + json)
├── scripts/                        # 라벨/fold 빌드 + UMAP
└── results/                        # 실험 결과 (gitignore)
```

---

## References

- 논문: *Benchmarking ECG FMs: A Reality Check Across Clinical Tasks* (ICLR 2026 submission)
- Bundled paper code: [AI4HealthUOL/ECG-FM-Benchmarking](https://github.com/AI4HealthUOL/ECG-FM-Benchmarking)
