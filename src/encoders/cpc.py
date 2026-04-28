"""
CPC Encoder Adapter
====================
Model sampling frequency: 240 Hz
Embedding dimension: 512

Hydra/Lightning 의존성 없이 체크포인트에서 직접 encoder + predictor를 로드합니다.
구조:
  - Encoder: 4층 Conv1d (12→512, stride=[2,1,1,1], ks=[3,1,1,1]) + BatchNorm + ReLU
  - Predictor: 4층 S4 (state-space model, dim=512)
임베딩에는 encoder + predictor 전체 출력을 사용합니다.
"""

import os
import sys
import ctypes.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ECG_FM_BENCH = Path(os.environ.get(
    "ECG_FM_BENCH_DIR",
    "/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code",
))
sys.path.insert(0, str(ECG_FM_BENCH))


def _prepend_env_path(name: str, value: Path) -> None:
    if not value.exists():
        return
    current = os.environ.get(name, "")
    parts = [p for p in current.split(":") if p]
    value_str = str(value)
    if value_str not in parts:
        os.environ[name] = ":".join([value_str, *parts]) if parts else value_str


def _configure_s4_runtime() -> None:
    """Expose CUDA toolkit and compilers so PyKeOps can build its NVRTC backend."""
    if getattr(_configure_s4_runtime, "_done", False):
        return

    env_prefix = Path(sys.executable).resolve().parents[1]
    bin_dir = env_prefix / "bin"
    lib_dir = env_prefix / "lib"
    cuda_root = env_prefix / "targets" / "x86_64-linux"

    gxx = bin_dir / "x86_64-conda-linux-gnu-g++"
    gcc = bin_dir / "x86_64-conda-linux-gnu-gcc"
    if gxx.exists():
        os.environ.setdefault("CXX", str(gxx))
    if gcc.exists():
        os.environ.setdefault("CC", str(gcc))

    _prepend_env_path("PATH", bin_dir)
    _prepend_env_path("LD_LIBRARY_PATH", lib_dir)

    # KeOps expects CUDA_PATH/include/{cuda.h,nvrtc.h}. In this conda layout,
    # those headers live under targets/x86_64-linux/include.
    if (cuda_root / "include" / "cuda.h").exists() and (cuda_root / "include" / "nvrtc.h").exists():
        os.environ.setdefault("CUDA_PATH", str(cuda_root))
        os.environ.setdefault("CUDA_HOME", str(cuda_root))

    if not getattr(ctypes.util, "_ecgfm_find_library_patched", False):
        real_find_library = ctypes.util.find_library
        lib_map = {
            "nvrtc": lib_dir / "libnvrtc.so",
            "cudart": lib_dir / "libcudart.so",
        }

        def _patched_find_library(name: str):
            candidate = lib_map.get(name)
            if candidate is not None and candidate.exists():
                return str(candidate)
            return real_find_library(name)

        ctypes.util.find_library = _patched_find_library
        ctypes.util._ecgfm_find_library_patched = True

    _configure_s4_runtime._done = True


_configure_s4_runtime()


def _build_conv_block(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv1d + BatchNorm + ReLU (CPC encoder 기본 블록)"""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                  padding=(kernel_size - 1) // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class CPCEncoder(nn.Module):
    """
    CPC encoder wrapper (Hydra-free).

    체크포인트에서 encoder (Conv1d) 와 predictor (S4) weights를 직접 로드합니다.
    S4 predictor 로드에 실패하면 encoder-only로 fallback합니다.

    forward(x) → (sequence_features, pooled_features)
      - x: (B, 12, T) at data target_fs → 600 samples (2.5s @ 240Hz)
      - pooled_features: (B, 512)
    """

    # Paper: input_size=2.5s, fs_model=240 → 600 samples per window.
    chunk_seconds = 2.5
    model_fs = 240
    model_seq_len = 600

    def __init__(self, checkpoint=None, config_path=None):
        super().__init__()
        self.feature_dim = 512
        self._has_predictor = False

        # ── Encoder: 4-layer Conv1d ──
        # Config: features=[512,512,512,512], kss=[3,1,1,1], strides=[2,1,1,1]
        self.encoder = nn.Sequential(
            _build_conv_block(12, 512, kernel_size=3, stride=2),   # layer 0
            _build_conv_block(512, 512, kernel_size=1, stride=1),  # layer 1
            _build_conv_block(512, 512, kernel_size=1, stride=1),  # layer 2
            _build_conv_block(512, 512, kernel_size=1, stride=1),  # layer 3
        )

        # ── Predictor: S4 (try to load, optional) ──
        try:
            from clinical_ts.ts.s4_modules.s4_model import S4Model
            self.predictor = S4Model(
                d_input=512,
                d_model=512,
                d_output=None,        # no output head
                d_state=8,            # state_dim from config
                n_layers=4,
                dropout=0.2,
                tie_dropout=True,
                prenorm=False,
                l_max=1200,             # matches checkpoint omega size: (1200//2)+1=601
                transposed_input=True,  # input is (B, D, L)
                bidirectional=False,    # causal=True
                layer_norm=True,        # not batchnorm
                pooling=False,          # keep sequence
                backbone="s42",
            )
            self._has_predictor = True
            print("[CPCEncoder] S4 predictor loaded")
        except Exception as e:
            print(f"[CPCEncoder] S4 predictor 사용 불가 (encoder-only fallback): {e}")
            self.predictor = None

        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        # ── Load encoder weights ──
        # checkpoint: ts_encoder.encoder.layers.X.Y → model: X.Y
        enc_state = {}
        for k, v in state.items():
            if k.startswith("ts_encoder.encoder.layers."):
                new_key = k.replace("ts_encoder.encoder.layers.", "")
                enc_state[new_key] = v

        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        if missing:
            print(f"[CPCEncoder] Encoder missing keys: {missing[:3]}...")
        print(f"[CPCEncoder] Encoder loaded ({len(enc_state)} keys)")

        # ── Load predictor weights ──
        if self._has_predictor and self.predictor is not None:
            pred_state = {}
            for k, v in state.items():
                if k.startswith("ts_encoder.predictor.predictor."):
                    new_key = k.replace("ts_encoder.predictor.predictor.", "")
                    pred_state[new_key] = v.clone()  # clone to avoid shared memory issues

            if pred_state:
                # S4 DPLR init creates shared-memory params (B, P, w are views).
                # Must manually assign each param to avoid the shared memory error.
                loaded, skipped = 0, []
                model_dict = dict(self.predictor.named_parameters())
                model_bufs = dict(self.predictor.named_buffers())
                for k, v in pred_state.items():
                    if k in model_dict:
                        model_dict[k].data = v.clone()
                        loaded += 1
                    elif k in model_bufs:
                        model_bufs[k].data = v.clone()
                        loaded += 1
                    else:
                        skipped.append(k)
                if skipped:
                    print(f"[CPCEncoder] Predictor skipped keys: {skipped[:3]}...")
                print(f"[CPCEncoder] Predictor loaded ({loaded}/{len(pred_state)} keys)")

        print(f"[CPCEncoder] Loaded from {path} (epoch={ckpt.get('epoch', '?')})")

    def forward(self, x):
        """x: (B, 12, T) at data target_fs → 600 samples (2.5s @ 240Hz)"""
        x = torch.nan_to_num(x)
        if x.shape[-1] != self.model_seq_len:
            x = F.interpolate(x, size=self.model_seq_len, mode="linear", align_corners=False)

        # Encoder: (B, 12, 600) → (B, 512, T')
        enc_out = self.encoder(x)  # (B, 512, T')

        if self._has_predictor and self.predictor is not None:
            try:
                pred_out = self.predictor(enc_out)
                seq = pred_out.transpose(1, 2)
            except Exception as e:
                if not getattr(CPCEncoder, "_s4_warned", False):
                    import traceback
                    print(f"[CPCEncoder] S4 forward 실패 → encoder-only fallback: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    print(f"[CPCEncoder] enc_out shape: {enc_out.shape}, dtype: {enc_out.dtype}, device: {enc_out.device}")
                    CPCEncoder._s4_warned = True
                self._has_predictor = False
                seq = enc_out.transpose(1, 2)
        else:
            seq = enc_out.transpose(1, 2)

        pooled = seq.mean(dim=1)  # (B, 512)
        return seq, pooled

    def get_layer_groups(self):
        early, late = [], []
        for name, param in self.named_parameters():
            if "encoder" in name:
                early.append(param)
            else:
                late.append(param)
        return {"early": early, "late": late}
