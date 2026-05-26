# =============================================================
# Model Registry
# -------------------------------------------------------------
# Sourced by run_full_benchmark.sh.
# To add a new model to the benchmark, edit only this file:
#
#   1) Write the adapter in src/encoders/<model>.py
#      (must expose feature_dim + forward(x))
#   2) Export the class from src/encoders/__init__.py
#   3) Add one line to each map below — [model_name]="..."
#   4) (optional) Add model_name to MODEL_NAMES_DEFAULT
#
# Run a subset of models:
#   MODELS_OVERRIDE="ecg_jepa st_mem" bash run_full_benchmark.sh
# =============================================================

declare -A MODEL_CLS_MAP=(
    [ecg_founder]="src.encoders.ecg_founder.ECGFounderEncoder"
    [ecg_jepa]="src.encoders.ecg_jepa.ECGJEPAEncoder"
    [st_mem]="src.encoders.st_mem.StMemEncoder"
    [merl]="src.encoders.merl.MerlResNetEncoder"
    [ecgfm_ked]="src.encoders.ecgfm_ked.EcgFmKEDEncoder"
    [hubert_ecg]="src.encoders.hubert_ecg.HuBERTECGEncoder"
    [ecg_fm]="src.encoders.ecg_fm.ECGFMEncoder"
    [cpc]="src.encoders.cpc.CPCEncoder"
    # MoRyECG (HEEDB-pretrained, codebook ablation v4) — 5 codebook sizes
    [moryecg_cb128]="src.encoders.moryecg.MoRyECGEncoder"
    [moryecg_cb256]="src.encoders.moryecg.MoRyECGEncoder"
    [moryecg_cb512]="src.encoders.moryecg.MoRyECGEncoder"
    [moryecg_cb1024]="src.encoders.moryecg.MoRyECGEncoder"
    [moryecg_cb2048]="src.encoders.moryecg.MoRyECGEncoder"
)

# MoRyECG pretrain repo. Defaults to the parent of this benchmark directory
# (i.e. the MoRyECG release root). Override with MORYECG_REPO if needed.
MORYECG_REPO_DEFAULT="${MORYECG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

declare -A MODEL_CKPT_MAP=(
    [ecg_founder]="${ECG_CKPT_ROOT}/ecg_founder/12_lead_ECGFounder.pth"
    [ecg_jepa]="${ECG_CKPT_ROOT}/ecg_jepa/multiblock_epoch100.pth"
    [st_mem]="${ECG_CKPT_ROOT}/st_mem/st_mem_vit_base_full.pth"
    [merl]="${ECG_CKPT_ROOT}/merl/res18_best_encoder.pth"
    [ecgfm_ked]="${ECG_CKPT_ROOT}/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt"
    [hubert_ecg]="${ECG_CKPT_ROOT}/hubert_ecg/hubert_ecg_base.safetensors"
    [ecg_fm]="${ECG_CKPT_ROOT}/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
    [cpc]="${ECG_CKPT_ROOT}/cpc/last_11597276.ckpt"
    # MoRyECG pretrain checkpoints (output of training/pretrain/train.py)
    [moryecg_cb128]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_heedb_cb128_v4/best.pt"
    [moryecg_cb256]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_heedb_cb256_v4/best.pt"
    [moryecg_cb512]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_heedb_cb512_v4/best.pt"
    [moryecg_cb1024]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_heedb_cb1024_v4/best.pt"
    [moryecg_cb2048]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_heedb_cb2048_v4/best.pt"
)

# Default run order (when MODELS_OVERRIDE is unset).
# MoRyECG variants are intentionally excluded from the default — request
# them explicitly via MODELS_OVERRIDE="moryecg_cb1024" etc.
MODEL_NAMES_DEFAULT=(ecg_founder ecg_jepa st_mem merl ecgfm_ked hubert_ecg ecg_fm cpc)
