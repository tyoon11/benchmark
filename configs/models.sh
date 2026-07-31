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
    # MoRyECG A5: Axial S4D + MHA backbone (d_model=384, ~26M), cb1024, full HEEDB pretrain.
    # Dedicated adapter (does NOT reuse the transformer-family MoRyECGEncoder).
    [moryecg_a5]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    # MoRyECG A5 data-scaling sweep — same adapter/backbone as moryecg_a5, only the
    # pretrain data size (and hence ckpt) differs. full == moryecg_a5 above.
    [moryecg_a5_p05]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_p1]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_p3]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_p5]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_p05_isocompute]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_p1_isocompute]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    # MoRyECG A5 heedb_scale_tables ladder (5k..3200k, nested; fixed val/test seed2025)
    [moryecg_a5_s5k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s10k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s25k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s50k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s100k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s200k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s400k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s800k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s1600k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    [moryecg_a5_s3200k]="src.encoders.moryecg_a5.MoRyECGA5Encoder"
    # MoRyECG A5 / GNN-RVQ: same axial backbone but driven by the frozen GNN-RVQ
    # patch tokenizer (backbone == "axial_s4_rvq"). Separate adapter + checkpoint.
    [moryecg_a5_rvqgnn]="src.encoders.moryecg_a5_rvqgnn.MoRyECGA5RVQGNNEncoder"
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
    # A5 axial_s4 pretrain output (configs/pretrain/axial_s4_a5_heedb_full_cb1024.yaml).
    # Tokenizer auto-derived → checkpoints/tokenizer_heedb_full_cb1024_v4/best.pt
    [moryecg_a5]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_axial_s4_a5_heedb_full_cb1024/best.pt"
    # A5 data-scaling sweep ckpts (output of scripts/scaling/run_scaling_sweep.sh).
    [moryecg_a5_p05]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scaling/p05/best.pt"
    [moryecg_a5_p1]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scaling/p1/best.pt"
    [moryecg_a5_p3]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scaling/p3/best.pt"
    [moryecg_a5_p5]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scaling/p5/best.pt"
    [moryecg_a5_p05_isocompute]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scaling/p05_isocompute/best.pt"
    [moryecg_a5_p1_isocompute]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scaling/p1_isocompute/best.pt"
    # heedb_scale_tables ladder ckpts (output of scripts/scaling/run_scale_tables_sweep.sh)
    [moryecg_a5_s5k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/5k/best.pt"
    [moryecg_a5_s10k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/10k/best.pt"
    [moryecg_a5_s25k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/25k/best.pt"
    [moryecg_a5_s50k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/50k/best.pt"
    [moryecg_a5_s100k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/100k/best.pt"
    [moryecg_a5_s200k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/200k/best.pt"
    [moryecg_a5_s400k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/400k/best.pt"
    [moryecg_a5_s800k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/800k/best.pt"
    [moryecg_a5_s1600k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/1600k/best.pt"
    [moryecg_a5_s3200k]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_a5_scale/3200k/best.pt"
    # RVQ-A5: tokenizer path is auto-read from the checkpoint's 'tokenizer_ckpt'.
    [moryecg_a5_rvqgnn]="${MORYECG_REPO_DEFAULT}/checkpoints/pretrain_axial_s4_a5_rvqgnn_patch100/best.pt"
)

# Default run order (when MODELS_OVERRIDE is unset).
# MoRyECG variants are intentionally excluded from the default — request
# them explicitly via MODELS_OVERRIDE="moryecg_cb1024" etc.
MODEL_NAMES_DEFAULT=(ecg_founder ecg_jepa st_mem merl ecgfm_ked hubert_ecg ecg_fm cpc)
