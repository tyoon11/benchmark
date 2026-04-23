# =============================================================
# Model Registry
# -------------------------------------------------------------
# run_full_benchmark.sh가 source해서 사용합니다.
# 새 모델을 벤치마크에 추가하려면 여기 2줄만 추가하면 됩니다:
#
#   1) src/encoders/<model>.py 에 adapter 작성 (feature_dim + forward(x))
#   2) src/encoders/__init__.py 에 export
#   3) 아래 두 map에 [model_name]="..." 1줄씩 추가
#   4) (선택) MODEL_NAMES_DEFAULT 에 model_name 추가
#
# 특정 모델만 돌리고 싶으면:
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
)

declare -A MODEL_CKPT_MAP=(
    [ecg_founder]="/home/irteam/ddn-opendata1/model/ECGFMs/ecg_founder/12_lead_ECGFounder.pth"
    [ecg_jepa]="/home/irteam/ddn-opendata1/model/ECGFMs/ecg_jepa/multiblock_epoch100.pth"
    [st_mem]="/home/irteam/ddn-opendata1/model/ECGFMs/st_mem/st_mem_vit_base_full.pth"
    [merl]="/home/irteam/ddn-opendata1/model/ECGFMs/merl/res18_best_encoder.pth"
    [ecgfm_ked]="/home/irteam/ddn-opendata1/model/ECGFMs/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt"
    [hubert_ecg]="/home/irteam/ddn-opendata1/model/ECGFMs/hubert_ecg/hubert_ecg_base.safetensors"
    [ecg_fm]="/home/irteam/ddn-opendata1/model/ECGFMs/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
    [cpc]="/home/irteam/ddn-opendata1/model/ECGFMs/cpc/last_11597276.ckpt"
)

# 기본 실행 순서 (MODELS_OVERRIDE 미지정 시 사용)
MODEL_NAMES_DEFAULT=(ecg_founder ecg_jepa st_mem merl ecgfm_ked hubert_ecg ecg_fm cpc)
