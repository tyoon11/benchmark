"""
전체 ECG Foundation Model UMAP 임베딩 비교 분석
=================================================
ECGFMs 디렉토리의 모든 모델에 대해 PTB-XL(성인) / ZZU-pECG(소아)
전체 샘플 임베딩을 추출하고 UMAP 시각화를 생성합니다.

실행:
  # 전체 샘플, 전체 GPU
  python scripts/run_all_embedding_umap.py --n_samples 0 --batch_size 256

  # 특정 모델만, 샘플 수 제한
  python scripts/run_all_embedding_umap.py --n_samples 2000 --models ECG-JEPA ST-MEM

출력:
  results/embeddings/{model_name}_ptbxl.npy   — PTB-XL 임베딩
  results/embeddings/{model_name}_zzu.npy     — ZZU 임베딩
  results/embeddings/{model_name}_ptbxl_labels.npy
  results/embeddings/{model_name}_zzu_labels.npy
  results/umap_all_models.png                 — 전체 모델 비교 그리드
  results/umap_silhouette_scores.csv          — 실루엣 스코어 요약
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import H5ECGDataset

# ══════════════════════════════════════════════════════════════════════
# Model Registry
# ══════════════════════════════════════════════════════════════════════

MODEL_DIR = Path("/home/irteam/ddn-opendata1/model/ECGFMs")

MODEL_REGISTRY = [
    {
        "name": "ECG-JEPA",
        "encoder_cls": "src.encoders.ecg_jepa.ECGJEPAEncoder",
        "checkpoint": str(MODEL_DIR / "ecg_jepa" / "multiblock_epoch100.pth"),
    },
    {
        "name": "ECG-FM",
        "encoder_cls": "src.encoders.ecg_fm.ECGFMEncoder",
        "checkpoint": str(MODEL_DIR / "ecg_fm" / "mimic_iv_ecg_physionet_pretrained.pt"),
    },
    {
        "name": "ECG-Founder",
        "encoder_cls": "src.encoders.ecg_founder.ECGFounderEncoder",
        "checkpoint": str(MODEL_DIR / "ecg_founder" / "12_lead_ECGFounder.pth"),
    },
    {
        "name": "ST-MEM",
        "encoder_cls": "src.encoders.st_mem.StMemEncoder",
        "checkpoint": str(MODEL_DIR / "st_mem" / "st_mem_vit_base_full.pth"),
    },
    {
        "name": "MERL (ResNet)",
        "encoder_cls": "src.encoders.merl.MerlResNetEncoder",
        "checkpoint": str(MODEL_DIR / "merl" / "res18_best_encoder.pth"),
    },
    {
        "name": "MERL (ViT)",
        "encoder_cls": "src.encoders.merl.MerlViTEncoder",
        "checkpoint": str(MODEL_DIR / "merl" / "vit_tiny_best_encoder.pth"),
    },
    {
        "name": "ECG-FM-KED",
        "encoder_cls": "src.encoders.ecgfm_ked.EcgFmKEDEncoder",
        "checkpoint": str(MODEL_DIR / "ecgfm_ked" / "best_valid_all_increase_with_augment_epoch_3.pt"),
    },
    {
        "name": "HuBERT-ECG",
        "encoder_cls": "src.encoders.hubert_ecg.HuBERTECGEncoder",
        "checkpoint": str(MODEL_DIR / "hubert_ecg" / "hubert_ecg_base.safetensors"),
    },
    {
        "name": "CPC",
        "encoder_cls": "src.encoders.cpc.CPCEncoder",
        "checkpoint": str(MODEL_DIR / "cpc" / "last_11597276.ckpt"),
    },
]


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def load_encoder(cfg, device):
    """모델 레지스트리 엔트리에서 인코더를 로드"""
    import importlib

    mod_path, cls_name = cfg["encoder_cls"].rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)

    kwargs = cfg.get("extra_kwargs", {})
    encoder = cls(checkpoint=cfg["checkpoint"], **kwargs)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def extract_embeddings_batched(encoder, dataset, gpu_ids, batch_size=256,
                               num_workers=8, n_samples=0):
    """
    DataLoader + DataParallel 배치 추론으로 임베딩 추출.

    Args:
        encoder:     인코더 모델 (단일 GPU에 이미 로드됨)
        dataset:     H5ECGDataset
        gpu_ids:     사용할 GPU ID 리스트
        batch_size:  배치 크기 (per-GPU)
        num_workers: DataLoader 워커 수
        n_samples:   0이면 전체, 양수면 해당 수만큼

    Returns:
        embeddings:  (N, D) ndarray
        labels:      (N, C) ndarray
    """
    # 서브셋 처리
    if n_samples > 0 and n_samples < len(dataset):
        dataset = Subset(dataset, list(range(n_samples)))

    # DataParallel 래핑
    if len(gpu_ids) > 1:
        dp_encoder = nn.DataParallel(encoder, device_ids=gpu_ids)
    else:
        dp_encoder = encoder
    dp_encoder.eval()

    total_batch_size = batch_size * len(gpu_ids)

    loader = DataLoader(
        dataset,
        batch_size=total_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_embeddings = []
    all_labels = []
    primary_device = torch.device(f"cuda:{gpu_ids[0]}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting", leave=False):
            signals = batch["signal"].to(primary_device)  # (B, C, T)
            labels = batch["label"]                        # (B, num_classes)

            out = dp_encoder(signals)

            # 출력 파싱
            if isinstance(out, tuple):
                pooled = out[1]  # (B, D)
            elif isinstance(out, dict):
                pooled = out.get("pooled", out.get("pooled_features"))
            elif out.dim() == 3:
                pooled = out.mean(dim=1)
            else:
                pooled = out

            all_embeddings.append(pooled.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels


def labels_to_normal(labels, label_idx=0):
    """Multi-hot 라벨에서 Normal/Abnormal 문자열 리스트로 변환"""
    return ["Normal" if row[label_idx] > 0 else "Abnormal" for row in labels]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="전체 모델 UMAP 임베딩 비교 (Multi-GPU)")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="추출할 샘플 수 (0=전체)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="per-GPU 배치 크기")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader 워커 수")
    parser.add_argument("--gpus", type=str, default=None,
                        help="사용할 GPU ID (예: '0,1,2,3,4,5,6'). 기본: 모든 GPU")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="실행할 모델 이름 (기본: 전체)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="이미 저장된 임베딩이 있으면 건너뜀")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── GPU 설정 ──
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    n_gpus = len(gpu_ids)
    logging.info(f"GPU {n_gpus}개 사용: {gpu_ids}")
    primary_device = torch.device(f"cuda:{gpu_ids[0]}")

    # ── 데이터셋 로드 (500Hz, 2500 samples = 5초) ──
    label_dir = SCRIPT_DIR / "labels"

    logging.info("PTB-XL (Adult) 데이터셋 로드...")
    ptbxl_ds = H5ECGDataset(
        h5_root="/home/irteam/ddn-opendata1/h5/physionet/v2.0",
        table_csv="/home/irteam/ddn-opendata1/h5/physionet/v2.0/ptbxl_table.csv",
        label_csv=str(label_dir / "ptbxl_super_bench_labels.csv"),
        target_fs=500, target_length=2500,
    )

    logging.info("ZZU-pECG (Pediatric) 데이터셋 로드...")
    zzu_ds = H5ECGDataset(
        h5_root="/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0",
        table_csv="/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0/ecg_table.csv",
        label_csv=str(label_dir / "zzu_bench_labels.csv"),
        target_fs=500, target_length=2500,
    )

    n_ptbxl = len(ptbxl_ds) if args.n_samples == 0 else min(args.n_samples, len(ptbxl_ds))
    n_zzu = len(zzu_ds) if args.n_samples == 0 else min(args.n_samples, len(zzu_ds))
    logging.info(f"PTB-XL: {n_ptbxl}/{len(ptbxl_ds)} samples, ZZU: {n_zzu}/{len(zzu_ds)} samples")
    logging.info(f"batch_size={args.batch_size} × {n_gpus} GPUs = {args.batch_size * n_gpus} effective")

    # ── 모델 필터링 ──
    if args.models:
        models_to_run = [m for m in MODEL_REGISTRY if m["name"] in args.models]
        if not models_to_run:
            logging.error(f"지정된 모델을 찾을 수 없습니다: {args.models}")
            logging.info(f"사용 가능한 모델: {[m['name'] for m in MODEL_REGISTRY]}")
            return
    else:
        models_to_run = MODEL_REGISTRY

    # ── 임베딩 저장 디렉토리 ──
    emb_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: 모델별 임베딩 추출
    # ══════════════════════════════════════════════════════════════════

    results = {}
    for cfg in models_to_run:
        name = cfg["name"]
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")

        logging.info(f"\n{'='*60}")
        logging.info(f"Model: {name}")
        logging.info(f"{'='*60}")

        # ── 기존 임베딩 확인 ──
        ptbxl_path = os.path.join(emb_dir, f"{safe_name}_ptbxl.npy")
        zzu_path = os.path.join(emb_dir, f"{safe_name}_zzu.npy")
        ptbxl_lbl_path = os.path.join(emb_dir, f"{safe_name}_ptbxl_labels.npy")
        zzu_lbl_path = os.path.join(emb_dir, f"{safe_name}_zzu_labels.npy")
        meta_path = os.path.join(emb_dir, f"{safe_name}_meta.npz")

        if args.skip_existing and all(
            os.path.exists(p) for p in [ptbxl_path, zzu_path, ptbxl_lbl_path, zzu_lbl_path]
        ):
            logging.info(f"  ⏭ 기존 임베딩 로드: {safe_name}")
            emb_ptbxl = np.load(ptbxl_path)
            emb_zzu = np.load(zzu_path)
            lbl_ptbxl = np.load(ptbxl_lbl_path)
            lbl_zzu = np.load(zzu_lbl_path)
            meta = np.load(meta_path, allow_pickle=True)
            feature_dim = int(meta["feature_dim"])

            results[name] = {
                "emb_ptbxl": emb_ptbxl,
                "emb_zzu": emb_zzu,
                "ptbxl_labels": labels_to_normal(lbl_ptbxl),
                "zzu_labels": labels_to_normal(lbl_zzu),
                "feature_dim": feature_dim,
            }
            continue

        try:
            encoder = load_encoder(cfg, primary_device)
            feature_dim = encoder.feature_dim
            logging.info(f"  feature_dim={feature_dim}")

            # ── PTB-XL ──
            logging.info(f"  PTB-XL 임베딩 추출 ({n_ptbxl}개, {n_gpus} GPUs)...")
            emb_ptbxl, lbl_ptbxl = extract_embeddings_batched(
                encoder, ptbxl_ds, gpu_ids,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                n_samples=args.n_samples,
            )

            # ── ZZU ──
            logging.info(f"  ZZU 임베딩 추출 ({n_zzu}개, {n_gpus} GPUs)...")
            emb_zzu, lbl_zzu = extract_embeddings_batched(
                encoder, zzu_ds, gpu_ids,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                n_samples=args.n_samples,
            )

            # ── 저장 ──
            np.save(ptbxl_path, emb_ptbxl)
            np.save(zzu_path, emb_zzu)
            np.save(ptbxl_lbl_path, lbl_ptbxl)
            np.save(zzu_lbl_path, lbl_zzu)
            np.savez(meta_path, feature_dim=feature_dim)

            results[name] = {
                "emb_ptbxl": emb_ptbxl,
                "emb_zzu": emb_zzu,
                "ptbxl_labels": labels_to_normal(lbl_ptbxl),
                "zzu_labels": labels_to_normal(lbl_zzu),
                "feature_dim": feature_dim,
            }

            # 메모리 해제
            del encoder
            torch.cuda.empty_cache()

            logging.info(f"  ✓ {name} 완료 (PTB-XL: {emb_ptbxl.shape}, ZZU: {emb_zzu.shape})")
            logging.info(f"  ✓ 저장: {emb_dir}/{safe_name}_*.npy")

        except Exception as e:
            logging.error(f"  ✗ {name} 실패: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        logging.error("성공한 모델이 없습니다.")
        return

    # ══════════════════════════════════════════════════════════════════
    # Phase 2a: 기존 3-column UMAP (Adult vs Pedi, PTB-XL N/Ab, ZZU N/Ab)
    # ══════════════════════════════════════════════════════════════════

    from umap import UMAP
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_models = len(results)
    fig3, axes3 = plt.subplots(n_models, 3, figsize=(24, 7 * n_models))
    if n_models == 1:
        axes3 = axes3[None, :]

    silhouette_3col = []
    # UMAP 좌표 캐시 (phase 2b에서 재사용 가능하지만 balanced는 다른 sample이므로 별도)
    phase_a_coords = {}

    for row_idx, (model_name, data) in enumerate(results.items()):
        emb_ptbxl = data["emb_ptbxl"]
        emb_zzu = data["emb_zzu"]
        ptbxl_labels = data["ptbxl_labels"]
        zzu_labels = data["zzu_labels"]

        logging.info(f"\n[3-col UMAP] {model_name} ({len(emb_ptbxl)} + {len(emb_zzu)} samples)")
        all_emb = np.concatenate([emb_ptbxl, emb_zzu], axis=0)

        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(all_emb)
        coords_ptbxl = coords[: len(emb_ptbxl)]
        coords_zzu = coords[len(emb_ptbxl) :]
        phase_a_coords[model_name] = (coords_ptbxl, coords_zzu)

        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        np.save(os.path.join(emb_dir, f"{safe_name}_umap_coords_all.npy"), coords)

        # Silhouette
        dataset_labels = [0] * len(emb_ptbxl) + [1] * len(emb_zzu)
        sample_sz = min(10000, len(all_emb))
        sil_dataset = silhouette_score(all_emb, dataset_labels, sample_size=sample_sz)

        ptbxl_binary = [0 if s == "Normal" else 1 for s in ptbxl_labels]
        sil_ptbxl = (silhouette_score(emb_ptbxl, ptbxl_binary,
                                      sample_size=min(10000, len(emb_ptbxl)))
                     if len(set(ptbxl_binary)) > 1 else float("nan"))
        zzu_binary = [0 if s == "Normal" else 1 for s in zzu_labels]
        sil_zzu = (silhouette_score(emb_zzu, zzu_binary,
                                    sample_size=min(10000, len(emb_zzu)))
                   if len(set(zzu_binary)) > 1 else float("nan"))

        silhouette_3col.append({
            "model": model_name,
            "feature_dim": data["feature_dim"],
            "n_ptbxl": len(emb_ptbxl),
            "n_zzu": len(emb_zzu),
            "adult_vs_pediatric": sil_dataset,
            "ptbxl_normal_vs_abnormal": sil_ptbxl,
            "zzu_normal_vs_abnormal": sil_zzu,
        })
        logging.info(
            f"  Silhouette — Adult/Pedi: {sil_dataset:.4f}, "
            f"PTB-XL N/Ab: {sil_ptbxl:.4f}, ZZU N/Ab: {sil_zzu:.4f}"
        )

        # Plot 1: Adult vs Pediatric
        ax = axes3[row_idx, 0]
        ax.scatter(coords_ptbxl[:, 0], coords_ptbxl[:, 1], c="steelblue", s=1, alpha=0.3,
                   rasterized=True, label=f"PTB-XL (Adult, n={len(emb_ptbxl)})")
        ax.scatter(coords_zzu[:, 0], coords_zzu[:, 1], c="coral", s=1, alpha=0.3,
                   rasterized=True, label=f"ZZU (Pediatric, n={len(emb_zzu)})")
        ax.set_title(f"{model_name}\nAdult vs Pediatric (sil={sil_dataset:.3f})", fontsize=12)
        ax.legend(fontsize=8, markerscale=5, loc="best")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

        # Plot 2: PTB-XL Normal vs Abnormal
        ax = axes3[row_idx, 1]
        colors = ["#2ecc71" if s == "Normal" else "#e74c3c" for s in ptbxl_labels]
        ax.scatter(coords_ptbxl[:, 0], coords_ptbxl[:, 1], c=colors, s=1, alpha=0.3, rasterized=True)
        legend_na = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
                   markersize=8, label="Normal"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=8, label="Abnormal"),
        ]
        ax.legend(handles=legend_na, fontsize=8)
        ax.set_title(f"PTB-XL: Normal vs Abnormal (sil={sil_ptbxl:.3f})", fontsize=12)
        ax.set_xlabel("UMAP 1")

        # Plot 3: ZZU Normal vs Abnormal
        ax = axes3[row_idx, 2]
        colors = ["#2ecc71" if s == "Normal" else "#e74c3c" for s in zzu_labels]
        ax.scatter(coords_zzu[:, 0], coords_zzu[:, 1], c=colors, s=1, alpha=0.3, rasterized=True)
        ax.legend(handles=legend_na, fontsize=8)
        ax.set_title(f"ZZU (Pediatric): Normal vs Abnormal (sil={sil_zzu:.3f})", fontsize=12)
        ax.set_xlabel("UMAP 1")

    plt.suptitle(
        "ECG Foundation Models — UMAP Embedding Comparison\n"
        "(PTB-XL: Adult / ZZU-pECG: Pediatric)",
        fontsize=16, y=1.01,
    )
    plt.tight_layout()
    fig_path_3col = os.path.join(args.output_dir, "umap_all_models.png")
    plt.savefig(fig_path_3col, dpi=150, bbox_inches="tight")
    logging.info(f"\n[3-col] 그림 저장: {fig_path_3col}")
    plt.close(fig3)

    # 3-column silhouette CSV
    import csv
    csv_path_3col = os.path.join(args.output_dir, "umap_silhouette_scores.csv")
    with open(csv_path_3col, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "feature_dim", "n_ptbxl", "n_zzu",
            "adult_vs_pediatric", "ptbxl_normal_vs_abnormal", "zzu_normal_vs_abnormal",
        ])
        writer.writeheader()
        writer.writerows(silhouette_3col)
    logging.info(f"[3-col] 실루엣 저장: {csv_path_3col}")

    # ══════════════════════════════════════════════════════════════════
    # Phase 2b: 4-way balanced UMAP (Adult-N, Adult-A, Pediatric-N, Pediatric-A)
    # ══════════════════════════════════════════════════════════════════

    # 4-way 색상 + 마커 팔레트 (확실히 구분되는 4색)
    GROUP_STYLES = {
        "Adult-Normal":       {"color": "#1f77b4", "marker": "o"},  # 파랑, 원
        "Adult-Abnormal":     {"color": "#ff7f0e", "marker": "s"},  # 주황, 사각
        "Pediatric-Normal":   {"color": "#2ca02c", "marker": "^"},  # 초록, 삼각
        "Pediatric-Abnormal": {"color": "#d62728", "marker": "D"},  # 빨강, 다이아
    }
    GROUP_COLORS = {k: v["color"] for k, v in GROUP_STYLES.items()}

    rng = np.random.RandomState(42)
    fig4, axes4 = plt.subplots(n_models, 1, figsize=(10, 10 * n_models))
    if n_models == 1:
        axes4 = [axes4]

    silhouette_records = []

    for row_idx, (model_name, data) in enumerate(results.items()):
        emb_ptbxl = data["emb_ptbxl"]
        emb_zzu = data["emb_zzu"]
        ptbxl_labels = data["ptbxl_labels"]
        zzu_labels = data["zzu_labels"]

        # ── 4개 그룹 인덱스 ──
        idx_adult_norm = [i for i, s in enumerate(ptbxl_labels) if s == "Normal"]
        idx_adult_abn  = [i for i, s in enumerate(ptbxl_labels) if s == "Abnormal"]
        idx_pedi_norm  = [i for i, s in enumerate(zzu_labels)   if s == "Normal"]
        idx_pedi_abn   = [i for i, s in enumerate(zzu_labels)   if s == "Abnormal"]

        group_sizes = {
            "Adult-Normal":       len(idx_adult_norm),
            "Adult-Abnormal":     len(idx_adult_abn),
            "Pediatric-Normal":   len(idx_pedi_norm),
            "Pediatric-Abnormal": len(idx_pedi_abn),
        }
        # 최소 그룹 크기로 balance
        n_per_group = min(group_sizes.values())

        logging.info(
            f"\nUMAP 수행: {model_name} — group sizes: {group_sizes}, "
            f"balanced n_per_group={n_per_group}"
        )

        # 랜덤 서브샘플
        sel_an = rng.choice(idx_adult_norm, n_per_group, replace=False)
        sel_aa = rng.choice(idx_adult_abn,  n_per_group, replace=False)
        sel_pn = rng.choice(idx_pedi_norm,  n_per_group, replace=False)
        sel_pa = rng.choice(idx_pedi_abn,   n_per_group, replace=False)

        # 임베딩/라벨 조합
        emb_list = [
            emb_ptbxl[sel_an],
            emb_ptbxl[sel_aa],
            emb_zzu[sel_pn],
            emb_zzu[sel_pa],
        ]
        group_labels_list = (
            ["Adult-Normal"]       * n_per_group
            + ["Adult-Abnormal"]   * n_per_group
            + ["Pediatric-Normal"] * n_per_group
            + ["Pediatric-Abnormal"] * n_per_group
        )
        all_emb = np.concatenate(emb_list, axis=0)

        # ── UMAP ──
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(all_emb)

        # ── UMAP 좌표 저장 ──
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        np.save(os.path.join(emb_dir, f"{safe_name}_umap_coords_balanced.npy"), coords)
        np.save(os.path.join(emb_dir, f"{safe_name}_umap_labels_balanced.npy"),
                np.array(group_labels_list))

        # ── Silhouette: 4-way, 성인vs소아(2-way), 정상vs비정상(2-way) ──
        group_int = np.array([list(GROUP_COLORS.keys()).index(g) for g in group_labels_list])
        ap_int = np.array([0 if g.startswith("Adult") else 1 for g in group_labels_list])
        na_int = np.array([0 if g.endswith("Normal") else 1 for g in group_labels_list])

        sample_sz = min(10000, len(all_emb))
        sil_4way = silhouette_score(all_emb, group_int, sample_size=sample_sz)
        sil_ap = silhouette_score(all_emb, ap_int, sample_size=sample_sz)
        sil_na = silhouette_score(all_emb, na_int, sample_size=sample_sz)

        silhouette_records.append({
            "model": model_name,
            "feature_dim": data["feature_dim"],
            "n_per_group": n_per_group,
            "silhouette_4way": sil_4way,
            "silhouette_adult_vs_pedi": sil_ap,
            "silhouette_normal_vs_abnormal": sil_na,
        })
        logging.info(
            f"  Silhouette — 4-way: {sil_4way:.4f}, "
            f"Adult/Pedi: {sil_ap:.4f}, Normal/Ab: {sil_na:.4f}"
        )

        # ── 그룹별 플롯 (한 ax에 4개 그룹 오버레이, 색+마커+edge로 확실히 구분) ──
        ax = axes4[row_idx]
        s0 = 0
        for gname, style in GROUP_STYLES.items():
            s1 = s0 + n_per_group
            ax.scatter(
                coords[s0:s1, 0], coords[s0:s1, 1],
                c=style["color"], marker=style["marker"],
                s=12, alpha=0.6, rasterized=True,
                edgecolors="white", linewidths=0.3,
                label=f"{gname} (n={n_per_group})",
            )
            s0 = s1

        ax.set_title(
            f"{model_name}  (feature_dim={data['feature_dim']})\n"
            f"4-way sil={sil_4way:.3f}  |  "
            f"Adult/Pedi sil={sil_ap:.3f}  |  Normal/Ab sil={sil_na:.3f}",
            fontsize=11,
        )
        ax.legend(fontsize=9, markerscale=3, loc="best")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    # ── 저장 ──
    plt.suptitle(
        "ECG Foundation Models — Balanced 4-Way UMAP\n"
        "(Adult-Normal / Adult-Abnormal / Pediatric-Normal / Pediatric-Abnormal)",
        fontsize=14, y=1.005,
    )
    plt.tight_layout()

    fig_path = os.path.join(args.output_dir, "umap_all_models_4way.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    logging.info(f"\n[4-way] 그림 저장: {fig_path}")
    plt.close(fig4)

    # ── Silhouette CSV ──
    csv_path = os.path.join(args.output_dir, "umap_silhouette_scores_4way.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "feature_dim", "n_per_group",
            "silhouette_4way", "silhouette_adult_vs_pedi", "silhouette_normal_vs_abnormal",
        ])
        writer.writeheader()
        writer.writerows(silhouette_records)
    logging.info(f"실루엣 스코어 저장: {csv_path}")

    # ── 터미널 요약 ──
    logging.info(f"\n{'='*80}")
    logging.info(f"{'Model':<20} {'Dim':>5} {'N/group':>8} {'4way':>8} {'Adult/Pedi':>12} {'Normal/Ab':>11}")
    logging.info(f"{'-'*80}")
    for rec in silhouette_records:
        logging.info(
            f"{rec['model']:<20} {rec['feature_dim']:>5} "
            f"{rec['n_per_group']:>8} "
            f"{rec['silhouette_4way']:>8.4f} "
            f"{rec['silhouette_adult_vs_pedi']:>12.4f} "
            f"{rec['silhouette_normal_vs_abnormal']:>11.4f}"
        )
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    main()
