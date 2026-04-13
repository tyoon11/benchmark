"""
임베딩 분석 스크립트
====================
Encoder의 representation을 추출하여 t-SNE/UMAP 시각화.

분석:
  1. PTB-XL vs ZZU (성인 vs 소아)
  2. Normal vs Abnormal (진단별 클러스터)

실행:
  python scripts/embedding_analysis.py \
      --encoder_cls src.encoders.ecg_jepa.ECGJEPAEncoder \
      --encoder_ckpt /path/to/best.pth \
      --n_samples 2000
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import H5ECGDataset


def extract_embeddings(encoder, dataset, device, n_samples=2000):
    """Encoder에서 pooled representation 추출"""
    encoder.eval()
    embeddings = []
    indices = list(range(min(n_samples, len(dataset))))

    with torch.no_grad():
        for i in tqdm(indices, desc="Extracting", leave=False):
            item = dataset[i]
            x = item["signal"].unsqueeze(0).to(device)  # (1, C, T)
            out = encoder(x)

            # 출력 파싱
            if isinstance(out, tuple):
                pooled = out[1]  # (1, D)
            elif isinstance(out, dict):
                pooled = out.get("pooled", out.get("pooled_features"))
            elif out.dim() == 3:
                pooled = out.mean(dim=1)
            else:
                pooled = out

            embeddings.append(pooled.cpu().numpy().squeeze())

    return np.array(embeddings), indices


def main():
    parser = argparse.ArgumentParser(description="임베딩 분석")
    parser.add_argument("--encoder_cls", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/embedding_analysis.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Encoder 로드
    import importlib
    mod_path, cls_name = args.encoder_cls.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)

    if args.encoder_ckpt:
        encoder = cls(checkpoint=args.encoder_ckpt)
    else:
        encoder = cls()
    encoder = encoder.to(device)
    encoder.eval()
    logging.info(f"Encoder: {args.encoder_cls} (feature_dim={encoder.feature_dim})")

    # ── 데이터셋 로드 ──
    label_dir = SCRIPT_DIR / "labels"
    n = args.n_samples

    # PTB-XL (성인)
    logging.info("PTB-XL 로드...")
    ptbxl_ds = H5ECGDataset(
        h5_root="/home/irteam/ddn-opendata1/h5/physionet/v2.0",
        table_csv="/home/irteam/ddn-opendata1/h5/physionet/v2.0/ptbxl_table.csv",
        label_csv=str(label_dir / "ptbxl_super_bench_labels.csv"),
        target_fs=500, target_length=2500,
    )

    # ZZU (소아)
    logging.info("ZZU 로드...")
    zzu_ds = H5ECGDataset(
        h5_root="/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0",
        table_csv="/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0/ecg_table.csv",
        label_csv=str(label_dir / "zzu_bench_labels.csv"),
        target_fs=500, target_length=2500,
    )

    # ── 임베딩 추출 ──
    logging.info(f"PTB-XL 임베딩 추출 ({n}개)...")
    emb_ptbxl, idx_ptbxl = extract_embeddings(encoder, ptbxl_ds, device, n)

    logging.info(f"ZZU 임베딩 추출 ({n}개)...")
    emb_zzu, idx_zzu = extract_embeddings(encoder, zzu_ds, device, n)

    # ── 라벨 수집 ──
    # PTB-XL: NORM 여부
    ptbxl_normal = []
    for i in idx_ptbxl:
        label = ptbxl_ds[i]["label"]
        ptbxl_normal.append("Normal" if label[0] > 0 else "Abnormal")  # NORM이 첫 컬럼

    # ZZU: is_Normal 여부
    zzu_normal = []
    for i in idx_zzu:
        label = zzu_ds[i]["label"]
        zzu_normal.append("Normal" if label[0] > 0 else "Abnormal")  # is_Normal이 첫 컬럼

    # ── 차원 축소 ──
    all_emb = np.concatenate([emb_ptbxl, emb_zzu], axis=0)
    logging.info(f"전체 임베딩: {all_emb.shape}")

    if args.method == "tsne":
        from sklearn.manifold import TSNE
        logging.info("t-SNE 수행...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        from umap import UMAP
        logging.info("UMAP 수행...")
        reducer = UMAP(n_components=2, random_state=42)

    coords = reducer.fit_transform(all_emb)
    coords_ptbxl = coords[:len(emb_ptbxl)]
    coords_zzu = coords[len(emb_ptbxl):]

    # ── 시각화 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Plot 1: 성인 vs 소아
    ax = axes[0]
    ax.scatter(coords_ptbxl[:, 0], coords_ptbxl[:, 1], c="steelblue", s=5, alpha=0.5, label=f"PTB-XL (Adult, n={len(emb_ptbxl)})")
    ax.scatter(coords_zzu[:, 0], coords_zzu[:, 1], c="coral", s=5, alpha=0.5, label=f"ZZU (Pediatric, n={len(emb_zzu)})")
    ax.set_title("Adult vs Pediatric", fontsize=14)
    ax.legend(fontsize=10, markerscale=3)
    ax.set_xlabel(f"{args.method.upper()} 1")
    ax.set_ylabel(f"{args.method.upper()} 2")

    # Plot 2: PTB-XL Normal vs Abnormal
    ax = axes[1]
    colors_ptbxl = ["#2ecc71" if s == "Normal" else "#e74c3c" for s in ptbxl_normal]
    ax.scatter(coords_ptbxl[:, 0], coords_ptbxl[:, 1], c=colors_ptbxl, s=5, alpha=0.5)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Abnormal'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_title("PTB-XL: Normal vs Abnormal", fontsize=14)
    ax.set_xlabel(f"{args.method.upper()} 1")

    # Plot 3: ZZU Normal vs Abnormal
    ax = axes[2]
    colors_zzu = ["#2ecc71" if s == "Normal" else "#e74c3c" for s in zzu_normal]
    ax.scatter(coords_zzu[:, 0], coords_zzu[:, 1], c=colors_zzu, s=5, alpha=0.5)
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_title("ZZU (Pediatric): Normal vs Abnormal", fontsize=14)
    ax.set_xlabel(f"{args.method.upper()} 1")

    plt.suptitle(f"ECG-JEPA Embedding Analysis ({args.method.upper()})", fontsize=16, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    logging.info(f"저장: {args.output}")

    # ── 수치 분석 ──
    from sklearn.metrics import silhouette_score

    # 성인 vs 소아 분리도
    dataset_labels = [0] * len(emb_ptbxl) + [1] * len(emb_zzu)
    sil_dataset = silhouette_score(all_emb, dataset_labels, sample_size=min(5000, len(all_emb)))

    # Normal vs Abnormal 분리도
    ptbxl_binary = [0 if s == "Normal" else 1 for s in ptbxl_normal]
    if len(set(ptbxl_binary)) > 1:
        sil_ptbxl = silhouette_score(emb_ptbxl, ptbxl_binary)
    else:
        sil_ptbxl = float("nan")

    zzu_binary = [0 if s == "Normal" else 1 for s in zzu_normal]
    if len(set(zzu_binary)) > 1:
        sil_zzu = silhouette_score(emb_zzu, zzu_binary)
    else:
        sil_zzu = float("nan")

    logging.info(f"\n{'='*50}")
    logging.info(f"Silhouette Scores:")
    logging.info(f"  Adult vs Pediatric:         {sil_dataset:.4f}")
    logging.info(f"  PTB-XL Normal vs Abnormal:  {sil_ptbxl:.4f}")
    logging.info(f"  ZZU Normal vs Abnormal:     {sil_zzu:.4f}")
    logging.info(f"{'='*50}")


if __name__ == "__main__":
    main()
