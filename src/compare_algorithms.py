# -*- coding: utf-8 -*-
"""
So sánh thuật toán Chameleon với K-Means và HAC.

Đánh giá bằng Silhouette Score, Adjusted Rand Index (ARI) và
Normalized Mutual Information (NMI).

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Dùng backend không cần GUI để lưu hình.
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# ---------------------------------------------------------------------------
# 1. Cấu hình đường dẫn
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "bank_churners_features.csv"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "processed" / "bank_churners_labels.csv"
DEFAULT_CHAMELEON_RESULTS = (
    PROJECT_ROOT / "outputs" / "chameleon" / "chameleon_cluster_results.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparison"


# ---------------------------------------------------------------------------
# 2. Đọc tham số dòng lệnh
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So sánh Chameleon với K-Means và HAC."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--chameleon-results", type=Path, default=DEFAULT_CHAMELEON_RESULTS,
        help="File kết quả cụm Chameleon (chameleon_cluster_results.csv).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument(
        "--n-clusters", type=int, default=None,
        help="Số cụm cho K-Means và HAC. Mặc định lấy từ kết quả Chameleon.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 3. Đọc dữ liệu và giảm chiều PCA
# ---------------------------------------------------------------------------

def load_data(features_path: Path, labels_path: Path):
    """Load preprocessed features and labels."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    X = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    if len(X) != len(labels):
        raise ValueError("Features and labels have different number of rows.")

    return X, labels


def reduce_dimension(X: pd.DataFrame, n_components: int):
    """Reduce dimensions with PCA, same pipeline as train_chameleon.py."""
    n_components = min(n_components, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} components, explained variance = {explained:.4f}")
    return X_pca


# ---------------------------------------------------------------------------
# 4. Đọc kết quả Chameleon đã train trước đó
# ---------------------------------------------------------------------------

def load_chameleon_labels(path: Path) -> np.ndarray | None:
    """Load Chameleon cluster labels from result file."""
    if not path.exists():
        print(f"[WARN] Chameleon result file not found: {path}")
        print("       Skipping Chameleon in comparison.")
        return None

    df = pd.read_csv(path)
    if "Cluster" not in df.columns:
        print("[WARN] Column 'Cluster' not found in Chameleon results file.")
        return None

    return df["Cluster"].values


# ---------------------------------------------------------------------------
# 5. Train các thuật toán phân cụm
# ---------------------------------------------------------------------------

def train_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Train K-Means."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return model.fit_predict(X)


def train_hac(X: np.ndarray, n_clusters: int, linkage: str = "ward") -> np.ndarray:
    """Train Hierarchical Agglomerative Clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(X)


# ---------------------------------------------------------------------------
# 6. Đánh giá chất lượng phân cụm
# ---------------------------------------------------------------------------

def evaluate(X: np.ndarray, cluster_labels: np.ndarray, true_labels: np.ndarray):
    """Compute Silhouette, ARI and NMI."""
    n_clusters = len(set(cluster_labels) - {-1})

    if n_clusters >= 2:
        sil = float(silhouette_score(X, cluster_labels))
    else:
        sil = float("nan")

    ari = float(adjusted_rand_score(true_labels, cluster_labels))
    nmi = float(normalized_mutual_info_score(true_labels, cluster_labels))

    return {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 6),
        "adjusted_rand_index": round(ari, 6),
        "normalized_mutual_info": round(nmi, 6),
    }


# ---------------------------------------------------------------------------
# 7. Bảng tổng hợp churn theo cụm (giống build_cluster_summary)
# ---------------------------------------------------------------------------

def cluster_summary(cluster_labels: np.ndarray, labels_df: pd.DataFrame):
    """Build churn rate summary per cluster."""
    df = labels_df.copy()
    df["Cluster"] = cluster_labels

    summary = (
        df.groupby("Cluster")
        .agg(
            Total_Customers=("CLIENTNUM", "count"),
            Attrited_Customers=("Attrition_Label", "sum"),
            Attrition_Rate=("Attrition_Label", "mean"),
        )
        .reset_index()
    )
    summary["Attrition_Rate"] = summary["Attrition_Rate"].round(4)
    return summary


# ---------------------------------------------------------------------------
# 8. Vẽ biểu đồ so sánh
# ---------------------------------------------------------------------------

def plot_metrics_comparison(comparison_df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing Silhouette, ARI, NMI across algorithms."""
    metrics_to_plot = ["silhouette_score", "adjusted_rand_index", "normalized_mutual_info"]
    labels_vi = ["Silhouette Score", "Adjusted Rand Index", "Normalized Mutual Info"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Clustering Algorithm Comparison", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    algorithms = comparison_df["algorithm"].tolist()

    for ax, metric, label_vi in zip(axes, metrics_to_plot, labels_vi):
        values = comparison_df[metric].tolist()
        bars = ax.bar(algorithms, values, color=colors[: len(algorithms)], edgecolor="black", linewidth=0.5)
        ax.set_title(label_vi, fontsize=11)
        ax.set_ylabel("Score")

        # Show value above each bar.
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=8,
                )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pca_scatter(
    X_pca: np.ndarray,
    all_labels: dict[str, np.ndarray],
    output_dir: Path,
):
    """PCA 2D scatter plot for each algorithm."""
    n_algos = len(all_labels)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5))
    fig.suptitle("Cluster Distribution on First 2 PCA Components", fontsize=14, fontweight="bold")

    if n_algos == 1:
        axes = [axes]

    for ax, (name, labels) in zip(axes, all_labels.items()):
        unique = sorted(set(labels))
        cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(unique), 2))

        for i, cluster_id in enumerate(unique):
            mask = labels == cluster_id
            label_text = f"Noise ({mask.sum()})" if cluster_id == -1 else f"Cluster {cluster_id} ({mask.sum()})"
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                s=4, alpha=0.5, color=cmap(i), label=label_text,
            )

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, markerscale=3, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / "pca_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 9. Luồng chạy chính
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    X_df, labels_df = load_data(args.features, args.labels)
    true_labels = labels_df["Attrition_Label"].values

    # --- PCA reduction (same pipeline as Chameleon) ---
    X_pca = reduce_dimension(X_df, args.pca_components)

    # --- Load pre-trained Chameleon results ---
    chameleon_labels = load_chameleon_labels(args.chameleon_results)

    # Kiểm tra độ dài chameleon_labels khớp với X_pca.
    if chameleon_labels is not None and len(chameleon_labels) != len(X_pca):
        print(
            f"[WARN] Chameleon labels length ({len(chameleon_labels)}) "
            f"!= X_pca length ({len(X_pca)}). Skipping Chameleon."
        )
        chameleon_labels = None

    # Determine n_clusters: CLI arg > Chameleon result > default=2.
    if args.n_clusters is not None:
        n_clusters = args.n_clusters
    elif chameleon_labels is not None:
        # Dùng set - {-1} để nhất quán với hàm evaluate() (loại bỏ noise label).
        n_clusters = len(set(chameleon_labels) - {-1})
    else:
        n_clusters = 2  # Default.

    print(f"\nNumber of clusters for K-Means and HAC: {n_clusters}")

    # --- Dict to store cluster labels for each algorithm ---
    all_labels: dict[str, np.ndarray] = {}
    results: list[dict] = []

    # 1) Chameleon (from pre-trained results)
    if chameleon_labels is not None:
        all_labels["Chameleon"] = chameleon_labels
        metrics = evaluate(X_pca, chameleon_labels, true_labels)
        metrics["algorithm"] = "Chameleon"
        results.append(metrics)
        print(f"\n[Chameleon] Silhouette={metrics['silhouette_score']:.4f}  "
              f"ARI={metrics['adjusted_rand_index']:.4f}  "
              f"NMI={metrics['normalized_mutual_info']:.4f}")

    # 2) K-Means
    print("\nTraining K-Means...")
    km_labels = train_kmeans(X_pca, n_clusters)
    all_labels["K-Means"] = km_labels
    metrics = evaluate(X_pca, km_labels, true_labels)
    metrics["algorithm"] = "K-Means"
    results.append(metrics)
    print(f"[K-Means]   Silhouette={metrics['silhouette_score']:.4f}  "
          f"ARI={metrics['adjusted_rand_index']:.4f}  "
          f"NMI={metrics['normalized_mutual_info']:.4f}")

    # 3) HAC
    print("\nTraining HAC (Ward linkage)...")
    hac_labels = train_hac(X_pca, n_clusters, linkage="ward")
    all_labels["HAC"] = hac_labels
    metrics = evaluate(X_pca, hac_labels, true_labels)
    metrics["algorithm"] = "HAC"
    results.append(metrics)
    print(f"[HAC]       Silhouette={metrics['silhouette_score']:.4f}  "
          f"ARI={metrics['adjusted_rand_index']:.4f}  "
          f"NMI={metrics['normalized_mutual_info']:.4f}")

    # --- Comparison summary table ---
    col_order = [
        "algorithm", "n_clusters",
        "silhouette_score", "adjusted_rand_index", "normalized_mutual_info",
    ]
    comparison_df = pd.DataFrame(results)[col_order]

    comparison_path = output_dir / "comparison_metrics.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved: {comparison_path}")

    print("\n" + "=" * 70)
    print("CLUSTERING ALGORITHM COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("=" * 70)

    # --- Churn summary for each algorithm ---
    for name, cluster_lbl in all_labels.items():
        summary = cluster_summary(cluster_lbl, labels_df)
        safe_name = name.lower().replace("-", "").replace(" ", "_")
        summary_path = output_dir / f"{safe_name}_cluster_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n[{name}] Cluster summary -> {summary_path}")
        print(summary.to_string(index=False))

    # --- Charts ---
    print("\nGenerating charts...")
    plot_metrics_comparison(comparison_df, output_dir)
    plot_pca_scatter(X_pca, all_labels, output_dir)

    print("\nDone! All outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
