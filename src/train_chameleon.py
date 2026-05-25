# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Đường dẫn mặc định tới các file.
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "bank_churners_features.csv"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "processed" / "bank_churners_labels.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chameleon"


def parse_args():
    # Tạo bộ đọc tham số dòng lệnh.
    parser = argparse.ArgumentParser(description="Train Chameleon clustering model.")

    # Cho phép người dùng truyền đường dẫn file features.
    # Nếu không truyền thì dùng DEFAULT_FEATURES.
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)

    # Cho phép người dùng truyền đường dẫn file labels.
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)

    # Cho phép người dùng truyền thư mục output.
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    # Số chiều PCA muốn giữ lại.
    parser.add_argument("--pca-components", type=int, default=10)

    # Số láng giềng gần nhất khi xây dựng kNN graph.
    parser.add_argument("--knn", type=int, default=15)

    # Số cụm nhỏ ban đầu.
    parser.add_argument("--initial-clusters", type=int, default=30)

    # Số cụm cuối cùng sau khi gộp.
    parser.add_argument("--final-clusters", type=int, default=2)

    # Tham số alpha trong công thức Chameleon:
    # score = RI * (RC ** alpha)
    parser.add_argument("--alpha", type=float, default=1.0)

    # Trả về toàn bộ tham số đã đọc.
    return parser.parse_args()

def load_data(features_path: Path, labels_path: Path):
    # Kiểm tra file features có tồn tại không.
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    # Kiểm tra file labels có tồn tại không.
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    # Đọc ma trận đặc trưng đã tiền xử lý.
    X = pd.read_csv(features_path)

    # Đọc file nhãn churn.
    labels = pd.read_csv(labels_path)

    # Kiểm tra số dòng của features và labels có khớp không.
    # Nếu không khớp thì không thể gán cụm cho đúng khách hàng.
    if len(X) != len(labels):
        raise ValueError("Features and labels have different number of rows.")

    return X, labels

# Giảm chiều PCA
def reduce_dimension(X: pd.DataFrame, n_components: int):
    n_components = min(n_components, X.shape[1])

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    explained_ratio = pca.explained_variance_ratio_.sum()

    print(f"PCA components: {n_components}")
    print(f"Explained variance ratio: {explained_ratio:.4f}")

    return X_pca, explained_ratio


# Xây dựng graph
def build_knn_graph(X: np.ndarray, k: int):
    # Xây dựng đồ thị kNN.
    # Mỗi dòng dữ liệu là một node.
    # Mỗi node nối với k điểm gần nhất.
    # mode="distance" nghĩa là trọng số cạnh ban đầu là khoảng cách.
    graph = kneighbors_graph(
        X,
        n_neighbors=k,
        mode="distance",
        include_self=False,
    )

    # Làm đồ thị đối xứng.
    # Vì kNN ban đầu có thể là một chiều:
    # A chọn B là láng giềng nhưng B chưa chắc chọn A.
    graph = 0.5 * (graph + graph.T)

    # Sao chép graph để chuyển từ distance sang similarity.
    similarity = graph.copy()

    # Chuyển khoảng cách thành độ tương đồng.
    # Distance càng nhỏ thì similarity càng lớn.
    similarity.data = 1 / (1 + similarity.data)

    return similarity



def initial_partition(similarity_graph, n_clusters: int):
    # Tạo cụm nhỏ ban đầu bằng Spectral Clustering.
    clustering = SpectralClustering(
        n_clusters=n_clusters,

        # affinity="precomputed" nghĩa là ta đã truyền sẵn ma trận similarity.
        affinity="precomputed",

        # Sau khi biến đổi phổ, sklearn dùng KMeans để gán nhãn cuối.
        assign_labels="kmeans",

        random_state=42,
    )

    # Fit mô hình và lấy nhãn cụm ban đầu.
    labels = clustering.fit_predict(similarity_graph)

    return labels

def cluster_internal_weight(graph, cluster_indices):
    # Lấy subgraph gồm các điểm nằm trong cùng một cụm.
    subgraph = graph[cluster_indices][:, cluster_indices]

    # Tổng trọng số cạnh bên trong cụm.
    # Chia 2 vì graph đối xứng nên mỗi cạnh bị tính 2 lần.
    return subgraph.sum() / 2

def cluster_between_weight(graph, cluster_a, cluster_b):
    # Lấy các cạnh nối từ cụm A sang cụm B.
    subgraph = graph[cluster_a][:, cluster_b]

    # Tổng trọng số liên kết giữa hai cụm.
    return subgraph.sum()

def chameleon_score(graph, cluster_a, cluster_b, alpha=1.0):
    # Tính tổng trọng số liên kết giữa hai cụm A và B.
    weight_ab = cluster_between_weight(graph, cluster_a, cluster_b)

    # Tính tổng trọng số liên kết nội bộ của cụm A.
    weight_a = cluster_internal_weight(graph, cluster_a)

    # Tính tổng trọng số liên kết nội bộ của cụm B.
    weight_b = cluster_internal_weight(graph, cluster_b)

    # Kích thước từng cụm.
    size_a = len(cluster_a)
    size_b = len(cluster_b)

    # Nếu không có liên kết giữa hai cụm hoặc cụm không có liên kết nội bộ, thì không gộp hai cụm này.
    if weight_ab == 0 or weight_a == 0 or weight_b == 0:
        return 0

    # 1. Relative Interconnectivit:  RI đo mức độ liên kết giữa hai cụm so với liên kết nội bộ của chúng.
    ri = weight_ab / ((weight_a + weight_b) / 2)

    # 2. Relative Closeness: 
    # Mật độ liên kết nội bộ của cụm A.
    density_a = weight_a / size_a

    # Mật độ liên kết nội bộ của cụm B.
    density_b = weight_b / size_b

    # Mật độ nội bộ trung bình của hai cụm.
    avg_internal_density = (density_a + density_b) / 2

    # Mật độ liên kết giữa hai cụm.
    inter_density = weight_ab / (size_a + size_b)

    # Nếu mật độ nội bộ bằng 0 thì không thể tính RC.
    if avg_internal_density == 0:
        return 0

    # RC đo độ gần giữa hai cụm so với độ gần nội bộ của chúng.
    rc = inter_density / avg_internal_density

    # 3. Chameleon merge score
    score = ri * (rc ** alpha)

    return score

def merge_clusters(graph, initial_labels, final_clusters: int, alpha: float = 1.0):
    # Tạo dictionary lưu danh sách điểm thuộc từng cụm ban đầu.
    # Key là mã cụm, value là danh sách index của các điểm trong cụm đó.
    clusters = {
        cluster_id: np.where(initial_labels == cluster_id)[0].tolist()
        for cluster_id in np.unique(initial_labels)
    }

    # ID mới cho cụm sau khi gộp.
    next_cluster_id = max(clusters.keys()) + 1

    # Lặp cho đến khi số cụm còn lại bằng final_clusters.
    while len(clusters) > final_clusters:
        best_pair = None
        best_score = -1

        # Lấy danh sách cụm hiện tại.
        cluster_items = list(clusters.items())

        # Xét tất cả cặp cụm có thể gộp.
        for (id_a, cluster_a), (id_b, cluster_b) in combinations(cluster_items, 2):
            # Tính điểm Chameleon cho cặp cụm.
            score = chameleon_score(graph, cluster_a, cluster_b, alpha=alpha)

            # Nếu điểm cao hơn điểm tốt nhất hiện tại, lưu lại cặp cụm này.
            if score > best_score:
                best_score = score
                best_pair = (id_a, id_b)

        # Nếu không tìm được cặp nào để gộp thì dừng.
        if best_pair is None:
            break

        # Lấy ID của hai cụm tốt nhất.
        id_a, id_b = best_pair

        # Gộp danh sách điểm của hai cụm.
        merged_cluster = clusters[id_a] + clusters[id_b]

        # Xóa hai cụm cũ.
        del clusters[id_a]
        del clusters[id_b]

        # Thêm cụm mới sau khi gộp.
        clusters[next_cluster_id] = merged_cluster
        next_cluster_id += 1

        print(f"Remaining clusters: {len(clusters)} | Best score: {best_score:.6f}")

    # Tạo mảng nhãn cụm cuối cùng cho toàn bộ dữ liệu.
    final_labels = np.empty(len(initial_labels), dtype=int)

    # Gán lại nhãn cụm từ 0, 1, 2, ...
    for new_label, indices in enumerate(clusters.values()):
        final_labels[indices] = new_label

    return final_labels

def evaluate_clustering(X, cluster_labels, true_labels):
    # Dictionary lưu kết quả đánh giá.
    result = {}

    # Số cụm cuối cùng.
    result["n_clusters"] = int(len(np.unique(cluster_labels)))

    # Silhouette đo chất lượng phân cụm dựa trên khoảng cách trong dữ liệu.
    result["silhouette_score"] = float(silhouette_score(X, cluster_labels))

    # ARI so sánh cụm tìm được với nhãn churn thật.
    result["adjusted_rand_index"] = float(adjusted_rand_score(true_labels, cluster_labels))

    # NMI đo mức độ thông tin chung giữa nhãn cụm và nhãn churn thật.
    result["normalized_mutual_info"] = float(
        normalized_mutual_info_score(true_labels, cluster_labels)
    )

    return result

def build_cluster_summary(result_df):
    # Gom dữ liệu theo cụm để phân tích churn trong từng cụm.
    summary = (
        result_df.groupby("Cluster")
        .agg(
            # Đếm số khách hàng trong từng cụm.
            Total_Customers=("CLIENTNUM", "count"),

            # Vì Attrition_Label: Existing = 0, Attrited = 1, nên tổng Attrition_Label chính là số khách hàng churn.
            Attrited_Customers=("Attrition_Label", "sum"),

            # Trung bình Attrition_Label chính là tỷ lệ churn.
            Attrition_Rate=("Attrition_Label", "mean"),
        )
        .reset_index()
    )

    # Làm tròn tỷ lệ churn 4 chữ số thập phân.
    summary["Attrition_Rate"] = summary["Attrition_Rate"].round(4)

    return summary

def main():
    # Đọc tham số dòng lệnh.
    args = parse_args()

    # Lấy thư mục output.
    output_dir = args.output_dir

    # Tạo thư mục output nếu chưa tồn tại.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu features và labels.
    X, labels = load_data(args.features, args.labels)

    # Giảm chiều dữ liệu bằng PCA.
    X_pca, explained_ratio = reduce_dimension(X, args.pca_components)

    # Xây dựng đồ thị kNN từ dữ liệu PCA.
    graph = build_knn_graph(X_pca, args.knn)

    print("Creating initial clusters...")

    # Tạo các cụm nhỏ ban đầu bằng Spectral Clustering.
    initial_labels = initial_partition(graph, args.initial_clusters)

    print("Merging clusters using Chameleon logic...")

    # Gộp cụm theo logic Chameleon cho đến khi còn final_clusters cụm.
    final_cluster_labels = merge_clusters(
        graph=graph,
        initial_labels=initial_labels,
        final_clusters=args.final_clusters,
        alpha=args.alpha,
    )

    # Lấy nhãn churn thật để đánh giá sau phân cụm.
    true_labels = labels["Attrition_Label"].values

    # Tính các chỉ số đánh giá clustering.
    metrics = evaluate_clustering(
        X=X_pca,
        cluster_labels=final_cluster_labels,
        true_labels=true_labels,
    )

    # Sao chép file labels để gắn thêm nhãn cụm.
    result_df = labels.copy()

    # Thêm cột Cluster vào kết quả.
    result_df["Cluster"] = final_cluster_labels

    # Tạo bảng tổng hợp số khách hàng và churn rate theo cụm.
    summary_df = build_cluster_summary(result_df)

    # Khai báo đường dẫn các file output.
    result_path = output_dir / "chameleon_cluster_results.csv"
    summary_path = output_dir / "chameleon_summary.csv"
    metrics_path = output_dir / "chameleon_metrics.csv"

    # Lưu kết quả từng khách hàng kèm cụm.
    result_df.to_csv(result_path, index=False)

    # Lưu bảng tổng hợp theo cụm.
    summary_df.to_csv(summary_path, index=False)

    # Lưu các chỉ số đánh giá.
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\nTraining completed successfully.")
    print(f"Result file: {result_path}")
    print(f"Summary file: {summary_path}")
    print(f"Metrics file: {metrics_path}")

    print("\nEvaluation metrics:")

    # In từng chỉ số đánh giá ra màn hình.
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nCluster summary:")

    # In bảng tổng hợp cụm ra màn hình.
    print(summary_df)


if __name__ == "__main__":
    main()
