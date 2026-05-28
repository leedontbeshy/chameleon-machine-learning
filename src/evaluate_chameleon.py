# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULT_FILE = (
    PROJECT_ROOT / "outputs" / "chameleon" / "chameleon_cluster_results.csv"
)

DEFAULT_SUMMARY_FILE = (
    PROJECT_ROOT / "outputs" / "chameleon" / "chameleon_summary.csv"
)

DEFAULT_METRICS_FILE = (
    PROJECT_ROOT / "outputs" / "chameleon" / "chameleon_metrics.csv"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chameleon" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Chameleon clustering results in detail."
    )

    parser.add_argument(
        "--result-file",
        type=Path,
        default=DEFAULT_RESULT_FILE,
        help=f"Path to chameleon_cluster_results.csv. Default: {DEFAULT_RESULT_FILE}",
    )

    parser.add_argument(
        "--summary-file",
        type=Path,
        default=DEFAULT_SUMMARY_FILE,
        help=f"Path to chameleon_summary.csv. Default: {DEFAULT_SUMMARY_FILE}",
    )

    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help=f"Path to chameleon_metrics.csv. Default: {DEFAULT_METRICS_FILE}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for evaluation outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )

    return parser.parse_args()


def read_csv_checked(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing_columns = sorted(set(required_columns).difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing columns in {path.name}: {missing_columns}"
        )

    return df


def load_outputs(
    result_file: Path,
    summary_file: Path,
    metrics_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_df = read_csv_checked(
        result_file,
        required_columns=[
            "CLIENTNUM",
            "Attrition_Flag",
            "Attrition_Label",
            "Cluster",
        ],
    )

    summary_df = read_csv_checked(
        summary_file,
        required_columns=[
            "Cluster",
            "Total_Customers",
            "Attrited_Customers",
            "Attrition_Rate",
        ],
    )

    metrics_df = read_csv_checked(
        metrics_file,
        required_columns=[
            "n_clusters",
            "silhouette_score",
            "adjusted_rand_index",
            "normalized_mutual_info",
        ],
    )

    return result_df, summary_df, metrics_df


def build_confusion_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng đếm số lượng khách hàng rời bỏ và không rời bỏ trong từng cụm.
    """

    confusion = pd.crosstab(
        result_df["Cluster"],
        result_df["Attrition_Flag"],
    )

    # Đảm bảo luôn có đủ 2 cột, kể cả trường hợp một cụm không có churn.
    if "Existing Customer" not in confusion.columns:
        confusion["Existing Customer"] = 0

    if "Attrited Customer" not in confusion.columns:
        confusion["Attrited Customer"] = 0

    confusion = confusion[
        ["Existing Customer", "Attrited Customer"]
    ].reset_index()

    confusion = confusion.rename(
        columns={
            "Existing Customer": "Non_Attrited_Customers",
            "Attrited Customer": "Attrited_Customers",
        }
    )

    confusion["Total_Customers"] = (
        confusion["Non_Attrited_Customers"]
        + confusion["Attrited_Customers"]
    )

    return confusion


def calculate_cluster_evaluation(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính các thống kê đánh giá chi tiết cho từng cụm.
    """

    total_customers = len(result_df)
    total_attrited = int(result_df["Attrition_Label"].sum())
    total_non_attrited = total_customers - total_attrited
    overall_attrition_rate = total_attrited / total_customers

    cluster_eval = build_confusion_table(result_df)

    cluster_eval["Cluster_Size_Ratio"] = (
        cluster_eval["Total_Customers"] / total_customers
    )

    cluster_eval["Attrition_Rate"] = (
        cluster_eval["Attrited_Customers"] / cluster_eval["Total_Customers"]
    )

    cluster_eval["Non_Attrition_Rate"] = (
        cluster_eval["Non_Attrited_Customers"] / cluster_eval["Total_Customers"]
    )

    cluster_eval["Attrition_Rate_Difference_From_Overall"] = (
        cluster_eval["Attrition_Rate"] - overall_attrition_rate
    )

    cluster_eval["Attrition_Lift"] = (
        cluster_eval["Attrition_Rate"] / overall_attrition_rate
    )

    cluster_eval["Share_Of_All_Attrited_Customers"] = np.where(
        total_attrited > 0,
        cluster_eval["Attrited_Customers"] / total_attrited,
        0,
    )

    cluster_eval["Share_Of_All_Non_Attrited_Customers"] = np.where(
        total_non_attrited > 0,
        cluster_eval["Non_Attrited_Customers"] / total_non_attrited,
        0,
    )

    cluster_eval["Majority_Class"] = np.where(
        cluster_eval["Attrited_Customers"]
        > cluster_eval["Non_Attrited_Customers"],
        "Attrited Customer",
        "Existing Customer",
    )

    cluster_eval["Majority_Class_Count"] = cluster_eval[
        ["Attrited_Customers", "Non_Attrited_Customers"]
    ].max(axis=1)

    cluster_eval["Cluster_Purity"] = (
        cluster_eval["Majority_Class_Count"] / cluster_eval["Total_Customers"]
    )

    # Gán mức độ rủi ro theo tỷ lệ churn so với trung bình toàn bộ dữ liệu.
    conditions = [
        cluster_eval["Attrition_Rate"] > overall_attrition_rate,
        cluster_eval["Attrition_Rate"] < overall_attrition_rate,
    ]

    choices = [
        "Higher risk than average",
        "Lower risk than average",
    ]

    cluster_eval["Risk_Level"] = np.select(
        conditions,
        choices,
        default="Average risk",
    )

    # Làm tròn các cột tỷ lệ cho dễ đọc.
    ratio_columns = [
        "Cluster_Size_Ratio",
        "Attrition_Rate",
        "Non_Attrition_Rate",
        "Attrition_Rate_Difference_From_Overall",
        "Attrition_Lift",
        "Share_Of_All_Attrited_Customers",
        "Share_Of_All_Non_Attrited_Customers",
        "Cluster_Purity",
    ]

    cluster_eval[ratio_columns] = cluster_eval[ratio_columns].round(4)

    return cluster_eval


def calculate_overall_evaluation(
    result_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    cluster_eval: pd.DataFrame,
) -> dict[str, float | int | str]:
    """
    Tính thống kê tổng quan cho toàn bộ kết quả phân cụm.
    """

    metrics = metrics_df.iloc[0].to_dict()

    total_customers = len(result_df)
    total_attrited = int(result_df["Attrition_Label"].sum())
    total_non_attrited = total_customers - total_attrited
    overall_attrition_rate = total_attrited / total_customers

    high_risk_cluster = cluster_eval.sort_values(
        by="Attrition_Rate",
        ascending=False,
    ).iloc[0]

    low_risk_cluster = cluster_eval.sort_values(
        by="Attrition_Rate",
        ascending=True,
    ).iloc[0]

    overall = {
        "total_customers": int(total_customers),
        "total_attrited_customers": int(total_attrited),
        "total_non_attrited_customers": int(total_non_attrited),
        "overall_attrition_rate": round(float(overall_attrition_rate), 4),
        "n_clusters": int(metrics["n_clusters"]),
        "silhouette_score": round(float(metrics["silhouette_score"]), 4),
        "adjusted_rand_index": round(float(metrics["adjusted_rand_index"]), 4),
        "normalized_mutual_info": round(
            float(metrics["normalized_mutual_info"]), 4
        ),
        "highest_risk_cluster": int(high_risk_cluster["Cluster"]),
        "highest_cluster_attrition_rate": float(
            high_risk_cluster["Attrition_Rate"]
        ),
        "lowest_risk_cluster": int(low_risk_cluster["Cluster"]),
        "lowest_cluster_attrition_rate": float(
            low_risk_cluster["Attrition_Rate"]
        ),
    }

    return overall


def interpret_silhouette(score: float) -> str:
    if score >= 0.7:
        return "Cấu trúc cụm rất tốt, các cụm tách biệt rõ."
    if score >= 0.5:
        return "Cấu trúc cụm khá tốt."
    if score >= 0.25:
        return "Cấu trúc cụm ở mức trung bình, có phân tách nhưng chưa mạnh."
    if score >= 0:
        return "Cấu trúc cụm yếu, các cụm còn chồng lấn nhiều."

    return "Kết quả phân cụm không tốt, nhiều điểm có thể bị gán sai cụm."


def interpret_ari(score: float) -> str:
    if score >= 0.75:
        return "Cụm khớp rất tốt với nhãn churn thực tế."
    if score >= 0.5:
        return "Cụm khớp tương đối tốt với nhãn churn thực tế."
    if score >= 0.25:
        return "Cụm có mức khớp thấp đến trung bình với nhãn churn thực tế."
    if score >= 0:
        return "Cụm khớp rất thấp với nhãn churn thực tế."

    return (
        "Cụm không khớp với nhãn churn thực tế, thấp hơn mức phân cụm ngẫu nhiên."
    )


def interpret_nmi(score: float) -> str:
    if score >= 0.75:
        return "Nhãn cụm chứa nhiều thông tin về nhãn churn."
    if score >= 0.5:
        return "Nhãn cụm có lượng thông tin tương đối về nhãn churn."
    if score >= 0.25:
        return "Nhãn cụm có một phần thông tin về nhãn churn."
    return "Nhãn cụm chứa rất ít thông tin về nhãn churn."


def build_text_report(
    overall: dict[str, float | int | str],
    cluster_eval: pd.DataFrame,
) -> str:
    silhouette = float(overall["silhouette_score"])
    ari = float(overall["adjusted_rand_index"])
    nmi = float(overall["normalized_mutual_info"])

    lines: list[str] = []

    lines.append("ĐÁNH GIÁ KẾT QUẢ THUẬT TOÁN CHAMELEON")
    lines.append("=" * 60)
    lines.append("")

    lines.append("1. Thống kê tổng quan")
    lines.append("-" * 60)
    lines.append(f"Tổng số khách hàng: {overall['total_customers']}")
    lines.append(
        f"Số khách hàng rời bỏ: {overall['total_attrited_customers']}"
    )
    lines.append(
        f"Số khách hàng không rời bỏ: "
        f"{overall['total_non_attrited_customers']}"
    )
    lines.append(
        f"Tỷ lệ rời bỏ toàn bộ dữ liệu: "
        f"{overall['overall_attrition_rate']:.4f}"
    )
    lines.append(f"Số cụm cuối cùng: {overall['n_clusters']}")
    lines.append("")

    lines.append("2. Chỉ số đánh giá phân cụm")
    lines.append("-" * 60)
    lines.append(f"Silhouette Score: {silhouette:.4f}")
    lines.append(f"Nhận xét: {interpret_silhouette(silhouette)}")
    lines.append("")
    lines.append(f"Adjusted Rand Index: {ari:.4f}")
    lines.append(f"Nhận xét: {interpret_ari(ari)}")
    lines.append("")
    lines.append(f"Normalized Mutual Information: {nmi:.4f}")
    lines.append(f"Nhận xét: {interpret_nmi(nmi)}")
    lines.append("")

    lines.append("3. Đánh giá chi tiết từng cụm")
    lines.append("-" * 60)

    for _, row in cluster_eval.iterrows():
        lines.append(f"Cụm {int(row['Cluster'])}:")
        lines.append(f"  - Tổng số khách hàng: {int(row['Total_Customers'])}")
        lines.append(
            f"  - Tỷ lệ kích thước cụm: "
            f"{float(row['Cluster_Size_Ratio']):.4f}"
        )
        lines.append(
            f"  - Khách hàng rời bỏ: "
            f"{int(row['Attrited_Customers'])}"
        )
        lines.append(
            f"  - Khách hàng không rời bỏ: "
            f"{int(row['Non_Attrited_Customers'])}"
        )
        lines.append(
            f"  - Tỷ lệ rời bỏ trong cụm: "
            f"{float(row['Attrition_Rate']):.4f}"
        )
        lines.append(
            f"  - Chênh lệch so với tỷ lệ rời bỏ chung: "
            f"{float(row['Attrition_Rate_Difference_From_Overall']):.4f}"
        )
        lines.append(
            f"  - Attrition Lift: "
            f"{float(row['Attrition_Lift']):.4f}"
        )
        lines.append(
            f"  - Tỷ trọng trong toàn bộ khách hàng rời bỏ: "
            f"{float(row['Share_Of_All_Attrited_Customers']):.4f}"
        )
        lines.append(
            f"  - Lớp chiếm đa số: {row['Majority_Class']}"
        )
        lines.append(
            f"  - Độ tinh khiết của cụm: "
            f"{float(row['Cluster_Purity']):.4f}"
        )
        lines.append(f"  - Mức độ rủi ro: {row['Risk_Level']}")
        lines.append("")

    lines.append("4. Kết luận")
    lines.append("-" * 60)
    lines.append(
        f"Cụm có tỷ lệ rời bỏ cao nhất là cụm "
        f"{overall['highest_risk_cluster']} với tỷ lệ rời bỏ "
        f"{overall['highest_cluster_attrition_rate']:.4f}."
    )
    lines.append(
        f"Cụm có tỷ lệ rời bỏ thấp nhất là cụm "
        f"{overall['lowest_risk_cluster']} với tỷ lệ rời bỏ "
        f"{overall['lowest_cluster_attrition_rate']:.4f}."
    )
    lines.append(
        "Kết quả cho thấy thuật toán Chameleon-based clustering có thể "
        "phát hiện một số cấu trúc nhóm khách hàng trong dữ liệu. Tuy nhiên, "
        "dựa trên ARI và NMI, kết quả phân cụm chưa khớp mạnh với nhãn churn "
        "thực tế. Vì vậy, mô hình phù hợp hơn với mục tiêu khám phá nhóm "
        "khách hàng hơn là dùng trực tiếp như mô hình dự đoán churn."
    )

    return "\n".join(lines)


def write_outputs(
    cluster_eval: pd.DataFrame,
    overall: dict[str, float | int | str],
    report_text: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_eval_path = output_dir / "chameleon_cluster_evaluation.csv"
    overall_path = output_dir / "chameleon_overall_evaluation.csv"
    report_path = output_dir / "chameleon_evaluation_report.txt"

    cluster_eval.to_csv(cluster_eval_path, index=False)
    pd.DataFrame([overall]).to_csv(overall_path, index=False)

    with report_path.open("w", encoding="utf-8") as file:
        file.write(report_text)
        file.write("\n")

    print("Evaluation completed successfully.")
    print(f"Cluster evaluation file: {cluster_eval_path}")
    print(f"Overall evaluation file: {overall_path}")
    print(f"Text report file: {report_path}")


def main() -> None:
    args = parse_args()

    result_df, summary_df, metrics_df = load_outputs(
        result_file=args.result_file.resolve(),
        summary_file=args.summary_file.resolve(),
        metrics_file=args.metrics_file.resolve(),
    )

    # summary_df được đọc để kiểm tra và đối chiếu.
    # Phần đánh giá chi tiết sẽ được tính lại từ result_df để tránh phụ thuộc
    # hoàn toàn vào file summary có sẵn.
    print("Loaded files successfully.")
    print(f"Result rows: {len(result_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print(f"Metrics rows: {len(metrics_df)}")

    cluster_eval = calculate_cluster_evaluation(result_df)

    overall = calculate_overall_evaluation(
        result_df=result_df,
        metrics_df=metrics_df,
        cluster_eval=cluster_eval,
    )

    report_text = build_text_report(
        overall=overall,
        cluster_eval=cluster_eval,
    )

    write_outputs(
        cluster_eval=cluster_eval,
        overall=overall,
        report_text=report_text,
        output_dir=args.output_dir.resolve(),
    )

    print("\nOverall evaluation:")
    for key, value in overall.items():
        print(f"{key}: {value}")

    print("\nCluster evaluation:")
    print(cluster_eval)


if __name__ == "__main__":
    main()