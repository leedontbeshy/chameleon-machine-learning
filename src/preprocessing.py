"""Preprocess the BankChurners dataset for Chameleon clustering.

The output feature matrix intentionally excludes identifiers, the churn label,
and Kaggle's precomputed Naive Bayes columns so the clustering step does not
receive answer-like information.

Run from the project root:
    python src/preprocessing.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "BankChurners.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ID_COLUMN = "CLIENTNUM"
TARGET_COLUMN = "Attrition_Flag"
REDUNDANT_FEATURE_COLUMNS = ["Avg_Open_To_Buy"]
NAIVE_BAYES_PREFIX = "Naive_Bayes_Classifier_"

CATEGORICAL_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

ATTRITION_LABEL_MAP = {
    "Existing Customer": 0,
    "Attrited Customer": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BankChurners.csv into Chameleon-ready CSV files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the raw BankChurners CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for processed outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def read_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    return pd.read_csv(input_path)


def find_naive_bayes_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith(NAIVE_BAYES_PREFIX)]


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        *REDUNDANT_FEATURE_COLUMNS,
        *CATEGORICAL_COLUMNS,
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    naive_bayes_columns = find_naive_bayes_columns(df)
    if len(naive_bayes_columns) != 2:
        raise ValueError(
            "Expected exactly 2 Kaggle Naive Bayes columns, "
            f"found {len(naive_bayes_columns)}: {naive_bayes_columns}"
        )


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = df[[ID_COLUMN, TARGET_COLUMN]].copy()
    labels["Attrition_Label"] = labels[TARGET_COLUMN].map(ATTRITION_LABEL_MAP)

    if labels["Attrition_Label"].isna().any():
        unknown_labels = sorted(
            set(labels.loc[labels["Attrition_Label"].isna(), TARGET_COLUMN])
        )
        raise ValueError(f"Unknown attrition labels found: {unknown_labels}")

    labels["Attrition_Label"] = labels["Attrition_Label"].astype(int)
    return labels


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    naive_bayes_columns = find_naive_bayes_columns(df)
    excluded_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
        *naive_bayes_columns,
        *REDUNDANT_FEATURE_COLUMNS,
    ]

    feature_source = df.drop(columns=excluded_columns).copy()

    missing_feature_values = feature_source.isna().sum()
    columns_with_missing = missing_feature_values[missing_feature_values > 0]
    if not columns_with_missing.empty:
        raise ValueError(
            "Missing feature values found. This pipeline keeps 'Unknown' as a "
            "category but does not impute actual missing cells: "
            f"{columns_with_missing.to_dict()}"
        )

    numeric_columns = [
        column for column in feature_source.columns if column not in CATEGORICAL_COLUMNS
    ]
    numeric_features = feature_source[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )

    scaler = RobustScaler()
    scaled_numeric = pd.DataFrame(
        scaler.fit_transform(numeric_features),
        columns=numeric_columns,
        index=feature_source.index,
    )

    categorical_features = feature_source[CATEGORICAL_COLUMNS].astype(str)
    encoded_categorical = pd.get_dummies(
        categorical_features,
        columns=CATEGORICAL_COLUMNS,
        dtype=np.int64,
    )

    features = pd.concat([scaled_numeric, encoded_categorical], axis=1)
    features = features.apply(pd.to_numeric, errors="raise")

    metadata = {
        "excluded_columns": excluded_columns,
        "numeric_scaled_columns": numeric_columns,
        "categorical_encoded_columns": CATEGORICAL_COLUMNS,
        "categorical_output_columns": encoded_categorical.columns.tolist(),
        "feature_columns": features.columns.tolist(),
        "scaler": "sklearn.preprocessing.RobustScaler",
        "scaler_center_median": dict(zip(numeric_columns, scaler.center_.tolist())),
        "scaler_scale_iqr": dict(zip(numeric_columns, scaler.scale_.tolist())),
    }

    return features, metadata


def count_unknown_values(df: pd.DataFrame) -> dict[str, int]:
    return {
        column: int((df[column].astype(str).str.strip() == "Unknown").sum())
        for column in CATEGORICAL_COLUMNS
    }


def value_counts_as_ints(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="raise")
        summary[column] = {
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
            "mean": round(float(values.mean()), 6),
            "median": round(float(values.median()), 6),
            "std": round(float(values.std(ddof=0)), 6),
        }
    return summary


def build_metadata(
    raw_df: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_metadata: dict[str, Any],
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    raw_missing_counts = {
        column: int(count)
        for column, count in raw_df.isna().sum().items()
        if int(count) > 0
    }

    category_levels = {
        column: sorted(raw_df[column].astype(str).unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }

    metadata = {
        "source_dataset": str(input_path),
        "output_directory": str(output_dir),
        "raw_shape": {
            "rows": int(raw_df.shape[0]),
            "columns": int(raw_df.shape[1]),
        },
        "processed_shape": {
            "feature_rows": int(features.shape[0]),
            "feature_columns": int(features.shape[1]),
            "label_rows": int(labels.shape[0]),
        },
        "data_quality": {
            "duplicate_rows": int(raw_df.duplicated().sum()),
            "duplicate_clientnum": int(raw_df[ID_COLUMN].duplicated().sum()),
            "missing_values_by_column": raw_missing_counts,
            "unknown_counts": count_unknown_values(raw_df),
        },
        "label_distribution": value_counts_as_ints(raw_df[TARGET_COLUMN]),
        "attrition_label_map": ATTRITION_LABEL_MAP,
        "category_levels": category_levels,
        "numeric_summary_raw_used_features": numeric_summary(
            raw_df, feature_metadata["numeric_scaled_columns"]
        ),
        "preprocessing": {
            "label_usage": (
                "Attrition_Flag is excluded from clustering features and kept "
                "only for cluster interpretation/evaluation."
            ),
            "unknown_handling": "Unknown values are kept as explicit categories.",
            "categorical_encoding": "One-hot encoding with pandas.get_dummies.",
            "numeric_scaling": "RobustScaler, using median and IQR.",
        },
        "columns": feature_metadata,
        "generated_files": {
            "features": str(output_dir / "bank_churners_features.csv"),
            "labels": str(output_dir / "bank_churners_labels.csv"),
            "metadata": str(output_dir / "preprocessing_metadata.json"),
        },
    }
    return metadata


def validate_processed_outputs(
    raw_df: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    excluded_columns: list[str],
) -> None:
    if len(features) != len(raw_df):
        raise ValueError("Feature row count does not match raw row count.")

    if len(labels) != len(raw_df):
        raise ValueError("Label row count does not match raw row count.")

    if features.isna().any().any():
        raise ValueError("Processed features contain missing values.")

    non_numeric_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Processed features contain non-numeric columns: {non_numeric_columns}")

    forbidden_columns = set(excluded_columns).intersection(features.columns)
    if forbidden_columns:
        raise ValueError(
            f"Excluded columns still appear in the feature matrix: {sorted(forbidden_columns)}"
        )

    expected_distribution = {"Existing Customer": 8500, "Attrited Customer": 1627}
    actual_distribution = value_counts_as_ints(labels[TARGET_COLUMN])
    if actual_distribution != expected_distribution:
        raise ValueError(
            "Unexpected Attrition_Flag distribution. "
            f"Expected {expected_distribution}, got {actual_distribution}"
        )


def write_outputs(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    features.to_csv(output_dir / "bank_churners_features.csv", index=False)
    labels.to_csv(output_dir / "bank_churners_labels.csv", index=False)

    with (output_dir / "preprocessing_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    raw_df = read_dataset(input_path)
    validate_required_columns(raw_df)

    labels = build_labels(raw_df)
    features, feature_metadata = build_features(raw_df)
    metadata = build_metadata(
        raw_df=raw_df,
        features=features,
        labels=labels,
        feature_metadata=feature_metadata,
        input_path=input_path,
        output_dir=output_dir,
    )

    validate_processed_outputs(
        raw_df=raw_df,
        features=features,
        labels=labels,
        excluded_columns=feature_metadata["excluded_columns"],
    )
    write_outputs(features, labels, metadata, output_dir)

    print("Preprocessing completed successfully.")
    print(f"Rows: {len(raw_df)}")
    print(f"Feature columns: {features.shape[1]}")
    print(f"Feature file: {output_dir / 'bank_churners_features.csv'}")
    print(f"Label file: {output_dir / 'bank_churners_labels.csv'}")
    print(f"Metadata file: {output_dir / 'preprocessing_metadata.json'}")


if __name__ == "__main__":
    main()
