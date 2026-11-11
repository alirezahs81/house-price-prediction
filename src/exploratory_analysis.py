"""Exploratory analysis and reproducible figure generation."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, TRAIN_FILE


def run_eda(path: str | Path = TRAIN_FILE) -> dict[str, Any]:
    """Generate exploratory figures and a compact dataset overview."""
    df = pd.read_csv(path)
    if "SalePrice" not in df.columns:
        raise ValueError("EDA input must contain the SalePrice column.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    target = df["SalePrice"]

    missing = df.isna().mean().sort_values(ascending=False)
    missing[missing > 0].head(30).sort_values().plot(
        kind="barh", figsize=(9, 8), title="Top Missing-Value Rates"
    )
    plt.xlabel("Fraction missing")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=150)
    plt.close()

    target.hist(bins=40, figsize=(8, 5))
    plt.title("SalePrice Distribution")
    plt.xlabel("SalePrice")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "saleprice_distribution.png", dpi=150)
    plt.close()

    numeric = df.select_dtypes(include="number")
    selected = [
        col for col in numeric.columns if col not in ("Id", "SalePrice")
    ][:16]
    if selected:
        numeric[selected].hist(figsize=(14, 10), bins=25)
        plt.suptitle("Numerical Feature Distributions")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=150)
        plt.close()

    categorical = df.select_dtypes(exclude="number")
    top_categories = (
        categorical.nunique().sort_values(ascending=False).head(12).index
    )
    if len(top_categories):
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        for axis, column in zip(axes.flat, top_categories):
            categorical[column].fillna("Missing").value_counts().head(10).plot(
                kind="bar", ax=axis
            )
            axis.set_title(column)
            axis.tick_params(axis="x", labelrotation=45)
        for axis in axes.flat[len(top_categories):]:
            axis.set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "categorical_distributions.png", dpi=150)
        plt.close(fig)

    outlier_rows = []
    for column in numeric.columns:
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        count = 0
        if pd.notna(iqr):
            count = int((
                (numeric[column] < q1 - 1.5 * iqr)
                | (numeric[column] > q3 + 1.5 * iqr)
            ).sum())
        outlier_rows.append({
            "feature": column,
            "outlier_count_iqr": count,
            "outlier_fraction_iqr": count / max(len(df), 1),
        })
    pd.DataFrame(outlier_rows).sort_values(
        "outlier_count_iqr", ascending=False
    ).to_csv(METRICS_DIR / "numeric_outlier_summary.csv", index=False)

    correlations = (
        numeric.corr(numeric_only=True)["SalePrice"]
        .drop("SalePrice")
        .abs()
        .sort_values(ascending=False)
        .head(20)
    )
    correlations.sort_values().plot(
        kind="barh", figsize=(9, 7),
        title="Top Numeric Correlations with SalePrice",
    )
    plt.xlabel("Absolute Pearson correlation")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_correlations.png", dpi=150)
    plt.close()

    overview: dict[str, Any] = {
        "train_rows": int(df.shape[0]),
        "columns_including_target": int(df.shape[1]),
        "numeric_columns": int(len(numeric.columns)),
        "categorical_columns": int(
            len(df.select_dtypes(exclude="number").columns)
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "saleprice_mean": float(target.mean()),
        "saleprice_median": float(target.median()),
        "saleprice_min": float(target.min()),
        "saleprice_max": float(target.max()),
        "top_missing_columns": {
            str(key): int(value)
            for key, value in df.isna().sum().sort_values(
                ascending=False
            ).head(15).items()
        },
    }
    (METRICS_DIR / "data_overview.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )
    return overview
