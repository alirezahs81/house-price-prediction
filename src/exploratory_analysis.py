"""Exploratory analysis and reproducible figure generation."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from src.config import TRAIN_FILE, REPORTS_DIR, FIGURES_DIR

def run_eda(path=TRAIN_FILE):
    df = pd.read_csv(path); FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    target = df["SalePrice"]
    missing = df.isna().mean().sort_values(ascending=False)
    missing[missing > 0].head(30).sort_values().plot(kind="barh", figsize=(9, 8), title="Top Missing-Value Rates")
    plt.xlabel("Fraction missing"); plt.tight_layout(); plt.savefig(FIGURES_DIR/"missing_values.png", dpi=150); plt.close()
    target.hist(bins=40, figsize=(8,5)); plt.title("SalePrice Distribution"); plt.xlabel("SalePrice"); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(FIGURES_DIR/"saleprice_distribution.png", dpi=150); plt.close()
    numeric = df.select_dtypes(include="number")
    selected_numeric = [c for c in numeric.columns if c not in ("Id", "SalePrice")][:16]
    if selected_numeric:
        axes = numeric[selected_numeric].hist(figsize=(14, 10), bins=25)
        plt.suptitle("Numerical Feature Distributions")
        plt.tight_layout(); plt.savefig(FIGURES_DIR/"numeric_distributions.png", dpi=150); plt.close()
    categorical = df.select_dtypes(exclude="number")
    top_cats = categorical.nunique().sort_values(ascending=False).head(12).index.tolist()
    if top_cats:
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        for ax, col in zip(axes.flat, top_cats):
            categorical[col].fillna("Missing").value_counts().head(10).plot(kind="bar", ax=ax)
            ax.set_title(col); ax.tick_params(axis="x", labelrotation=45)
        for ax in axes.flat[len(top_cats):]: ax.set_visible(False)
        fig.tight_layout(); fig.savefig(FIGURES_DIR/"categorical_distributions.png", dpi=150); plt.close(fig)
    outlier_rows=[]
    for col in numeric.columns:
        q1, q3 = numeric[col].quantile([0.25, 0.75])
        iqr = q3-q1
        count = int(((numeric[col] < q1-1.5*iqr) | (numeric[col] > q3+1.5*iqr)).sum()) if pd.notna(iqr) else 0
        outlier_rows.append({"feature": col, "outlier_count_iqr": count, "outlier_fraction_iqr": count/max(len(df),1)})
    pd.DataFrame(outlier_rows).sort_values("outlier_count_iqr", ascending=False).to_csv(REPORTS_DIR/"metrics"/"numeric_outlier_summary.csv", index=False)
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(20)
    corr.sort_values().plot(kind="barh", figsize=(9,7), title="Top Numeric Correlations with SalePrice"); plt.xlabel("Absolute Pearson correlation"); plt.tight_layout(); plt.savefig(FIGURES_DIR/"target_correlations.png", dpi=150); plt.close()
    overview = {"train_rows": int(df.shape[0]), "columns_including_target": int(df.shape[1]),
                "numeric_columns": int(len(numeric.columns)), "categorical_columns": int(df.select_dtypes(exclude="number").shape[1]),
                "duplicate_rows": int(df.duplicated().sum()), "missing_cells": int(df.isna().sum().sum()),
                "saleprice_mean": float(target.mean()), "saleprice_median": float(target.median()),
                "saleprice_min": float(target.min()), "saleprice_max": float(target.max()),
                "top_missing_columns": {str(k): int(v) for k,v in df.isna().sum().sort_values(ascending=False).head(15).items()}}
    (REPORTS_DIR/"metrics"/"data_overview.json").parent.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR/"metrics"/"data_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    return overview
