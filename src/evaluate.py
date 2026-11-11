"""Evaluation metrics and diagnostic reports for regression models."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, METRICS_DIR, REPORTS_DIR


def regression_metrics(y_true: Any, pred: Any) -> dict[str, float]:
    """Calculate common regression metrics on the original price scale."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and pred must have identical shapes.")
    if actual.size == 0:
        raise ValueError("Metrics cannot be calculated for empty arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("Sale prices must be non-negative.")

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    rmsle = float(
        np.sqrt(
            np.mean(
                (
                    np.log1p(predicted)
                    - np.log1p(actual)
                )
                ** 2
            )
        )
    )
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": float(mean_squared_error(actual, predicted)),
        "RMSE": rmse,
        "R2": float(r2_score(actual, predicted)),
        "RMSLE": rmsle,
    }


def _save_error_tables(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_ids: Any = None,
) -> None:
    """Persist row-level errors, price-band metrics, and largest misses."""
    residuals = actual - predicted
    rows = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "residual": residuals,
            "absolute_error": np.abs(residuals),
            "absolute_percentage_error": (
                np.abs(residuals) / np.maximum(actual, 1.0)
            ),
            "log_error": np.log1p(np.maximum(predicted, 0))
            - np.log1p(np.maximum(actual, 0)),
        }
    )
    if sample_ids is not None:
        ids = np.asarray(sample_ids).reshape(-1)
        if len(ids) != len(rows):
            raise ValueError("sample_ids must match the number of predictions.")
        rows.insert(0, "sample_id", ids)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(REPORTS_DIR / "validation_residuals.csv", index=False)
    rows.nlargest(20, "absolute_error").to_csv(
        REPORTS_DIR / "worst_validation_errors.csv", index=False
    )

    band_source = pd.DataFrame(
        {"actual": actual, "predicted": predicted, "absolute_error": np.abs(residuals)}
    )
    band_source["actual_price_band"] = pd.qcut(
        band_source["actual"], q=5, duplicates="drop"
    )
    band_rows = []
    for band, group in band_source.groupby(
        "actual_price_band", observed=True
    ):
        band_rows.append(
            {
                "actual_price_band": str(band),
                "rows": int(len(group)),
                "mean_actual_price": float(group["actual"].mean()),
                "mean_predicted_price": float(group["predicted"].mean()),
                "MAE": float(group["absolute_error"].mean()),
                "RMSE": float(
                    np.sqrt(
                        np.mean(
                            (group["actual"] - group["predicted"]) ** 2
                        )
                    )
                ),
            }
        )
    pd.DataFrame(band_rows).to_csv(
        REPORTS_DIR / "validation_error_by_price_band.csv", index=False
    )


def save_diagnostics(
    y_true: Any,
    pred: Any,
    name: str = "best_model",
    sample_ids: Any = None,
) -> dict[str, float]:
    """Save validation plots, metrics, and row-level error diagnostics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted arrays must be non-empty and aligned.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Diagnostics require finite actual and predicted values.")

    residuals = actual - predicted
    _save_error_tables(actual, predicted, sample_ids)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, predicted, alpha=0.65)
    limits = [
        float(min(actual.min(), predicted.min())),
        float(max(actual.max(), predicted.max())),
    ]
    ax.plot(limits, limits, linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="Actual SalePrice", ylabel="Predicted SalePrice",
           title="Actual vs Predicted")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=35, edgecolor="white")
    ax.set(xlabel="Residual (actual - predicted)", ylabel="Count",
           title="Residual Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residual_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predicted, residuals, alpha=0.65)
    ax.axhline(0, linestyle="--")
    ax.set(xlabel="Predicted SalePrice", ylabel="Residual",
           title="Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residuals_vs_predicted.png", dpi=150)
    plt.close(fig)

    metrics = regression_metrics(actual, predicted)
    metrics_path: Path = METRICS_DIR / f"{name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
