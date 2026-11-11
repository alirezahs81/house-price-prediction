"""Evaluation metrics and diagnostic plots."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, METRICS_DIR


def regression_metrics(y_true: Any, pred: Any) -> dict[str, float]:
    """Calculate common regression metrics on the original price scale."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and pred must have identical shapes.")
    if actual.size == 0:
        raise ValueError("Metrics cannot be calculated for empty arrays.")
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    rmsle = float(np.sqrt(np.mean(
        (np.log1p(np.maximum(predicted, 0))
         - np.log1p(np.maximum(actual, 0))) ** 2
    )))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": float(mean_squared_error(actual, predicted)),
        "RMSE": rmse,
        "R2": float(r2_score(actual, predicted)),
        "RMSLE": rmsle,
    }


def save_diagnostics(
    y_true: Any, pred: Any, name: str = "best_model"
) -> dict[str, float]:
    """Save validation plots and metrics, creating output directories first."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(pred, dtype=float)
    residuals = actual - predicted

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, predicted, alpha=0.65)
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
