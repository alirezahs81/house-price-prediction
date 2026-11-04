"""Evaluation metrics and diagnostic plots."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import METRICS_DIR, FIGURES_DIR

def regression_metrics(y_true, pred):
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    return {"MAE": float(mean_absolute_error(y_true, pred)), "MSE": float(mean_squared_error(y_true, pred)),
            "RMSE": rmse, "R2": float(r2_score(y_true, pred)),
            "RMSLE": float(np.sqrt(np.mean((np.log1p(np.maximum(pred, 0)) - np.log1p(np.maximum(y_true, 0)))**2)))}

def save_diagnostics(y_true, pred, name="best_model"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True); METRICS_DIR.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - np.asarray(pred)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(y_true, pred, alpha=.65); ax.set(xlabel="Actual SalePrice", ylabel="Predicted SalePrice", title="Actual vs Predicted"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES_DIR/"actual_vs_predicted.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.hist(residuals, bins=35, edgecolor="white"); ax.set(xlabel="Residual (actual - predicted)", ylabel="Count", title="Residual Distribution"); fig.tight_layout(); fig.savefig(FIGURES_DIR/"residual_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(pred, residuals, alpha=.65); ax.axhline(0, linestyle="--"); ax.set(xlabel="Predicted SalePrice", ylabel="Residual", title="Residuals vs Predicted"); fig.tight_layout(); fig.savefig(FIGURES_DIR/"residuals_vs_predicted.png", dpi=150); plt.close(fig)
    metrics=regression_metrics(y_true,pred)
    (METRICS_DIR/f"{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
