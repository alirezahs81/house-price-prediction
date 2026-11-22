"""Paired-seed experiment for price weighting on the selected HistGradientBoosting model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import REPORTS_DIR, TARGET
from src.data_loader import load_train
from src.modeling import make_pipeline
from src.preprocessing import prepare_features, transform_absence_categories

SEEDS = list(range(42, 72))
HIGH_PRICE_QUANTILE = 0.80
LOW_WEIGHT = 0.75
HIGH_WEIGHT = 1.75
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261122

# Exact selected HistGradientBoosting configuration from PR #10's final
# validation report. Both arms use the same configuration and split seeds.
MODEL_PARAMS: dict[str, Any] = {
    "max_leaf_nodes": 7,
    "max_iter": 400,
    "learning_rate": 0.04,
    "l2_regularization": 0.0,
}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate metrics on original dollar prices."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "RMSLE": float(
            np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))
        ),
    }


def _paired_summary(
    frame: pd.DataFrame, metric_columns: list[str]
) -> dict[str, Any]:
    """Summarize paired weighted-minus-baseline deltas with seed bootstrap CIs."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result: dict[str, Any] = {}
    for metric in metric_columns:
        base_col = f"baseline_{metric}"
        weighted_col = f"weighted_{metric}"
        delta = frame[weighted_col].to_numpy() - frame[base_col].to_numpy()
        bootstrap = rng.choice(
            delta, size=(BOOTSTRAP_REPLICATES, len(delta)), replace=True
        ).mean(axis=1)
        result[metric] = {
            "baseline_mean": float(frame[base_col].mean()),
            "weighted_mean": float(frame[weighted_col].mean()),
            "mean_delta_weighted_minus_baseline": float(delta.mean()),
            "delta_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "delta_ci95_high": float(np.quantile(bootstrap, 0.975)),
            "weighted_wins": int(
                (delta > 0).sum() if metric.endswith("R2") else (delta < 0).sum()
            ),
            "paired_seeds": int(len(delta)),
        }
    return result


def run_experiment() -> dict[str, Any]:
    """Run 30 paired holdout splits and save machine/human-readable reports."""
    frame = load_train()
    if TARGET not in frame:
        raise ValueError(f"Training data must contain {TARGET!r}.")

    prices = frame[TARGET].astype(float).to_numpy()
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("SalePrice must contain finite, non-negative values.")

    X = transform_absence_categories(prepare_features(frame))
    log_prices = np.log1p(prices)
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        X_train, X_valid, y_train_log, y_valid_log, price_train, price_valid = (
            train_test_split(
                X,
                log_prices,
                prices,
                test_size=0.2,
                random_state=seed,
            )
        )
        cutoff = float(np.quantile(price_train, HIGH_PRICE_QUANTILE))
        train_weights = np.where(
            price_train >= cutoff, HIGH_WEIGHT, LOW_WEIGHT
        ).astype(float)
        train_weights /= train_weights.mean()

        y_train = pd.Series(y_train_log, index=X_train.index)
        y_valid = np.asarray(y_valid_log, dtype=float)
        high_valid = price_valid >= cutoff
        if high_valid.sum() < 2:
            raise ValueError(f"Seed {seed} has too few high-price validation rows.")

        for arm in ("baseline", "weighted"):
            estimator = HistGradientBoostingRegressor(
                **MODEL_PARAMS, random_state=seed
            )
            pipeline = make_pipeline(estimator, X_train)
            if arm == "weighted":
                pipeline.fit(
                    X_train,
                    y_train,
                    model__sample_weight=train_weights,
                )
            else:
                pipeline.fit(X_train, y_train)

            predicted = np.expm1(pipeline.predict(X_valid))
            full_metrics = _metrics(price_valid, predicted)
            high_metrics = _metrics(price_valid[high_valid], predicted[high_valid])
            row: dict[str, Any] = {
                "seed": seed,
                "arm": arm,
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_valid)),
                "high_price_cutoff_from_train": cutoff,
                "high_price_validation_rows": int(high_valid.sum()),
            }
            row.update({f"overall_{key}": value for key, value in full_metrics.items()})
            row.update({f"high_price_{key}": value for key, value in high_metrics.items()})
            rows.append(row)

    results = pd.DataFrame(rows)
    wide = results.pivot(index="seed", columns="arm")
    paired = pd.DataFrame({"seed": wide.index})
    for metric in [
        "overall_MAE",
        "overall_RMSE",
        "overall_R2",
        "overall_RMSLE",
        "high_price_MAE",
        "high_price_RMSE",
        "high_price_R2",
        "high_price_RMSLE",
    ]:
        paired[f"baseline_{metric}"] = wide[(metric, "baseline")].to_numpy()
        paired[f"weighted_{metric}"] = wide[(metric, "weighted")].to_numpy()

    overall_summary = _paired_summary(
        paired,
        ["overall_MAE", "overall_RMSE", "overall_R2", "overall_RMSLE"],
    )
    high_summary = _paired_summary(
        paired,
        ["high_price_MAE", "high_price_RMSE", "high_price_R2", "high_price_RMSLE"],
    )
    summary: dict[str, Any] = {
        "experiment": "paired_histgradientboosting_price_weighting",
        "seeds": SEEDS,
        "holdout_fraction": 0.2,
        "high_price_quantile_from_each_train_split": HIGH_PRICE_QUANTILE,
        "weight_bounds_before_mean_normalization": [LOW_WEIGHT, HIGH_WEIGHT],
        "model_params": MODEL_PARAMS,
        "target_transform": "log1p; predictions inverse-transformed with expm1",
        "overall": overall_summary,
        "high_price_subset": high_summary,
        "interpretation_note": (
            "Bootstrap intervals resample the 30 paired seed-level deltas. "
            "Holdout splits overlap, so intervals are descriptive and are not "
            "independent external-test confidence intervals."
        ),
    }

    output_dir = REPORTS_DIR / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "histgb_weighting_seed_results.csv", index=False)
    paired.to_csv(output_dir / "histgb_weighting_paired_results.csv", index=False)
    (output_dir / "histgb_weighting_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# HistGradientBoosting price-weighting experiment",
        "",
        "- Paired seeds: 42–71 (30 splits)",
        "- Same 80/20 holdout split for both arms per seed",
        "- Weight cutoff: training-only 80th price percentile",
        "- Model configuration: PR #10 selected HistGradientBoosting parameters",
        "- Weighted arm: 0.75 / 1.75 class weights, normalized to mean 1",
        "",
        "## Overall validation metrics",
        "",
        "| Metric | Baseline mean | Weighted mean | Delta (weighted − baseline) | 95% paired bootstrap CI | Wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, values in overall_summary.items():
        lines.append(
            f"| {metric} | {values['baseline_mean']:.6f} | "
            f"{values['weighted_mean']:.6f} | "
            f"{values['mean_delta_weighted_minus_baseline']:.6f} | "
            f"[{values['delta_ci95_low']:.6f}, {values['delta_ci95_high']:.6f}] | "
            f"{values['weighted_wins']}/30 |"
        )
    lines.extend(
        [
            "",
            "## High-price validation subset",
            "",
            "| Metric | Baseline mean | Weighted mean | Delta (weighted − baseline) | 95% paired bootstrap CI | Wins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, values in high_summary.items():
        lines.append(
            f"| {metric} | {values['baseline_mean']:.6f} | "
            f"{values['weighted_mean']:.6f} | "
            f"{values['mean_delta_weighted_minus_baseline']:.6f} | "
            f"[{values['delta_ci95_low']:.6f}, {values['delta_ci95_high']:.6f}] | "
            f"{values['weighted_wins']}/30 |"
        )
    lines.extend(["", summary["interpretation_note"], ""])
    (output_dir / "histgb_weighting_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    return summary


if __name__ == "__main__":
    report = run_experiment()
    print(json.dumps(report, indent=2, ensure_ascii=False))
