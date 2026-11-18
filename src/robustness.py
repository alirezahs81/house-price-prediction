"""Assess model stability and retain row-level errors across repeated holdouts.

All model settings are fixed in advance. Each split fits preprocessing only on its
training partition. Price bands are descriptive reporting groups; they do not
affect training, model selection, or predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET, TRAIN_FILE
from src.data_validation import validate_training_data
from src.evaluate import regression_metrics
from src.modeling import make_pipeline
from src.preprocessing import prepare_features, transform_absence_categories

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("reports/robustness")
METRICS = ("MAE", "RMSE", "R2", "RMSLE")
DEFAULT_WEAK_SEEDS = (47, 56, 60, 62)

# Fixed before the robustness study; this script does not tune parameters.
MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 2,
    "subsample": 0.8,
    "loss": "huber",
    "alpha": 0.95,
    "random_state": RANDOM_STATE,
}


def validate_seed_count(seed_count: int) -> None:
    """Require at least two splits for a variability estimate."""
    if seed_count < 2:
        raise ValueError("seed_count must be at least 2.")


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize holdout metrics across seeds using descriptive statistics."""
    missing = set(METRICS).difference(results.columns)
    if missing:
        raise ValueError(f"Missing required metric columns: {sorted(missing)}")
    if len(results) < 2:
        raise ValueError("At least two seed results are required.")

    rows: list[dict[str, float | str]] = []
    for metric in METRICS:
        values = results[metric].astype(float)
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(values.median()),
                "p10": float(values.quantile(0.10)),
                "p90": float(values.quantile(0.90)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def make_price_band_edges(target: Any) -> np.ndarray:
    """Create shared target-quantile boundaries for descriptive error groups."""
    values = np.asarray(target, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Target values must be non-empty and finite.")
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 6))
    unique_edges = np.unique(quantiles)
    if unique_edges.size < 3:
        raise ValueError("At least two distinct price bands are required.")
    return np.concatenate(([-np.inf], unique_edges[1:-1], [np.inf]))


def build_prediction_diagnostics(
    seed: int,
    sample_ids: Any,
    actual: Any,
    predicted: Any,
    price_band_edges: Any,
) -> pd.DataFrame:
    """Build aligned per-house predictions and errors for one holdout seed."""
    ids = np.asarray(sample_ids).reshape(-1)
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    edges = np.asarray(price_band_edges, dtype=float).reshape(-1)
    if not (len(ids) == len(y_true) == len(y_pred)):
        raise ValueError("IDs, actual values, and predictions must have equal lengths.")
    if len(y_true) == 0 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Actual values and predictions must be non-empty and finite.")
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("Actual values and predictions must be non-negative.")
    if len(edges) < 3 or not np.all(np.diff(edges) > 0):
        raise ValueError("Price-band edges must be strictly increasing.")

    labels = [f"Q{index}" for index in range(1, len(edges))]
    bands = pd.cut(
        y_true,
        bins=edges,
        labels=labels,
        include_lowest=True,
    )
    residuals = y_true - y_pred
    return pd.DataFrame(
        {
            "seed": seed,
            "sample_id": ids,
            "actual": y_true,
            "predicted": y_pred,
            "residual": residuals,
            "absolute_error": np.abs(residuals),
            "absolute_percentage_error": np.abs(residuals) / np.maximum(y_true, 1.0),
            "price_band": bands.astype(str),
        }
    )


def summarize_errors_by_seed_and_band(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize error magnitude and bias for each seed and actual-price band."""
    required = {
        "seed",
        "price_band",
        "actual",
        "predicted",
        "residual",
        "absolute_error",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required diagnostic columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Prediction diagnostics cannot be empty.")

    grouped = predictions.groupby(["seed", "price_band"], observed=True)
    summary = grouped.agg(
        rows=("actual", "size"),
        mean_actual_price=("actual", "mean"),
        mean_predicted_price=("predicted", "mean"),
        mean_residual=("residual", "mean"),
        MAE=("absolute_error", "mean"),
    ).reset_index()
    rmse = grouped.apply(
        lambda group: float(np.sqrt(np.mean(np.square(group["residual"])))),
        include_groups=False,
    )
    summary["RMSE"] = [
        float(rmse.loc[(row.seed, row.price_band)]) for row in summary.itertuples()
    ]
    return summary


def run_robustness(
    seed_count: int = 30,
    start_seed: int = RANDOM_STATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the fixed model on repeated splits and save aggregate and row-level reports."""
    validate_seed_count(seed_count)
    frame = pd.read_csv(TRAIN_FILE)
    validate_training_data(frame)

    features = transform_absence_categories(prepare_features(frame))
    if TARGET in features.columns or "Id" in features.columns:
        raise ValueError("Target or identifier leakage detected in model features.")

    target = frame[TARGET].astype(float)
    target_log = np.log1p(target)
    price_band_edges = make_price_band_edges(target.to_numpy())
    metric_rows: list[dict[str, float | int]] = []
    diagnostic_frames: list[pd.DataFrame] = []

    for seed in range(start_seed, start_seed + seed_count):
        X_train, X_valid, y_train, y_valid_log = train_test_split(
            features,
            target_log,
            test_size=0.2,
            random_state=seed,
        )
        estimator = GradientBoostingRegressor(**MODEL_PARAMS)
        pipeline = make_pipeline(estimator, X_train)
        pipeline.fit(X_train, y_train)

        predictions = np.maximum(np.expm1(pipeline.predict(X_valid)), 1.0)
        actual = np.expm1(y_valid_log).to_numpy(dtype=float)
        metrics = regression_metrics(actual, predictions)
        metric_rows.append(
            {
                "seed": seed,
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_valid)),
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
        )
        diagnostic_frames.append(
            build_prediction_diagnostics(
                seed=seed,
                sample_ids=frame.loc[X_valid.index, "Id"].to_numpy(),
                actual=actual,
                predicted=predictions,
                price_band_edges=price_band_edges,
            )
        )
        LOGGER.info(
            "Seed %d/%d complete: RMSE=%.2f, R2=%.4f",
            seed - start_seed + 1,
            seed_count,
            metrics["RMSE"],
            metrics["R2"],
        )

    per_seed = pd.DataFrame(metric_rows)
    summary = summarize_results(per_seed)
    predictions_by_row = pd.concat(diagnostic_frames, ignore_index=True)
    error_by_band = summarize_errors_by_seed_and_band(predictions_by_row)

    weak_seed_rows = per_seed[per_seed["seed"].isin(DEFAULT_WEAK_SEEDS)].copy()
    if not weak_seed_rows.empty:
        weak_seed_rows["seed_group"] = "previously_identified_weak_seed"
    other_seed_rows = per_seed[~per_seed["seed"].isin(DEFAULT_WEAK_SEEDS)].copy()
    if not other_seed_rows.empty:
        other_seed_rows["seed_group"] = "other_seed"
    seed_group_summary = pd.concat(
        [weak_seed_rows, other_seed_rows], ignore_index=True
    )
    if not seed_group_summary.empty:
        seed_group_summary = (
            seed_group_summary.groupby("seed_group", as_index=False)[list(METRICS)]
            .mean()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "robustness_by_seed.csv", index=False)
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    predictions_by_row.to_csv(
        output_dir / "robustness_predictions.csv", index=False
    )
    error_by_band.to_csv(
        output_dir / "robustness_error_by_seed_price_band.csv", index=False
    )
    seed_group_summary.to_csv(
        output_dir / "robustness_weak_seed_comparison.csv", index=False
    )

    payload = {
        "experiment": "repeated_random_holdout_robustness_with_error_analysis",
        "model": "GradientBoostingRegressor",
        "model_params": MODEL_PARAMS,
        "seed_count": seed_count,
        "seeds": list(range(start_seed, start_seed + seed_count)),
        "previously_identified_weak_seeds": list(DEFAULT_WEAK_SEEDS),
        "split": {
            "test_size": 0.2,
            "train_rows_per_split": int(per_seed["train_rows"].iloc[0]),
            "validation_rows_per_split": int(per_seed["validation_rows"].iloc[0]),
        },
        "methodology": (
            "Each seed creates a new 80/20 random split. Preprocessing is fitted "
            "inside the sklearn pipeline on that seed's training partition only. "
            "The model configuration is fixed in advance; validation results are "
            "descriptive and are not used for hyperparameter selection. Row-level "
            "predictions retain the original Id for cross-seed error inspection."
        ),
        "price_band_method": (
            "Shared quintile boundaries are calculated once from the training "
            "target distribution for descriptive reporting only. Price bands do "
            "not affect model fitting, selection, or predictions."
        ),
        "limitations": (
            "Repeated random holdouts reuse observations across splits and are "
            "not independent external test sets. The reported standard deviation "
            "describes split sensitivity on this dataset, not a confidence "
            "interval for real-world deployment performance."
        ),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "robustness_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(per_seed["seed"], per_seed["RMSE"], marker="o")
    axes[0].set(
        xlabel="Random seed",
        ylabel="RMSE (price units)",
        title="Holdout RMSE by seed",
    )
    axes[0].grid(alpha=0.2)
    axes[1].plot(per_seed["seed"], per_seed["R2"], marker="o")
    axes[1].set(
        xlabel="Random seed",
        ylabel="R2",
        title="Holdout R2 by seed",
    )
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_metrics.png", dpi=160)
    plt.close(fig)

    return per_seed, summary


def main() -> None:
    """Parse CLI options and run the robustness experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    per_seed, summary = run_robustness(
        seed_count=args.seeds,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
    )
    print(f"Completed {len(per_seed)} holdout runs.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
