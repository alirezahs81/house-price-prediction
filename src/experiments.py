"""Run controlled, holdout-consistent experiments for price-error analysis.

The experiment compares the established GradientBoosting configuration with:
1. removal of QualityTimesLivingArea;
2. moderate price-aware sample weighting.

The production training pipeline and saved model are not modified.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET, TRAIN_FILE
from src.data_validation import validate_training_data
from src.evaluate import regression_metrics
from src.preprocessing import (
    prepare_features,
    transform_absence_categories,
)
from src.modeling import make_pipeline

LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path("reports/experiments")

# Exact winning configuration from the successful 2026-11-15 training run.
BASELINE_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 2,
    "subsample": 0.8,
    "loss": "huber",
    "alpha": 0.95,
    "random_state": RANDOM_STATE,
}


def _fit_and_score(
    name: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train_log: pd.Series,
    y_valid_actual: pd.Series,
    high_value_threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one fixed estimator and report overall and high-value errors."""
    estimator = GradientBoostingRegressor(**BASELINE_PARAMS)
    pipeline = make_pipeline(estimator, X_train)
    if sample_weight is None:
        pipeline.fit(X_train, y_train_log)
    else:
        pipeline.fit(
            X_train,
            y_train_log,
            model__sample_weight=sample_weight,
        )

    predictions = np.maximum(
        np.expm1(pipeline.predict(X_valid)),
        1.0,
    )
    actual = y_valid_actual.to_numpy(dtype=float)
    high_mask = actual >= high_value_threshold
    overall = regression_metrics(actual, predictions)
    high_value = regression_metrics(actual[high_mask], predictions[high_mask])

    return {
        "experiment": name,
        **{f"overall_{key}": value for key, value in overall.items()},
        **{f"high_value_{key}": value for key, value in high_value.items()},
        "high_value_rows": int(high_mask.sum()),
        "high_value_threshold_from_train_only": float(high_value_threshold),
    }


def run_experiments() -> pd.DataFrame:
    """Run three controlled variants on the same fixed train/validation split."""
    frame = pd.read_csv(TRAIN_FILE)
    validate_training_data(frame)

    features = transform_absence_categories(prepare_features(frame))
    if TARGET in features.columns or "Id" in features.columns:
        raise ValueError("Target or identifier leakage detected.")

    target_log = np.log1p(frame[TARGET].astype(float))
    X_train, X_valid, y_train_log, y_valid_log = train_test_split(
        features,
        target_log,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    y_train_actual = np.expm1(y_train_log)
    y_valid_actual = np.expm1(y_valid_log)

    # Define the luxury/high-value segment from training data only.
    high_value_threshold = float(y_train_actual.quantile(0.80))

    results: list[dict[str, Any]] = []

    results.append(
        _fit_and_score(
            "baseline_original_features",
            X_train,
            X_valid,
            y_train_log,
            y_valid_actual,
            high_value_threshold,
        )
    )

    ablated_train = X_train.drop(
        columns=["QualityTimesLivingArea"],
        errors="ignore",
    )
    ablated_valid = X_valid.drop(
        columns=["QualityTimesLivingArea"],
        errors="ignore",
    )
    results.append(
        _fit_and_score(
            "ablation_without_QualityTimesLivingArea",
            ablated_train,
            ablated_valid,
            y_train_log,
            y_valid_actual,
            high_value_threshold,
        )
    )

    # Square-root weighting is intentionally moderate and bounded to avoid
    # allowing a few luxury observations to dominate the fit.
    median_price = float(y_train_actual.median())
    raw_weights = np.sqrt(
        y_train_actual.to_numpy(dtype=float) / max(median_price, 1.0)
    )
    weights = np.clip(raw_weights, 0.75, 1.75)
    weights = weights / weights.mean()

    results.append(
        _fit_and_score(
            "price_weighted_sqrt_clipped_0.75_1.75",
            X_train,
            X_valid,
            y_train_log,
            y_valid_actual,
            high_value_threshold,
            sample_weight=weights,
        )
    )

    result_frame = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(OUTPUT_DIR / "controlled_experiments.csv", index=False)

    payload = {
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
        },
        "baseline_params": BASELINE_PARAMS,
        "high_value_definition": (
            "Validation houses at or above the 80th percentile of training "
            "partition SalePrice; threshold is derived from training only."
        ),
        "weighting": (
            "sqrt(training SalePrice / training median), clipped to [0.75, "
            "1.75], then normalized to mean 1."
        ),
        "holdout_caveat": (
            "This fixed validation partition is used to compare the three "
            "predeclared experiments. Treat it as an experiment comparison "
            "set after this run; do not repeatedly tune against it."
        ),
        "results": results,
    }
    (OUTPUT_DIR / "controlled_experiments.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    LOGGER.info(
        "Controlled experiments complete. Results: %s",
        OUTPUT_DIR / "controlled_experiments.csv",
    )
    return result_frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run_experiments().to_string(index=False))
