"""Compare unweighted and price-weighted models on identical repeated holdouts.

Both variants use the same fixed GradientBoosting configuration and the same
30 train/validation splits. Sample weights and high-value thresholds are derived
from each split's training partition only. This is an analysis experiment; it
does not modify the production training pipeline or saved model.
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
from src.robustness import MODEL_PARAMS

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("reports/weighted_robustness")


def validate_seed_count(seed_count: int) -> None:
    """Require at least two paired splits for a variability estimate."""
    if seed_count < 2:
        raise ValueError("seed_count must be at least 2.")


def make_price_weights(
    training_prices: pd.Series,
    lower: float = 0.75,
    upper: float = 1.75,
) -> np.ndarray:
    """Create bounded square-root price weights using training rows only."""
    prices = pd.to_numeric(training_prices, errors="coerce").to_numpy(dtype=float)
    if prices.size == 0 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Training prices must be finite, positive, and non-empty.")
    if lower <= 0 or upper < lower:
        raise ValueError("Weight bounds must satisfy 0 < lower <= upper.")

    median_price = float(np.median(prices))
    weights = np.sqrt(prices / median_price)
    weights = np.clip(weights, lower, upper)
    weights = weights / weights.mean()
    return weights.astype(float)


def _fit_and_score(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train_log: pd.Series,
    y_valid_actual: pd.Series,
    high_value_threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Fit a fixed model and score overall and training-defined high-value rows."""
    estimator = GradientBoostingRegressor(**MODEL_PARAMS)
    pipeline = make_pipeline(estimator, X_train)
    if sample_weight is None:
        pipeline.fit(X_train, y_train_log)
    else:
        if len(sample_weight) != len(X_train):
            raise ValueError("sample_weight must match the training row count.")
        pipeline.fit(
            X_train,
            y_train_log,
            model__sample_weight=sample_weight,
        )

    actual = y_valid_actual.to_numpy(dtype=float)
    predicted = np.maximum(np.expm1(pipeline.predict(X_valid)), 1.0)
    high_mask = actual >= high_value_threshold
    if not high_mask.any():
        raise ValueError(
            "No validation rows met the training-derived high-value threshold."
        )

    overall = regression_metrics(actual, predicted)
    high_value = regression_metrics(actual[high_mask], predicted[high_mask])
    return {
        **{f"overall_{key}": float(value) for key, value in overall.items()},
        **{f"high_value_{key}": float(value) for key, value in high_value.items()},
        "high_value_rows": int(high_mask.sum()),
    }


def summarize_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize model metrics and paired weighted-minus-baseline differences."""
    metric_names = ("MAE", "RMSE", "R2", "RMSLE")
    rows: list[dict[str, Any]] = []
    for segment in ("overall", "high_value"):
        for metric in metric_names:
            baseline = results[f"baseline_{segment}_{metric}"].astype(float)
            weighted = results[f"weighted_{segment}_{metric}"].astype(float)
            delta = weighted - baseline
            rows.extend(
                [
                    {
                        "segment": segment,
                        "metric": metric,
                        "result": "baseline",
                        "mean": float(baseline.mean()),
                        "std": float(baseline.std(ddof=1)),
                        "median": float(baseline.median()),
                        "p10": float(baseline.quantile(0.10)),
                        "p90": float(baseline.quantile(0.90)),
                    },
                    {
                        "segment": segment,
                        "metric": metric,
                        "result": "price_weighted",
                        "mean": float(weighted.mean()),
                        "std": float(weighted.std(ddof=1)),
                        "median": float(weighted.median()),
                        "p10": float(weighted.quantile(0.10)),
                        "p90": float(weighted.quantile(0.90)),
                    },
                    {
                        "segment": segment,
                        "metric": metric,
                        "result": "paired_weighted_minus_baseline",
                        "mean": float(delta.mean()),
                        "std": float(delta.std(ddof=1)),
                        "median": float(delta.median()),
                        "p10": float(delta.quantile(0.10)),
                        "p90": float(delta.quantile(0.90)),
                        "weighted_wins": int((delta < 0).sum()),
                        "baseline_wins": int((delta > 0).sum()),
                        "ties": int((delta == 0).sum()),
                    },
                ]
            )
    return pd.DataFrame(rows)


def run_weighted_robustness(
    seed_count: int = 30,
    start_seed: int = RANDOM_STATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired 80/20 holdouts for baseline and price-weighted variants."""
    validate_seed_count(seed_count)
    frame = pd.read_csv(TRAIN_FILE)
    validate_training_data(frame)

    features = transform_absence_categories(prepare_features(frame))
    if TARGET in features.columns or "Id" in features.columns:
        raise ValueError("Target or identifier leakage detected in model features.")

    target_log = np.log1p(frame[TARGET].astype(float))
    target_actual = frame[TARGET].astype(float)
    rows: list[dict[str, float | int]] = []

    for seed in range(start_seed, start_seed + seed_count):
        X_train, X_valid, y_train_log, y_valid_log = train_test_split(
            features,
            target_log,
            test_size=0.2,
            random_state=seed,
        )
        y_train_actual = target_actual.loc[X_train.index]
        y_valid_actual = np.expm1(y_valid_log)
        threshold = float(y_train_actual.quantile(0.80))
        weights = make_price_weights(y_train_actual)

        baseline = _fit_and_score(
            X_train,
            X_valid,
            y_train_log,
            y_valid_actual,
            threshold,
        )
        weighted = _fit_and_score(
            X_train,
            X_valid,
            y_train_log,
            y_valid_actual,
            threshold,
            sample_weight=weights,
        )

        row: dict[str, float | int] = {
            "seed": seed,
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
            "high_value_threshold": threshold,
            "high_value_rows": int(baseline["high_value_rows"]),
        }
        for metric_key, value in baseline.items():
            if metric_key != "high_value_rows":
                row[f"baseline_{metric_key}"] = value
        for metric_key, value in weighted.items():
            if metric_key != "high_value_rows":
                row[f"weighted_{metric_key}"] = value
        for segment in ("overall", "high_value"):
            for metric in ("MAE", "RMSE", "R2", "RMSLE"):
                row[f"delta_{segment}_{metric}"] = (
                    float(weighted[f"{segment}_{metric}"])
                    - float(baseline[f"{segment}_{metric}"])
                )
        rows.append(row)
        LOGGER.info(
            "Seed %d/%d: baseline RMSE=%.2f, weighted RMSE=%.2f; "
            "high-value baseline=%.2f, weighted=%.2f",
            seed - start_seed + 1,
            seed_count,
            baseline["overall_RMSE"],
            weighted["overall_RMSE"],
            baseline["high_value_RMSE"],
            weighted["high_value_RMSE"],
        )

    per_seed = pd.DataFrame(rows)
    summary = summarize_comparison(per_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "weighted_robustness_by_seed.csv", index=False)
    summary.to_csv(output_dir / "weighted_robustness_summary.csv", index=False)

    payload = {
        "experiment": "paired_30_seed_baseline_vs_price_weighted",
        "model": "GradientBoostingRegressor",
        "model_params": MODEL_PARAMS,
        "seed_count": seed_count,
        "seeds": list(range(start_seed, start_seed + seed_count)),
        "split": {
            "test_size": 0.2,
            "train_rows_per_split": int(per_seed["train_rows"].iloc[0]),
            "validation_rows_per_split": int(per_seed["validation_rows"].iloc[0]),
        },
        "high_value_definition": (
            "Validation observations at or above the 80th percentile of that "
            "seed's training-partition SalePrice. Threshold is recalculated "
            "from training rows only for each seed."
        ),
        "weighting": (
            "sqrt(training SalePrice / training median), clipped to [0.75, "
            "1.75], then normalized to mean 1. Weights are recalculated "
            "independently inside each training split."
        ),
        "paired_design": (
            "Baseline and price-weighted variants use exactly the same "
            "train/validation rows for each seed. Validation rows are never "
            "used to calculate sample weights or high-value thresholds."
        ),
        "limitations": (
            "Repeated holdouts reuse observations and are not independent "
            "external test sets. Results describe split sensitivity on this "
            "dataset, not a deployment confidence interval. R2 on the smaller "
            "high-value subset may be variable."
        ),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "weighted_robustness_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    for segment, filename, title in (
        ("overall", "overall_rmse_by_seed.png", "Overall RMSE by seed"),
        ("high_value", "high_value_rmse_by_seed.png", "High-value RMSE by seed"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            per_seed["seed"],
            per_seed[f"baseline_{segment}_RMSE"],
            marker="o",
            label="Baseline",
        )
        ax.plot(
            per_seed["seed"],
            per_seed[f"weighted_{segment}_RMSE"],
            marker="o",
            label="Price-weighted",
        )
        ax.set(xlabel="Random seed", ylabel="RMSE (price units)", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    return per_seed, summary


def main() -> None:
    """Parse CLI options and run the paired comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    per_seed, summary = run_weighted_robustness(
        seed_count=args.seeds,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
    )
    print(f"Completed {len(per_seed)} paired holdout runs.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
