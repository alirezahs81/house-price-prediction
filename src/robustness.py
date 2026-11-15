"""Assess model stability across repeated, independent holdout splits.

The experiment reuses the fixed, previously selected GradientBoosting settings.
Each seed creates a fresh 80/20 split; preprocessing is fitted only on that
seed's training partition. No seed's validation partition is used for tuning.
The production training pipeline and saved model are never modified.
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

# Fixed before this robustness study; this script does not tune parameters.
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
    """Require at least two independent splits for a variability estimate."""
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


def run_robustness(
    seed_count: int = 30,
    start_seed: int = RANDOM_STATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the fixed model on repeated splits and save per-seed results."""
    validate_seed_count(seed_count)
    frame = pd.read_csv(TRAIN_FILE)
    validate_training_data(frame)

    features = transform_absence_categories(prepare_features(frame))
    if TARGET in features.columns or "Id" in features.columns:
        raise ValueError("Target or identifier leakage detected in model features.")

    target_log = np.log1p(frame[TARGET].astype(float))
    rows: list[dict[str, float | int]] = []

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
        rows.append(
            {
                "seed": seed,
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_valid)),
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
        )
        LOGGER.info(
            "Seed %d/%d complete: RMSE=%.2f, R2=%.4f",
            seed - start_seed + 1,
            seed_count,
            metrics["RMSE"],
            metrics["R2"],
        )

    per_seed = pd.DataFrame(rows)
    summary = summarize_results(per_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "robustness_by_seed.csv", index=False)
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)

    payload = {
        "experiment": "repeated_random_holdout_robustness",
        "model": "GradientBoostingRegressor",
        "model_params": MODEL_PARAMS,
        "seed_count": seed_count,
        "seeds": list(range(start_seed, start_seed + seed_count)),
        "split": {
            "test_size": 0.2,
            "train_rows_per_split": int(per_seed["train_rows"].iloc[0]),
            "validation_rows_per_split": int(per_seed["validation_rows"].iloc[0]),
        },
        "methodology": (
            "Each seed creates a new 80/20 random split. Preprocessing is fitted "
            "inside the sklearn pipeline on that seed's training partition only. "
            "The model configuration is fixed in advance; validation results are "
            "descriptive and are not used for hyperparameter selection."
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
    axes[0].boxplot(
        [per_seed["RMSE"], per_seed["R2"]],
        labels=["RMSE", "R2"],
    )
    axes[0].set_title("Metric variability across random splits")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(per_seed["seed"], per_seed["RMSE"], marker="o", label="RMSE")
    axes[1].set_xlabel("Random seed")
    axes[1].set_ylabel("RMSE (price units)")
    axes[1].set_title("Holdout RMSE by seed")
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
