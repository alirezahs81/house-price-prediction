"""Compare fixed tree-ensemble candidates on identical repeated holdout splits.

Every model sees the same train/validation indices for each seed. Preprocessing is
fitted only on that split's training rows. Results are descriptive and are not used
for automatic hyperparameter tuning or production-model replacement.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET, TRAIN_FILE
from src.data_validation import validate_training_data
from src.evaluate import regression_metrics
from src.modeling import make_pipeline
from src.preprocessing import prepare_features, transform_absence_categories

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("reports/model_benchmark")
METRICS = ("MAE", "RMSE", "R2", "RMSLE")


def validate_seed_count(seed_count: int) -> None:
    """Require at least two paired holdout splits."""
    if seed_count < 2:
        raise ValueError("seed_count must be at least 2.")


def build_benchmark_models(seed: int) -> dict[str, Any]:
    """Return fixed candidate configurations with split-specific random states."""
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=2,
            subsample=0.8,
            loss="huber",
            alpha=0.95,
            random_state=seed,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive mean, standard deviation, and quantiles per model."""
    required = {"model", *METRICS}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if results.groupby("model").size().min() < 2:
        raise ValueError("Each model requires at least two seed results.")

    rows: list[dict[str, float | str]] = []
    for model, group in results.groupby("model", sort=True):
        for metric in METRICS:
            values = group[metric].astype(float)
            rows.append(
                {
                    "model": str(model),
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


def run_benchmark(
    seed_count: int = 30,
    start_seed: int = RANDOM_STATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all fixed candidates on the same repeated random holdout splits."""
    validate_seed_count(seed_count)
    frame = pd.read_csv(TRAIN_FILE)
    validate_training_data(frame)

    features = transform_absence_categories(prepare_features(frame))
    if TARGET in features.columns or "Id" in features.columns:
        raise ValueError("Target or identifier leakage detected in model features.")
    target_log = np.log1p(frame[TARGET].astype(float))
    rows: list[dict[str, float | int | str]] = []

    for seed in range(start_seed, start_seed + seed_count):
        X_train, X_valid, y_train, y_valid_log = train_test_split(
            features,
            target_log,
            test_size=0.2,
            random_state=seed,
        )
        actual = np.expm1(y_valid_log).to_numpy(dtype=float)
        for model_name, estimator in build_benchmark_models(seed).items():
            pipeline = make_pipeline(estimator, X_train)
            pipeline.fit(X_train, y_train)
            predicted = np.maximum(np.expm1(pipeline.predict(X_valid)), 1.0)
            metrics = regression_metrics(actual, predicted)
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "train_rows": int(len(X_train)),
                    "validation_rows": int(len(X_valid)),
                    **{metric: float(metrics[metric]) for metric in METRICS},
                }
            )
            LOGGER.info(
                "Seed %d, %s: RMSE=%.2f, R2=%.4f",
                seed,
                model_name,
                metrics["RMSE"],
                metrics["R2"],
            )

    per_seed = pd.DataFrame(rows)
    summary = summarize_results(per_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "model_benchmark_by_seed.csv", index=False)
    summary.to_csv(output_dir / "model_benchmark_summary.csv", index=False)

    payload = {
        "experiment": "paired_repeated_random_holdout_model_benchmark",
        "seed_count": seed_count,
        "seeds": list(range(start_seed, start_seed + seed_count)),
        "split": {
            "test_size": 0.2,
            "train_rows_per_split": int(per_seed["train_rows"].iloc[0]),
            "validation_rows_per_split": int(per_seed["validation_rows"].iloc[0]),
            "identical_indices_for_all_models_within_each_seed": True,
        },
        "models": list(build_benchmark_models(start_seed)),
        "methodology": (
            "All candidate models use identical train/validation indices for each "
            "seed. Each model's preprocessing is fitted only on that split's "
            "training rows. Model configurations are fixed in advance; validation "
            "results are descriptive and are not used for hyperparameter tuning."
        ),
        "limitations": (
            "Repeated random holdouts reuse observations and are not independent "
            "external test sets. These results measure split sensitivity on this "
            "dataset and do not establish real-world deployment performance."
        ),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "model_benchmark_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    rmse = summary[summary["metric"] == "RMSE"].sort_values("mean")
    fig, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(
        figsize=(9, 5)
    )
    ax.bar(rmse["model"], rmse["mean"], yerr=rmse["std"], capsize=4)
    ax.set(
        xlabel="Model",
        ylabel="Mean RMSE (price units)",
        title=f"Model comparison across {seed_count} identical holdout seeds",
    )
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "model_benchmark_rmse.png", dpi=160)
    __import__("matplotlib.pyplot", fromlist=["close"]).close(fig)

    return per_seed, summary


def main() -> None:
    """Parse command-line arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    per_seed, summary = run_benchmark(
        seed_count=args.seeds,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
    )
    print(f"Completed {len(per_seed)} model/split evaluations.")
    print(
        summary[summary["metric"].isin(["MAE", "RMSE", "R2", "RMSLE"])]
        .to_string(index=False)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
