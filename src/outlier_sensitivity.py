"""Paired sensitivity experiment for two repeatedly mispredicted houses.

The baseline and sensitivity scenarios use identical train/validation splits.
The sensitivity scenario removes IDs 524 and 1299 from the training partition
only; every validation row remains present. This isolates their influence on
model fitting without making validation artificially easier.
"""

from __future__ import annotations

import argparse
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
from src.modeling import make_pipeline
from src.preprocessing import prepare_features, transform_absence_categories

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("reports/outlier_sensitivity")
CANDIDATE_IDS = (524, 1299)
METRICS = ("MAE", "RMSE", "R2", "RMSLE")
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


def training_rows_without_ids(
    train_frame: pd.DataFrame,
    excluded_ids: tuple[int, ...] = CANDIDATE_IDS,
) -> pd.DataFrame:
    """Remove selected IDs from a training partition without changing validation."""
    if "Id" not in train_frame.columns:
        raise ValueError("Training frame must contain the Id column.")
    return train_frame.loc[~train_frame["Id"].isin(excluded_ids)].copy()


def paired_metric_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Return per-seed metrics and filtered-minus-baseline paired differences."""
    required = {"seed", "scenario", *METRICS}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required result columns: {sorted(missing)}")
    baseline = results[results["scenario"] == "baseline"].set_index("seed")
    filtered = results[results["scenario"] == "exclude_candidates_from_training"].set_index("seed")
    common = baseline.index.intersection(filtered.index)
    if len(common) == 0 or len(common) != len(baseline) or len(common) != len(filtered):
        raise ValueError("Each seed must have exactly one result for both scenarios.")

    rows: list[dict[str, float | int]] = []
    for seed in common:
        row: dict[str, float | int] = {"seed": int(seed)}
        for metric in METRICS:
            row[f"baseline_{metric}"] = float(baseline.loc[seed, metric])
            row[f"filtered_{metric}"] = float(filtered.loc[seed, metric])
            row[f"delta_{metric}"] = (
                float(filtered.loc[seed, metric]) - float(baseline.loc[seed, metric])
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def run_sensitivity(
    seed_count: int = 30,
    start_seed: int = RANDOM_STATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run paired holdouts with identical validation rows in both scenarios."""
    if seed_count < 2:
        raise ValueError("seed_count must be at least 2.")

    frame = pd.read_csv(TRAIN_FILE)
    validate_training_data(frame)
    if not set(CANDIDATE_IDS).issubset(set(frame["Id"])):
        raise ValueError("One or more candidate IDs are absent from the training data.")

    features = transform_absence_categories(prepare_features(frame))
    if TARGET in features.columns or "Id" in features.columns:
        raise ValueError("Target or identifier leakage detected in model features.")
    target_log = np.log1p(frame[TARGET].astype(float))
    all_positions = np.arange(len(frame))
    rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []

    for seed in range(start_seed, start_seed + seed_count):
        train_positions, valid_positions = train_test_split(
            all_positions, test_size=0.2, random_state=seed
        )
        train_frame = frame.iloc[train_positions]
        filtered_train_frame = training_rows_without_ids(train_frame)
        filtered_positions = filtered_train_frame.index.to_numpy()
        valid_frame = frame.iloc[valid_positions]
        X_valid = features.loc[valid_frame.index]
        y_valid_log = target_log.loc[valid_frame.index]
        actual = np.expm1(y_valid_log).to_numpy(dtype=float)

        for scenario, selected_positions in (
            ("baseline", train_frame.index.to_numpy()),
            ("exclude_candidates_from_training", filtered_positions),
        ):
            X_train = features.loc[selected_positions]
            y_train = target_log.loc[selected_positions]
            pipeline = make_pipeline(
                GradientBoostingRegressor(**MODEL_PARAMS), X_train
            )
            pipeline.fit(X_train, y_train)
            predicted = np.maximum(np.expm1(pipeline.predict(X_valid)), 1.0)
            metrics = regression_metrics(actual, predicted)
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "train_rows": int(len(selected_positions)),
                    "validation_rows": int(len(valid_positions)),
                    **{metric: float(metrics[metric]) for metric in METRICS},
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "sample_id": valid_frame["Id"].to_numpy(),
                        "actual": actual,
                        "predicted": predicted,
                        "residual": actual - predicted,
                        "absolute_error": np.abs(actual - predicted),
                    }
                )
            )
        LOGGER.info("Completed paired sensitivity split for seed %d", seed)

    results = pd.DataFrame(rows)
    paired = paired_metric_comparison(results)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    candidate_errors = predictions[
        predictions["sample_id"].isin(CANDIDATE_IDS)
    ].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "outlier_sensitivity_by_seed.csv", index=False)
    paired.to_csv(output_dir / "outlier_sensitivity_paired_deltas.csv", index=False)
    predictions.to_csv(output_dir / "outlier_sensitivity_predictions.csv", index=False)
    candidate_errors.to_csv(
        output_dir / "candidate_house_prediction_errors.csv", index=False
    )

    summary = (
        results.groupby("scenario", as_index=False)[list(METRICS)]
        .agg(["mean", "std", "median"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "outlier_sensitivity_summary.csv", index=False)

    payload = {
        "experiment": "paired_outlier_training_sensitivity",
        "candidate_ids": list(CANDIDATE_IDS),
        "seed_count": seed_count,
        "seeds": list(range(start_seed, start_seed + seed_count)),
        "split": "Identical 80/20 random holdout for both scenarios per seed.",
        "baseline": "Fit on all rows in the training partition.",
        "sensitivity": (
            "Exclude IDs 524 and 1299 from the training partition only. "
            "Keep every validation row, including either candidate if selected."
        ),
        "model_params": MODEL_PARAMS,
        "interpretation": (
            "This is a sensitivity analysis, not permission to remove records "
            "from the production dataset. Compare paired per-seed metrics and "
            "candidate-row errors; no model settings are tuned."
        ),
        "mean_paired_deltas_filtered_minus_baseline": {
            metric: float(paired[f"delta_{metric}"].mean()) for metric in METRICS
        },
        "limitations": (
            "Repeated random holdouts overlap and are not independent external "
            "test sets. The result measures sensitivity on this dataset only."
        ),
    }
    (output_dir / "outlier_sensitivity_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return results, paired, candidate_errors


def main() -> None:
    """Run the paired sensitivity experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    results, paired, candidate_errors = run_sensitivity(
        seed_count=args.seeds,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
    )
    print(results.groupby("scenario")[list(METRICS)].mean().to_string())
    print("\nMean paired deltas (filtered - baseline):")
    print(paired[[f"delta_{metric}" for metric in METRICS]].mean().to_string())
    print(f"Candidate validation predictions: {len(candidate_errors)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
