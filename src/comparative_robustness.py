"""Paired 30-seed comparison of baseline and robust training strategies.

All strategies use identical 80/20 splits. The influence-control strategy
removes the two pre-identified rows from training only; every validation row
remains in evaluation. No validation result is used for tuning.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.config import RANDOM_STATE, TARGET, TRAIN_FILE
from src.data_validation import validate_training_data
from src.evaluate import regression_metrics
from src.modeling import make_pipeline
from src.preprocessing import prepare_features, transform_absence_categories
from src.robustness import MODEL_PARAMS, split_holdout, validate_seed_count

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("reports/comparative_robustness")
INFLUENTIAL_IDS = (524, 1299)
SCENARIOS = ("baseline", "influence_control", "absolute_error")


def _fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    loss: str,
) -> np.ndarray:
    """Fit one fixed GradientBoosting pipeline and return positive prices."""
    params = {**MODEL_PARAMS, "loss": loss}
    pipeline = make_pipeline(GradientBoostingRegressor(**params), X_train)
    pipeline.fit(X_train, y_train)
    return np.maximum(np.expm1(pipeline.predict(X_valid)), 1.0)


def _price_band(value: float) -> str:
    """Assign an actual sale price to a stable reporting band."""
    if value <= 100_000:
        return "<=100k"
    if value <= 150_000:
        return "100k-150k"
    if value <= 200_000:
        return "150k-200k"
    if value <= 250_000:
        return "200k-250k"
    return ">250k"


def run_comparative_experiment(
    seed_count: int = 30,
    start_seed: int = RANDOM_STATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run paired baseline, influence-control, and absolute-error experiments."""
    validate_seed_count(seed_count)
    frame = pd.read_csv(TRAIN_FILE)
    validate_training_data(frame)
    if "Id" not in frame.columns:
        raise ValueError("The training data must contain the Id column.")

    features = transform_absence_categories(prepare_features(frame))
    if TARGET in features.columns or "Id" in features.columns:
        raise ValueError("Target or identifier leakage detected in model features.")
    target_log = np.log1p(frame[TARGET].astype(float))

    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []

    for seed in range(start_seed, start_seed + seed_count):
        X_train, X_valid, y_train, y_valid_log = split_holdout(
            features, target_log, seed=seed, test_size=0.2
        )
        actual = np.expm1(y_valid_log).to_numpy(dtype=float)
        validation_ids = frame.loc[X_valid.index, "Id"].to_numpy()
        training_ids = frame.loc[X_train.index, "Id"]
        control_mask = ~training_ids.isin(INFLUENTIAL_IDS)
        X_control = X_train.loc[control_mask]
        y_control = y_train.loc[control_mask]
        removed_ids = sorted(set(training_ids) & set(INFLUENTIAL_IDS))
        if X_control.empty:
            raise RuntimeError(
                "Influence control unexpectedly removed all training rows."
            )

        scenario_predictions = {
            "baseline": _fit_predict(X_train, y_train, X_valid, "huber"),
            "influence_control": _fit_predict(
                X_control, y_control, X_valid, "huber"
            ),
            "absolute_error": _fit_predict(
                X_train, y_train, X_valid, "absolute_error"
            ),
        }

        for scenario, predicted in scenario_predictions.items():
            regression_scores = regression_metrics(actual, predicted)
            metric_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "train_rows": len(
                        X_control if scenario == "influence_control" else X_train
                    ),
                    "validation_rows": len(X_valid),
                    "training_influential_ids_removed": (
                        ",".join(map(str, removed_ids))
                        if scenario == "influence_control"
                        else ""
                    ),
                    **{
                        name: float(regression_scores[name])
                        for name in ("MAE", "RMSE", "R2", "RMSLE")
                    },
                }
            )
            details = pd.DataFrame(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "id": validation_ids,
                    "actual_price": actual,
                    "predicted_price": predicted,
                }
            )
            details["absolute_error"] = np.abs(
                details["actual_price"] - details["predicted_price"]
            )
            details["percentage_error"] = (
                details["absolute_error"] / details["actual_price"] * 100.0
            )
            details["price_band"] = details["actual_price"].map(_price_band)
            prediction_rows.append(details)

        LOGGER.info("Completed paired comparison for seed %d.", seed)

    per_seed = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary = (
        per_seed.groupby("scenario", sort=False)[["MAE", "RMSE", "R2", "RMSLE"]]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary = summary.set_axis(
        [
            "_".join(str(part) for part in column if part).rstrip("_")
            if isinstance(column, tuple)
            else column
            for column in summary.columns
        ],
        axis=1,
    )
    price_band_summary = (
        predictions.groupby(["scenario", "price_band"], observed=True, sort=True)
        .agg(
            rows=("absolute_error", "size"),
            MAE=("absolute_error", "mean"),
            RMSE=("absolute_error", lambda values: float(np.sqrt(np.mean(values**2)))),
            mean_percentage_error=("percentage_error", "mean"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "comparative_by_seed.csv", index=False)
    summary.to_csv(output_dir / "comparative_summary.csv", index=False)
    price_band_summary.to_csv(
        output_dir / "comparative_by_price_band.csv", index=False
    )
    predictions.to_csv(
        output_dir / "comparative_row_level_predictions.csv", index=False
    )
    metadata = {
        "experiment": "paired_30_seed_model_comparison",
        "seed_count": seed_count,
        "seeds": list(range(start_seed, start_seed + seed_count)),
        "split": "Identical deterministic 80/20 random holdout per seed",
        "scenarios": {
            "baseline": (
                "GradientBoostingRegressor with the existing fixed Huber "
                "loss settings"
            ),
            "influence_control": (
                "Same Huber model; IDs 524 and 1299 are excluded from training "
                "only when present there. All validation rows remain included."
            ),
            "absolute_error": (
                "Same fixed GradientBoosting settings with absolute_error loss; "
                "no parameter search or validation-based tuning."
            ),
        },
        "validation_policy": (
            "Every scenario is evaluated on the complete validation partition "
            "for its seed. No row is removed from validation."
        ),
        "limitations": (
            "Repeated holdouts reuse observations and are not independent external "
            "test sets. This comparison measures split sensitivity on this dataset."
        ),
        "influential_ids": list(INFLUENTIAL_IDS),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "comparative_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return per_seed, summary, price_band_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    per_seed, summary, bands = run_comparative_experiment(
        seed_count=args.seeds,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
    )
    print(f"Completed {len(per_seed)} model/seed evaluations.")
    print(summary.to_string(index=False))
    print("\nPrice-band diagnostics:")
    print(bands.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
