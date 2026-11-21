"""Run a train-only sensitivity analysis for pre-identified influential records.

The flagged records are excluded only from each training partition. Every
validation partition remains unchanged, so the experiment never improves its
headline metrics by removing difficult validation observations.
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

from src.config import RANDOM_STATE, TARGET, TRAIN_FILE
from src.data_validation import validate_training_data
from src.evaluate import regression_metrics
from src.modeling import make_pipeline
from src.preprocessing import prepare_features, transform_absence_categories
from src.robustness import MODEL_PARAMS, split_holdout

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("reports/outlier_sensitivity")
DEFAULT_FLAGGED_IDS = (524, 1299)
METRICS = ("MAE", "RMSE", "R2", "RMSLE")


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return the four primary regression metrics."""
    metrics = regression_metrics(actual, predicted)
    return {name: float(metrics[name]) for name in METRICS}


def run_outlier_sensitivity(
    flagged_ids: tuple[int, ...] = DEFAULT_FLAGGED_IDS,
    seed_count: int = 30,
    start_seed: int = RANDOM_STATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare baseline and flagged-row-excluded training on identical splits."""
    if not flagged_ids:
        raise ValueError("At least one flagged record ID is required.")
    if len(set(flagged_ids)) != len(flagged_ids):
        raise ValueError("Flagged record IDs must be unique.")
    if seed_count < 2:
        raise ValueError("seed_count must be at least 2.")

    frame = pd.read_csv(TRAIN_FILE)
    validate_training_data(frame)
    if "Id" not in frame.columns:
        raise ValueError("Training data must contain the Id column.")

    id_counts = frame["Id"].value_counts()
    missing_ids = [record_id for record_id in flagged_ids if record_id not in id_counts]
    duplicate_ids = [
        record_id
        for record_id in flagged_ids
        if id_counts.get(record_id, 0) != 1
    ]
    if missing_ids:
        raise ValueError(f"Flagged record IDs not found: {missing_ids}")
    if duplicate_ids:
        raise ValueError(f"Flagged record IDs must occur exactly once: {duplicate_ids}")

    features = transform_absence_categories(prepare_features(frame))
    if TARGET in features.columns or "Id" in features.columns:
        raise ValueError("Target or identifier leakage detected in model features.")
    target_log = np.log1p(frame[TARGET].astype(float))

    seed_rows: list[dict[str, float | int]] = []
    flagged_prediction_rows: list[dict[str, float | int]] = []

    for seed in range(start_seed, start_seed + seed_count):
        X_train, X_valid, y_train, y_valid_log = split_holdout(
            features, target_log, seed=seed, test_size=0.2
        )
        train_ids = frame.loc[X_train.index, "Id"]
        keep_train = ~train_ids.isin(flagged_ids)
        X_train_filtered = X_train.loc[keep_train]
        y_train_filtered = y_train.loc[keep_train]
        removed_count = int((~keep_train).sum())
        if len(X_train_filtered) < 2:
            raise ValueError(
                "Too few training rows remain after the sensitivity filter."
            )

        baseline = make_pipeline(
            GradientBoostingRegressor(**MODEL_PARAMS), X_train
        )
        baseline.fit(X_train, y_train)
        sensitivity = make_pipeline(
            GradientBoostingRegressor(**MODEL_PARAMS), X_train_filtered
        )
        sensitivity.fit(X_train_filtered, y_train_filtered)

        actual = np.expm1(y_valid_log).to_numpy(dtype=float)
        baseline_pred = np.maximum(np.expm1(baseline.predict(X_valid)), 1.0)
        sensitivity_pred = np.maximum(np.expm1(sensitivity.predict(X_valid)), 1.0)

        baseline_metrics = _metric_values(actual, baseline_pred)
        sensitivity_metrics = _metric_values(actual, sensitivity_pred)
        non_flagged_valid = ~frame.loc[X_valid.index, "Id"].isin(
            flagged_ids
        ).to_numpy()
        non_flagged_baseline = _metric_values(
            actual[non_flagged_valid], baseline_pred[non_flagged_valid]
        )
        non_flagged_sensitivity = _metric_values(
            actual[non_flagged_valid], sensitivity_pred[non_flagged_valid]
        )

        row: dict[str, float | int] = {
            "seed": seed,
            "train_rows_baseline": int(len(X_train)),
            "train_rows_sensitivity": int(len(X_train_filtered)),
            "flagged_rows_removed_from_training": removed_count,
            "validation_rows": int(len(X_valid)),
            "flagged_rows_in_validation": int((~non_flagged_valid).sum()),
        }
        for metric in METRICS:
            row[f"baseline_{metric}"] = baseline_metrics[metric]
            row[f"sensitivity_{metric}"] = sensitivity_metrics[metric]
            row[f"delta_{metric}"] = (
                sensitivity_metrics[metric] - baseline_metrics[metric]
            )
            row[f"baseline_non_flagged_{metric}"] = non_flagged_baseline[metric]
            row[f"sensitivity_non_flagged_{metric}"] = non_flagged_sensitivity[metric]
        seed_rows.append(row)

        valid_ids = frame.loc[X_valid.index, "Id"].to_numpy()
        for position, record_id in enumerate(valid_ids):
            if record_id in flagged_ids:
                flagged_prediction_rows.append(
                    {
                        "seed": seed,
                        "id": int(record_id),
                        "actual_price": float(actual[position]),
                        "baseline_prediction": float(baseline_pred[position]),
                        "sensitivity_prediction": float(sensitivity_pred[position]),
                        "baseline_absolute_error": float(
                            abs(actual[position] - baseline_pred[position])
                        ),
                        "sensitivity_absolute_error": float(
                            abs(actual[position] - sensitivity_pred[position])
                        ),
                    }
                )
        LOGGER.info(
            "Seed %d completed: removed %d training row(s); baseline RMSE %.2f; "
            "sensitivity RMSE %.2f",
            seed,
            removed_count,
            baseline_metrics["RMSE"],
            sensitivity_metrics["RMSE"],
        )

    per_seed = pd.DataFrame(seed_rows)
    flagged_predictions = pd.DataFrame(flagged_prediction_rows)
    summary_rows: list[dict[str, float | int | str]] = []
    for metric in METRICS:
        deltas = per_seed[f"delta_{metric}"].astype(float)
        if metric == "R2":
            wins = int((deltas > 0).sum())
        else:
            wins = int((deltas < 0).sum())
        summary_rows.append(
            {
                "metric": metric,
                "baseline_mean": float(per_seed[f"baseline_{metric}"].mean()),
                "sensitivity_mean": float(per_seed[f"sensitivity_{metric}"].mean()),
                "mean_delta_sensitivity_minus_baseline": float(deltas.mean()),
                "delta_std": float(deltas.std(ddof=1)),
                "delta_p10": float(deltas.quantile(0.10)),
                "delta_p90": float(deltas.quantile(0.90)),
                "sensitivity_wins": wins,
                "ties": int(np.isclose(deltas, 0.0).sum()),
                "seeds": int(len(deltas)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "outlier_sensitivity_by_seed.csv", index=False)
    summary.to_csv(output_dir / "outlier_sensitivity_summary.csv", index=False)
    flagged_predictions.to_csv(
        output_dir / "flagged_record_validation_predictions.csv", index=False
    )
    metadata: dict[str, Any] = {
        "experiment": "train_only_flagged_record_sensitivity",
        "flagged_record_ids": list(flagged_ids),
        "seed_count": seed_count,
        "seeds": list(range(start_seed, start_seed + seed_count)),
        "split": {"test_size": 0.2},
        "model": "GradientBoostingRegressor",
        "model_params": MODEL_PARAMS,
        "intervention": (
            "Flagged IDs are removed only from each seed's training partition. "
            "The original validation partition is unchanged and includes flagged "
            "records whenever the split assigns them to validation."
        ),
        "interpretation": (
            "This is a sensitivity analysis, not a recommendation to delete records "
            "from the production dataset. Repeated holdouts overlap and are not "
            "independent external test sets. Metrics on all validation rows are "
            "the primary comparison; non-flagged-only metrics are secondary."
        ),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "outlier_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_titles = {
        "MAE": "MAE (USD; lower is better)",
        "RMSE": "RMSE (USD; lower is better)",
        "R2": "R² (higher is better)",
        "RMSLE": "RMSLE (lower is better)",
    }
    labels = ("Baseline", "Train-only exclusion")
    for ax, metric in zip(axes.flat, METRICS, strict=True):
        values = [
            float(per_seed[f"baseline_{metric}"].mean()),
            float(per_seed[f"sensitivity_{metric}"].mean()),
        ]
        bars = ax.bar(labels, values)
        ax.set_title(metric_titles[metric])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        value_span = max(values) - min(values)
        padding = max(
            value_span * 0.15,
            max(abs(value) for value in values) * 0.04,
            0.01,
        )
        lower = min(0.0, min(values) - padding)
        upper = max(values) + padding
        ax.set_ylim(lower, upper)
        for bar, value in zip(bars, values, strict=True):
            if metric in ("MAE", "RMSE"):
                label = f"${value:,.0f}"
            else:
                label = f"{value:.4f}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("30-seed train-only outlier sensitivity comparison")
    fig.text(
        0.5,
        0.015,
        (
            "Primary metrics use every validation row; "
            "flagged rows are excluded only from training."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_dir / "outlier_sensitivity_comparison.png", dpi=160)
    plt.close(fig)

    return per_seed, summary, flagged_predictions


def main() -> None:
    """Run the train-only sensitivity experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--flagged-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_FLAGGED_IDS),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    per_seed, summary, flagged_predictions = run_outlier_sensitivity(
        flagged_ids=tuple(args.flagged_ids),
        seed_count=args.seeds,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
    )
    print(f"Completed {len(per_seed)} paired holdout comparisons.")
    print(f"Flagged-record validation predictions: {len(flagged_predictions)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
