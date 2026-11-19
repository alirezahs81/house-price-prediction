"""Tests for the train-only outlier sensitivity experiment."""

from pathlib import Path

import pandas as pd
import pytest

import src.outlier_sensitivity as sensitivity


def _synthetic_training_frame() -> pd.DataFrame:
    rows = 40
    identifiers = list(range(1, rows + 1))
    identifiers[4] = 524
    identifiers[19] = 1299
    return pd.DataFrame(
        {
            "Id": identifiers,
            "GrLivArea": [850 + index * 35 for index in range(rows)],
            "OverallQual": [3 + index % 7 for index in range(rows)],
            "GarageArea": [
                0 if index % 5 == 0 else 220 + index * 3
                for index in range(rows)
            ],
            "SalePrice": [75000 + index * 6200 for index in range(rows)],
        }
    )


def test_run_outlier_sensitivity_keeps_validation_rows_and_saves_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _synthetic_training_frame()
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(sensitivity, "TRAIN_FILE", data_path)
    monkeypatch.setattr(
        sensitivity,
        "MODEL_PARAMS",
        {**sensitivity.MODEL_PARAMS, "n_estimators": 5},
    )

    output_dir = tmp_path / "sensitivity"
    per_seed, summary, flagged_predictions = sensitivity.run_outlier_sensitivity(
        flagged_ids=(524, 1299),
        seed_count=3,
        start_seed=10,
        output_dir=output_dir,
    )

    assert per_seed["seed"].tolist() == [10, 11, 12]
    assert per_seed["validation_rows"].eq(8).all()
    assert per_seed["train_rows_baseline"].eq(32).all()
    assert (
        per_seed["train_rows_sensitivity"]
        == per_seed["train_rows_baseline"]
        - per_seed["flagged_rows_removed_from_training"]
    ).all()
    assert per_seed["flagged_rows_in_validation"].between(0, 2).all()
    assert set(summary["metric"]) == {"MAE", "RMSE", "R2", "RMSLE"}
    assert set(flagged_predictions.columns) >= {
        "seed",
        "id",
        "actual_price",
        "baseline_prediction",
        "sensitivity_prediction",
    }
    assert (output_dir / "outlier_sensitivity_by_seed.csv").is_file()
    assert (output_dir / "outlier_sensitivity_summary.csv").is_file()
    assert (output_dir / "flagged_record_validation_predictions.csv").is_file()
    assert (output_dir / "outlier_sensitivity_metadata.json").is_file()
    assert (output_dir / "outlier_sensitivity_comparison.png").is_file()


def test_run_outlier_sensitivity_rejects_missing_flagged_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _synthetic_training_frame()
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(sensitivity, "TRAIN_FILE", data_path)

    with pytest.raises(ValueError, match="not found"):
        sensitivity.run_outlier_sensitivity(
            flagged_ids=(999999,),
            seed_count=2,
            output_dir=tmp_path / "missing",
        )


def test_run_outlier_sensitivity_rejects_duplicate_flagged_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        sensitivity.run_outlier_sensitivity(
            flagged_ids=(524, 524),
            seed_count=2,
        )


def test_run_outlier_sensitivity_requires_two_seeds() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        sensitivity.run_outlier_sensitivity(seed_count=1)
