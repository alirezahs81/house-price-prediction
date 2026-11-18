"""Tests for paired outlier sensitivity analysis."""

import pandas as pd
import pytest

from src.outlier_sensitivity import (
    paired_metric_comparison,
    training_rows_without_ids,
)


def test_training_rows_without_ids_excludes_only_requested_training_records() -> None:
    training = pd.DataFrame(
        {"Id": [10, 524, 1299, 20], "feature": [1, 2, 3, 4]}
    )
    filtered = training_rows_without_ids(training)
    assert filtered["Id"].tolist() == [10, 20]
    assert training["Id"].tolist() == [10, 524, 1299, 20]


def test_training_rows_without_ids_requires_identifier_column() -> None:
    with pytest.raises(ValueError, match="Id column"):
        training_rows_without_ids(pd.DataFrame({"feature": [1, 2]}))


def test_paired_metric_comparison_calculates_filtered_minus_baseline() -> None:
    results = pd.DataFrame(
        [
            {
                "seed": 42,
                "scenario": "baseline",
                "MAE": 100.0,
                "RMSE": 150.0,
                "R2": 0.8,
                "RMSLE": 0.1,
            },
            {
                "seed": 42,
                "scenario": "exclude_candidates_from_training",
                "MAE": 90.0,
                "RMSE": 130.0,
                "R2": 0.82,
                "RMSLE": 0.09,
            },
        ]
    )
    comparison = paired_metric_comparison(results).iloc[0]
    assert comparison["delta_MAE"] == pytest.approx(-10.0)
    assert comparison["delta_RMSE"] == pytest.approx(-20.0)
    assert comparison["delta_R2"] == pytest.approx(0.02)
    assert comparison["delta_RMSLE"] == pytest.approx(-0.01)


def test_paired_metric_comparison_requires_both_scenarios() -> None:
    results = pd.DataFrame(
        [
            {
                "seed": 42,
                "scenario": "baseline",
                "MAE": 100.0,
                "RMSE": 150.0,
                "R2": 0.8,
                "RMSLE": 0.1,
            }
        ]
    )
    with pytest.raises(ValueError, match="exactly one result"):
        paired_metric_comparison(results)
