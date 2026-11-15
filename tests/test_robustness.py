"""Tests for repeated-split robustness reporting."""

import pandas as pd
import pytest

from src.robustness import summarize_results, validate_seed_count


def test_validate_seed_count_requires_two_splits() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        validate_seed_count(1)


def test_validate_seed_count_accepts_multiple_splits() -> None:
    validate_seed_count(30)


def test_summarize_results_reports_mean_and_sample_std() -> None:
    results = pd.DataFrame(
        {
            "MAE": [10.0, 20.0, 30.0],
            "RMSE": [12.0, 22.0, 32.0],
            "R2": [0.7, 0.8, 0.9],
            "RMSLE": [0.1, 0.2, 0.3],
        }
    )

    summary = summarize_results(results).set_index("metric")

    assert summary.loc["MAE", "mean"] == pytest.approx(20.0)
    assert summary.loc["MAE", "std"] == pytest.approx(10.0)
    assert summary.loc["R2", "median"] == pytest.approx(0.8)
    assert summary.loc["RMSLE", "p10"] == pytest.approx(0.12)


def test_summarize_results_rejects_missing_metrics() -> None:
    results = pd.DataFrame({"MAE": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Missing required"):
        summarize_results(results)


def test_summarize_results_requires_two_rows() -> None:
    results = pd.DataFrame(
        {
            "MAE": [1.0],
            "RMSE": [1.0],
            "R2": [0.5],
            "RMSLE": [0.1],
        }
    )

    with pytest.raises(ValueError, match="At least two"):
        summarize_results(results)
