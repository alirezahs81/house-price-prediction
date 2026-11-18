"""Tests for repeated-split robustness and row-level error reporting."""

import numpy as np
import pandas as pd
import pytest

from src.robustness import (
    build_prediction_diagnostics,
    make_price_band_edges,
    summarize_errors_by_seed_and_band,
    summarize_results,
    validate_seed_count,
)


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
    with pytest.raises(ValueError, match="Missing required"):
        summarize_results(pd.DataFrame({"MAE": [1.0, 2.0]}))


def test_summarize_results_requires_two_rows() -> None:
    results = pd.DataFrame(
        {"MAE": [1.0], "RMSE": [1.0], "R2": [0.5], "RMSLE": [0.1]}
    )
    with pytest.raises(ValueError, match="At least two"):
        summarize_results(results)


def test_make_price_band_edges_returns_strictly_increasing_edges() -> None:
    edges = make_price_band_edges([100, 200, 300, 400, 500, 600])
    assert np.all(np.diff(edges) > 0)
    assert np.isneginf(edges[0])
    assert np.isposinf(edges[-1])


def test_build_prediction_diagnostics_preserves_ids_and_errors() -> None:
    edges = make_price_band_edges([100, 200, 300, 400, 500, 600])
    result = build_prediction_diagnostics(
        seed=47,
        sample_ids=[101, 102],
        actual=[200.0, 500.0],
        predicted=[180.0, 550.0],
        price_band_edges=edges,
    )
    assert result["sample_id"].tolist() == [101, 102]
    assert result["seed"].tolist() == [47, 47]
    assert result["residual"].tolist() == pytest.approx([20.0, -50.0])
    assert result["absolute_error"].tolist() == pytest.approx([20.0, 50.0])
    assert result["price_band"].notna().all()


def test_build_prediction_diagnostics_rejects_misaligned_inputs() -> None:
    edges = make_price_band_edges([100, 200, 300, 400, 500, 600])
    with pytest.raises(ValueError, match="equal lengths"):
        build_prediction_diagnostics(
            seed=47,
            sample_ids=[101],
            actual=[200.0, 300.0],
            predicted=[180.0, 290.0],
            price_band_edges=edges,
        )


def test_summarize_errors_by_seed_and_band_calculates_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "seed": [47, 47, 56],
            "price_band": ["Q1", "Q1", "Q2"],
            "actual": [100.0, 200.0, 300.0],
            "predicted": [90.0, 220.0, 270.0],
            "residual": [10.0, -20.0, 30.0],
            "absolute_error": [10.0, 20.0, 30.0],
        }
    )
    result = summarize_errors_by_seed_and_band(predictions)
    first = result[(result["seed"] == 47) & (result["price_band"] == "Q1")].iloc[0]
    assert first["rows"] == 2
    assert first["MAE"] == pytest.approx(15.0)
    assert first["RMSE"] == pytest.approx(np.sqrt(250.0))
