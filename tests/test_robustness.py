"""Tests for repeated-split robustness and row-level error reporting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.robustness as robustness
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


def test_run_robustness_executes_small_end_to_end_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = 30
    frame = pd.DataFrame(
        {
            "Id": range(1, rows + 1),
            "GrLivArea": [900 + index * 25 for index in range(rows)],
            "OverallQual": [4 + index % 6 for index in range(rows)],
            "GarageArea": [
                0 if index % 4 == 0 else 250 + index for index in range(rows)
            ],
            "SalePrice": [80000 + index * 7500 for index in range(rows)],
        }
    )
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(robustness, "TRAIN_FILE", data_path)
    monkeypatch.setattr(
        robustness,
        "MODEL_PARAMS",
        {**robustness.MODEL_PARAMS, "n_estimators": 5},
    )

    output_dir = tmp_path / "robustness"
    per_seed, summary = robustness.run_robustness(
        seed_count=3,
        start_seed=10,
        output_dir=output_dir,
    )

    assert len(per_seed) == 3
    assert per_seed["seed"].tolist() == [10, 11, 12]
    assert per_seed["train_rows"].eq(24).all()
    assert per_seed["validation_rows"].eq(6).all()
    assert set(summary["metric"]) == {"MAE", "RMSE", "R2", "RMSLE"}
    assert (output_dir / "robustness_by_seed.csv").is_file()
    assert (output_dir / "robustness_summary.csv").is_file()
    assert (output_dir / "robustness_summary.json").is_file()
    assert (output_dir / "robustness_metrics.png").is_file()



def test_run_seed_diagnostics_saves_row_level_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = 30
    frame = pd.DataFrame(
        {
            "Id": range(1, rows + 1),
            "GrLivArea": [900 + index * 25 for index in range(rows)],
            "OverallQual": [4 + index % 6 for index in range(rows)],
            "GarageArea": [
                0 if index % 4 == 0 else 250 + index for index in range(rows)
            ],
            "SalePrice": [80000 + index * 7500 for index in range(rows)],
        }
    )
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(robustness, "TRAIN_FILE", data_path)
    monkeypatch.setattr(
        robustness,
        "MODEL_PARAMS",
        {**robustness.MODEL_PARAMS, "n_estimators": 5},
    )

    output_dir = tmp_path / "diagnostics"
    predictions, summary = robustness.run_seed_diagnostics(
        [10, 11], output_dir=output_dir
    )

    assert len(predictions) == 12
    assert set(predictions["seed"]) == {10, 11}
    assert {
        "id",
        "actual_price",
        "predicted_price",
        "absolute_error",
        "percentage_error",
        "price_band",
        "OverallQual",
        "GrLivArea",
    }.issubset(predictions.columns)
    assert predictions["absolute_error"].ge(0).all()
    assert predictions["percentage_error"].ge(0).all()
    assert not summary.empty
    assert (output_dir / "robustness_row_level_diagnostics.csv").is_file()
    assert (output_dir / "robustness_error_by_seed_and_price_band.csv").is_file()


def test_run_seed_diagnostics_rejects_duplicate_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        robustness.run_seed_diagnostics([47, 47], output_dir=tmp_path)
