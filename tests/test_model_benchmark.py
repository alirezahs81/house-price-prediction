"""Tests for paired repeated-split model benchmarking."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import src.model_benchmark as benchmark


def test_validate_seed_count_requires_two_splits() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        benchmark.validate_seed_count(1)


def test_summarize_results_groups_metrics_by_model() -> None:
    rows = []
    for model, offset in (("A", 0.0), ("B", 10.0)):
        for seed in range(2):
            rows.append(
                {
                    "model": model,
                    "MAE": 10.0 + offset + seed,
                    "RMSE": 12.0 + offset + seed,
                    "R2": 0.7 + seed * 0.1,
                    "RMSLE": 0.1 + seed * 0.01,
                }
            )
    summary = benchmark.summarize_results(pd.DataFrame(rows))
    model_a_mae = summary[
        (summary["model"] == "A") & (summary["metric"] == "MAE")
    ].iloc[0]
    assert model_a_mae["mean"] == pytest.approx(10.5)
    assert model_a_mae["std"] == pytest.approx(0.70710678)


def test_summarize_results_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required"):
        benchmark.summarize_results(pd.DataFrame({"model": ["A", "A"]}))


def test_run_benchmark_uses_shared_splits_and_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = 40
    frame = pd.DataFrame(
        {
            "Id": range(1, rows + 1),
            "GrLivArea": [850 + index * 30 for index in range(rows)],
            "OverallQual": [3 + index % 7 for index in range(rows)],
            "GarageArea": [
                0 if index % 4 == 0 else 200 + index for index in range(rows)
            ],
            "SalePrice": [75000 + index * 6500 for index in range(rows)],
        }
    )
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(benchmark, "TRAIN_FILE", data_path)
    monkeypatch.setattr(
        benchmark,
        "build_benchmark_models",
        lambda seed: {
            "DummyA": DummyRegressor(strategy="mean"),
            "DummyB": DummyRegressor(strategy="median"),
        },
    )

    output_dir = tmp_path / "benchmark"
    per_seed, summary = benchmark.run_benchmark(
        seed_count=2,
        start_seed=20,
        output_dir=output_dir,
    )

    assert len(per_seed) == 4
    assert set(per_seed["seed"]) == {20, 21}
    assert set(per_seed["model"]) == {"DummyA", "DummyB"}
    assert per_seed.groupby("seed")["validation_rows"].nunique().eq(1).all()
    assert set(summary["metric"]) == {"MAE", "RMSE", "R2", "RMSLE"}
    assert (output_dir / "model_benchmark_by_seed.csv").is_file()
    assert (output_dir / "model_benchmark_summary.csv").is_file()
    assert (output_dir / "model_benchmark_summary.json").is_file()
    assert (output_dir / "model_benchmark_rmse.png").is_file()
