"""Tests for paired price-weighted robustness comparisons."""

# The integration test deliberately uses identical seeded holdouts for both models.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.weighted_robustness as weighted


def test_make_price_weights_are_bounded_and_mean_one() -> None:
    prices = pd.Series([50_000, 80_000, 120_000, 200_000, 500_000])
    weights = weighted.make_price_weights(prices)

    assert len(weights) == len(prices)
    assert weights.mean() == pytest.approx(1.0)
    assert weights.min() >= 0.75
    assert weights.max() <= 1.75


def test_make_price_weights_reject_invalid_prices() -> None:
    with pytest.raises(ValueError, match="finite, positive"):
        weighted.make_price_weights(pd.Series([100_000, np.nan]))


def test_validate_seed_count_requires_two_or_more() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        weighted.validate_seed_count(1)
    weighted.validate_seed_count(2)


def test_summarize_comparison_contains_paired_deltas() -> None:
    frame = pd.DataFrame(
        {
            "baseline_overall_MAE": [10.0, 12.0],
            "weighted_overall_MAE": [9.0, 14.0],
            "baseline_overall_RMSE": [11.0, 13.0],
            "weighted_overall_RMSE": [10.0, 14.0],
            "baseline_overall_R2": [0.8, 0.7],
            "weighted_overall_R2": [0.82, 0.72],
            "baseline_overall_RMSLE": [0.1, 0.12],
            "weighted_overall_RMSLE": [0.09, 0.13],
            "baseline_high_value_MAE": [20.0, 22.0],
            "weighted_high_value_MAE": [18.0, 24.0],
            "baseline_high_value_RMSE": [25.0, 27.0],
            "weighted_high_value_RMSE": [23.0, 29.0],
            "baseline_high_value_R2": [0.6, 0.5],
            "weighted_high_value_R2": [0.62, 0.52],
            "baseline_high_value_RMSLE": [0.15, 0.17],
            "weighted_high_value_RMSLE": [0.14, 0.18],
        }
    )
    summary = weighted.summarize_comparison(frame)
    row = summary[
        (summary["segment"] == "overall")
        & (summary["metric"] == "MAE")
        & (summary["result"] == "paired_weighted_minus_baseline")
    ].iloc[0]

    assert row["mean"] == pytest.approx(0.5)
    assert row["weighted_wins"] == 1
    assert row["baseline_wins"] == 1

    r2_row = summary[
        (summary["segment"] == "overall")
        & (summary["metric"] == "R2")
        & (summary["result"] == "paired_weighted_minus_baseline")
    ].iloc[0]
    assert r2_row["weighted_wins"] == 2
    assert r2_row["baseline_wins"] == 0

    high_value_r2_row = summary[
        (summary["segment"] == "high_value")
        & (summary["metric"] == "R2")
        & (summary["result"] == "paired_weighted_minus_baseline")
    ].iloc[0]
    assert high_value_r2_row["weighted_wins"] == 2
    assert high_value_r2_row["baseline_wins"] == 0


def test_run_weighted_robustness_uses_paired_splits_and_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = 60
    frame = pd.DataFrame(
        {
            "Id": range(1, rows + 1),
            "GrLivArea": [800 + index * 20 for index in range(rows)],
            "OverallQual": [4 + index % 6 for index in range(rows)],
            "GarageArea": [
                0 if index % 4 == 0 else 250 + index for index in range(rows)
            ],
            "SalePrice": [75_000 + index * 5_000 for index in range(rows)],
        }
    )
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(weighted, "TRAIN_FILE", data_path)
    monkeypatch.setattr(
        weighted,
        "MODEL_PARAMS",
        {**weighted.MODEL_PARAMS, "n_estimators": 5},
    )

    per_seed, summary = weighted.run_weighted_robustness(
        seed_count=3,
        start_seed=10,
        output_dir=tmp_path / "weighted",
    )

    assert per_seed["seed"].tolist() == [10, 11, 12]
    assert per_seed["train_rows"].eq(48).all()
    assert per_seed["validation_rows"].eq(12).all()
    assert per_seed["high_value_rows"].ge(0).all()
    empty_high_value = per_seed["high_value_rows"].eq(0)
    if empty_high_value.any():
        assert per_seed.loc[
            empty_high_value, "baseline_high_value_RMSE"
        ].isna().all()
        assert per_seed.loc[
            empty_high_value, "weighted_high_value_RMSE"
        ].isna().all()
    assert {
        "baseline_overall_RMSE",
        "weighted_overall_RMSE",
        "delta_overall_RMSE",
        "baseline_high_value_RMSE",
        "weighted_high_value_RMSE",
        "delta_high_value_RMSE",
    }.issubset(per_seed.columns)
    assert set(summary["result"]) == {
        "baseline",
        "price_weighted",
        "paired_weighted_minus_baseline",
    }
    assert (tmp_path / "weighted" / "weighted_robustness_by_seed.csv").is_file()
    assert (tmp_path / "weighted" / "weighted_robustness_summary.csv").is_file()
    assert (tmp_path / "weighted" / "weighted_robustness_summary.json").is_file()
    assert (tmp_path / "weighted" / "overall_rmse_by_seed.png").is_file()
    assert (tmp_path / "weighted" / "high_value_rmse_by_seed.png").is_file()
