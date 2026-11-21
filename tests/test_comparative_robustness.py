"""Tests for paired comparative robustness experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.comparative_robustness as comparative


def _sample_frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": range(1, rows + 1),
            "GrLivArea": [900 + index * 20 for index in range(rows)],
            "OverallQual": [4 + index % 6 for index in range(rows)],
            "GarageArea": [0 if index % 4 == 0 else 250 + index for index in range(rows)],
            "SalePrice": [80000 + index * 5000 for index in range(rows)],
        }
    )


def test_comparison_uses_all_validation_rows_and_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _sample_frame()
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(comparative, "TRAIN_FILE", data_path)
    monkeypatch.setattr(
        comparative,
        "MODEL_PARAMS",
        {**comparative.MODEL_PARAMS, "n_estimators": 5},
    )

    per_seed, summary, bands = comparative.run_comparative_experiment(
        seed_count=2, start_seed=10, output_dir=tmp_path / "reports"
    )

    assert set(per_seed["scenario"]) == set(comparative.SCENARIOS)
    assert per_seed.groupby("seed")["validation_rows"].nunique().eq(1).all()
    assert per_seed["validation_rows"].eq(8).all()
    assert set(summary["scenario"]) == set(comparative.SCENARIOS)
    assert not bands.empty
    for filename in (
        "comparative_by_seed.csv",
        "comparative_paired_differences.csv",
        "comparative_paired_summary.csv",
        "comparative_summary.csv",
        "comparative_by_price_band.csv",
        "comparative_row_level_predictions.csv",
        "comparative_summary.json",
    ):
        assert (tmp_path / "reports" / filename).is_file()


def test_influence_control_never_removes_validation_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _sample_frame()
    frame.loc[frame["Id"].isin([11, 12]), "SalePrice"] = [180000, 190000]
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(comparative, "TRAIN_FILE", data_path)
    monkeypatch.setattr(
        comparative,
        "MODEL_PARAMS",
        {**comparative.MODEL_PARAMS, "n_estimators": 5},
    )

    per_seed, _, _ = comparative.run_comparative_experiment(
        seed_count=2, start_seed=42, output_dir=tmp_path / "reports"
    )
    control = per_seed[per_seed["scenario"] == "influence_control"]
    assert control["validation_rows"].eq(8).all()
    assert control["train_rows"].le(32).all()


def test_comparison_requires_at_least_two_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        comparative.run_comparative_experiment(seed_count=1, output_dir=tmp_path)


def test_comparison_baseline_uses_shared_fitting_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _sample_frame()
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(comparative, "TRAIN_FILE", data_path)
    monkeypatch.setattr(
        comparative,
        "MODEL_PARAMS",
        {**comparative.MODEL_PARAMS, "n_estimators": 5},
    )
    calls: list[int] = []

    def shared_fit(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        model_params: dict[str, object] | None = None,
    ) -> np.ndarray:
        calls.append(len(X_valid))
        return np.full(len(X_valid), 150000.0)

    monkeypatch.setattr(comparative, "fit_baseline_holdout", shared_fit)
    comparative.run_comparative_experiment(
        seed_count=2, start_seed=20, output_dir=tmp_path / "reports"
    )

    assert calls == [8, 8]
