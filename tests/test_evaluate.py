import numpy as np
import pandas as pd
import pytest

from src.evaluate import regression_metrics, save_diagnostics


def test_regression_metrics_reject_misaligned_or_invalid_values():
    with pytest.raises(ValueError, match="identical shapes"):
        regression_metrics([1, 2], [1])
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, np.nan], [1, 2])


def test_save_diagnostics_writes_error_reports(tmp_path, monkeypatch):
    import src.evaluate as evaluate

    monkeypatch.setattr(evaluate, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path / "metrics")

    actual = np.array([100_000, 120_000, 180_000, 250_000, 400_000])
    predicted = np.array([105_000, 110_000, 170_000, 230_000, 350_000])
    metrics = save_diagnostics(
        actual, predicted, sample_ids=np.array([1, 2, 3, 4, 5])
    )

    assert metrics["RMSE"] > 0
    residuals = pd.read_csv(tmp_path / "validation_residuals.csv")
    assert list(residuals["sample_id"]) == [1, 2, 3, 4, 5]
    assert (tmp_path / "worst_validation_errors.csv").exists()
    bands = pd.read_csv(tmp_path / "validation_error_by_price_band.csv")
    assert bands["rows"].sum() == 5
    assert (tmp_path / "figures" / "actual_vs_predicted.png").exists()
