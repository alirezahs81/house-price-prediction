"""Tests for exploratory analysis module."""
import pandas as pd
import pytest
from src.exploratory_analysis import run_eda
from src.config import TRAIN_FILE


def test_run_eda_returns_overview():
    """Test that run_eda returns a dictionary with expected keys."""
    result = run_eda(TRAIN_FILE)
    
    assert isinstance(result, dict)
    expected_keys = {
        "train_rows", "columns_including_target", "numeric_columns",
        "categorical_columns", "duplicate_rows", "missing_cells",
        "saleprice_mean", "saleprice_median", "saleprice_min", "saleprice_max",
        "top_missing_columns"
    }
    assert expected_keys.issubset(set(result.keys()))


def test_run_eda_creates_figures():
    """Test that run_eda creates figure files."""
    from src.config import FIGURES_DIR
    
    run_eda(TRAIN_FILE)
    
    expected_figures = [
        "missing_values.png",
        "saleprice_distribution.png",
        "numeric_distributions.png",
        "categorical_distributions.png",
        "target_correlations.png",
    ]
    
    for fig in expected_figures:
        assert (FIGURES_DIR / fig).exists(), f"Missing figure: {fig}"


def test_run_eda_creates_metrics():
    """Test that run_eda creates metrics files."""
    from src.config import METRICS_DIR
    
    run_eda(TRAIN_FILE)
    
    expected_metrics = [
        "data_overview.json",
        "numeric_outlier_summary.csv",
    ]
    
    for metric in expected_metrics:
        assert (METRICS_DIR / metric).exists(), f"Missing metric: {metric}"


def test_eda_handles_missing_data():
    """Test EDA with synthetic data containing missing values."""
    df = pd.DataFrame({
        "Id": [1, 2, 3, 4, 5],
        "SalePrice": [100000, 200000, 150000, 300000, 250000],
        "LotArea": [10000, 12000, None, 15000, 11000],
        "Neighborhood": ["A", "B", "A", None, "C"],
        "OverallQual": [5, 7, 6, 8, 7],
    })
    
    # Save temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        result = run_eda(f.name)
    
    assert result["missing_cells"] > 0
    assert "LotArea" in str(result["top_missing_columns"]) or "Neighborhood" in str(result["top_missing_columns"])