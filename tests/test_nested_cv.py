"""Tests for leakage-safe nested cross-validation."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.nested_cv import nested_cv_model_comparison


def _pipeline(estimator: Any, _X: pd.DataFrame) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def test_nested_cv_uses_same_outer_validation_indices_for_each_family() -> None:
    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    y = pd.Series(np.arange(30, dtype=float))
    models = {
        "median_a": DummyRegressor(strategy="median"),
        "median_b": DummyRegressor(strategy="mean"),
    }

    summary, _, fold_rows = nested_cv_model_comparison(
        X_train=X,
        y_train=y,
        models=models,
        search_spaces={},
        pipeline_factory=_pipeline,
        outer_splits=5,
        inner_splits=3,
        random_state=11,
    )

    assert len(summary) == 2
    assert set(summary["cv_folds"]) == {5}
    indices_a = [
        row["validation_indices"] for row in fold_rows if row["model"] == "median_a"
    ]
    indices_b = [
        row["validation_indices"] for row in fold_rows if row["model"] == "median_b"
    ]
    assert indices_a == indices_b
    all_indices = sorted(index for fold in indices_a for index in fold)
    assert all_indices == list(range(len(X)))


def test_nested_cv_rejects_invalid_fold_counts() -> None:
    X = pd.DataFrame({"x": np.arange(10, dtype=float)})
    y = pd.Series(np.arange(10, dtype=float))

    with pytest.raises(ValueError, match="CV splits"):
        nested_cv_model_comparison(
            X_train=X,
            y_train=y,
            models={"median": DummyRegressor()},
            search_spaces={},
            pipeline_factory=_pipeline,
            outer_splits=1,
        )
