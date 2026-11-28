"""Tests for leakage-safe nested cross-validation."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

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



def test_inner_search_receives_only_outer_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify each inner search is restricted to its matching outer train fold."""
    from sklearn.model_selection import KFold, RandomizedSearchCV

    import src.nested_cv as nested_cv

    recorded_search_inputs: list[tuple[int, ...]] = []

    class RecordingRandomizedSearchCV(RandomizedSearchCV):
        def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params: Any) -> Any:
            recorded_search_inputs.append(tuple(sorted(X["x"].astype(int).tolist())))
            return super().fit(X, y, **fit_params)

    monkeypatch.setattr(nested_cv, "RandomizedSearchCV", RecordingRandomizedSearchCV)

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.sin(np.arange(12, dtype=float)))
    outer_splits = 3
    inner_splits = 2
    seed = 19

    expected_outer_train_rows = []
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    for train_idx, _ in outer_cv.split(X, y):
        expected_outer_train_rows.append(tuple(sorted(train_idx.tolist())))

    summary, tuning_rows, fold_rows = nested_cv_model_comparison(
        X_train=X,
        y_train=y,
        models={"tree": DecisionTreeRegressor(random_state=seed)},
        search_spaces={"tree": {"model__max_depth": [1, 2]}},
        pipeline_factory=lambda estimator, _frame: Pipeline([("model", estimator)]),
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        n_iter=2,
        random_state=seed,
    )

    assert recorded_search_inputs == expected_outer_train_rows
    assert len(fold_rows) == outer_splits
    assert len(summary) == 1
    assert len(tuning_rows) == outer_splits * 2
    assert {row["stage"] for row in tuning_rows} == {"nested_inner_tuning"}
    assert all(np.isfinite(row["cv_log_rmse_mean"]) for row in tuning_rows)
