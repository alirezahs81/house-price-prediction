"""Leakage-safe nested cross-validation for model-family selection."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error


def nested_cv_model_comparison(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Mapping[str, Any],
    search_spaces: Mapping[str, dict[str, list[Any]]],
    pipeline_factory: Callable[[Any, pd.DataFrame], Pipeline],
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_iter: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare families with shared outer folds and fold-local tuning.

    Each outer validation fold is held out from all inner hyperparameter
    selection. All model families use the same outer fold indices.
    """
    if outer_splits < 2 or inner_splits < 2 or n_iter < 1:
        raise ValueError("CV splits must be >= 2 and n_iter must be >= 1.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have equal lengths.")
    if len(X_train) < outer_splits:
        raise ValueError("Not enough samples for the requested outer folds.")

    outer_cv = KFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    outer_indices = list(outer_cv.split(X_train, y_train))
    fold_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for model_name, estimator in models.items():
        space = search_spaces.get(model_name, {})
        for fold_number, (train_idx, valid_idx) in enumerate(outer_indices, start=1):
            X_outer_train = X_train.iloc[train_idx]
            y_outer_train = y_train.iloc[train_idx]
            X_outer_valid = X_train.iloc[valid_idx]
            y_outer_valid = y_train.iloc[valid_idx]
            pipeline = pipeline_factory(clone(estimator), X_outer_train)

            if space:
                inner_cv = KFold(
                    n_splits=inner_splits,
                    shuffle=True,
                    random_state=random_state + fold_number,
                )
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=space,
                    n_iter=min(n_iter, len(list(ParameterGrid(space)))),
                    scoring="neg_root_mean_squared_error",
                    cv=inner_cv,
                    random_state=random_state + fold_number,
                    n_jobs=1,
                    refit=True,
                    return_train_score=False,
                )
                search.fit(X_outer_train, y_outer_train)
                fitted = search.best_estimator_
                for candidate_index, candidate_params in enumerate(
                    search.cv_results_["params"]
                ):
                    tuning_rows.append(
                        {
                            "stage": "nested_inner_tuning",
                            "model": model_name,
                            "outer_fold": fold_number,
                            "params": candidate_params,
                            "cv_log_rmse_mean": float(
                                -search.cv_results_["mean_test_score"][candidate_index]
                            ),
                            "cv_log_rmse_std": float(
                                search.cv_results_["std_test_score"][candidate_index]
                            ),
                            "rank": int(
                                search.cv_results_["rank_test_score"][candidate_index]
                            ),
                        }
                    )
            else:
                fitted = pipeline.fit(X_outer_train, y_outer_train)

            predictions = fitted.predict(X_outer_valid)
            fold_rmse = float(
                np.sqrt(mean_squared_error(y_outer_valid, predictions))
            )
            fold_rows.append(
                {
                    "model": model_name,
                    "outer_fold": fold_number,
                    "outer_train_size": len(train_idx),
                    "outer_validation_size": len(valid_idx),
                    "cv_log_rmse": fold_rmse,
                    "validation_indices": valid_idx.tolist(),
                }
            )

    fold_frame = pd.DataFrame(fold_rows)
    summary = (
        fold_frame.groupby("model", as_index=False)
        .agg(
            cv_log_rmse_mean=("cv_log_rmse", "mean"),
            cv_log_rmse_std=("cv_log_rmse", "std"),
            cv_folds=("outer_fold", "count"),
        )
        .sort_values("cv_log_rmse_mean")
        .reset_index(drop=True)
    )
    summary["cv_seconds"] = np.nan
    summary["cv_method"] = "nested_kfold"
    summary["inner_splits"] = inner_splits
    summary["search_iterations"] = n_iter
    return summary, tuning_rows, fold_rows
