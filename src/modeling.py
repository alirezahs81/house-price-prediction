"""Candidate estimators and model evaluation helpers."""

from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def candidate_models() -> dict[str, Any]:
    """Return deterministic baseline and nonlinear regression estimators."""
    return {
        "DummyMedian": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=20.0),
        "DecisionTree": DecisionTreeRegressor(
            min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=2,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(estimator: Any, X: pd.DataFrame) -> Pipeline:
    """Create preprocessing and estimator as one persisted pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X)),
            ("model", estimator),
        ]
    )
