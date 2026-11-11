"""Train, compare, select, and persist a house-price regression pipeline."""

import json
import logging
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from src.data_loader import load_train
from src.data_validation import validate_training_data
from src.evaluate import save_diagnostics
from src.exploratory_analysis import run_eda
from src.modeling import candidate_models, make_pipeline
from src.preprocessing import prepare_features, transform_absence_categories

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
LOGGER = logging.getLogger(__name__)


def _search_space(model_name: str) -> dict[str, list[Any]]:
    """Return bounded search spaces for supported estimator families."""
    spaces: dict[str, dict[str, list[Any]]] = {
        "Ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 30.0, 100.0],
        },
        "RandomForest": {
            "model__n_estimators": [250, 400, 600],
            "model__min_samples_leaf": [1, 2, 3, 4],
            "model__max_features": [0.6, 0.75, 0.9, 1.0],
        },
        "ExtraTrees": {
            "model__n_estimators": [250, 400, 600],
            "model__min_samples_leaf": [1, 2, 3, 4],
            "model__max_features": [0.6, 0.75, 0.9, 1.0],
        },
        "GradientBoosting": {
            "model__n_estimators": [150, 250, 350, 500],
            "model__learning_rate": [0.02, 0.03, 0.05, 0.07],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_leaf": [2, 4, 6, 8],
            "model__subsample": [0.7, 0.85, 1.0],
        },
        "HistGradientBoosting": {
            "model__max_iter": [150, 250, 400, 600],
            "model__learning_rate": [0.02, 0.04, 0.06, 0.08],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__l2_regularization": [0.0, 0.5, 1.0, 5.0],
        },
    }
    return spaces.get(model_name, {})


def train() -> dict[str, Any]:
    """Train candidate models, evaluate the winner, and persist the final model."""
    df = load_train()
    validate_training_data(df)

    # EDA is descriptive only; no EDA statistic is used to fit model features.
    run_eda()

    y = np.log1p(df[TARGET].astype(float))
    X = transform_absence_categories(prepare_features(df))
    if TARGET in X.columns or "Id" in X.columns:
        raise ValueError("Target or identifier leakage detected in model features.")

    # Keep a final holdout completely outside model and hyperparameter selection.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    model_cv = RepeatedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=RANDOM_STATE,
    )
    tuning_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows: list[dict[str, Any]] = []

    for name, estimator in candidate_models().items():
        LOGGER.info("Repeated cross-validating %s", name)
        pipeline = make_pipeline(estimator, X_train)
        started = time.time()
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=model_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "cv_log_rmse_mean": float(-scores.mean()),
                "cv_log_rmse_std": float(scores.std()),
                "cv_folds": int(len(scores)),
                "cv_seconds": round(time.time() - started, 2),
            }
        )

    results = pd.DataFrame(rows).sort_values("cv_log_rmse_mean")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(results["model"], results["cv_log_rmse_mean"])
    ax.set(
        xlabel="Repeated 5x2 CV RMSE (log-price)",
        title="Candidate Model Comparison",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    best_name = str(results.iloc[0]["model"])
    best_cv_row = results.iloc[0]
    LOGGER.info("Best cross-validated model family: %s", best_name)
    base_pipeline = make_pipeline(candidate_models()[best_name], X_train)
    params = _search_space(best_name)

    if params:
        search = RandomizedSearchCV(
            base_pipeline,
            params,
            n_iter=8,
            scoring="neg_root_mean_squared_error",
            cv=tuning_cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        selected = search.best_estimator_
        best_params = search.best_params_
    else:
        selected = base_pipeline.fit(X_train, y_train)
        best_params = {}

    train_log_predictions = selected.predict(X_train)
    validation_log_predictions = selected.predict(X_val)
    train_log_rmse = float(
        np.sqrt(mean_squared_error(y_train, train_log_predictions))
    )
    validation_log_rmse = float(
        np.sqrt(mean_squared_error(y_val, validation_log_predictions))
    )

    validation_predictions = np.maximum(
        np.expm1(validation_log_predictions), 1.0
    )
    validation_actual = np.expm1(y_val)
    metrics: dict[str, Any] = save_diagnostics(
        validation_actual,
        validation_predictions,
    )
    metrics.update(
        {
            "selected_model": best_name,
            "best_params": best_params,
            "validation_rows": int(len(y_val)),
            "repeated_cv_folds": 10,
            "cv_log_rmse_mean": float(best_cv_row["cv_log_rmse_mean"]),
            "cv_log_rmse_std": float(best_cv_row["cv_log_rmse_std"]),
            "train_log_rmse": train_log_rmse,
            "validation_log_rmse": validation_log_rmse,
            "log_rmse_generalization_gap": (
                validation_log_rmse - train_log_rmse
            ),
            "target_transform": "log1p; predictions inverse-transformed with expm1",
            "holdout_used_for_model_selection": False,
        }
    )
    (METRICS_DIR / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Refit only after holdout evaluation; the holdout is never used for fitting.
    final_pipeline = selected.fit(X, y)
    try:
        feature_names = final_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        importances = final_pipeline.named_steps["model"].feature_importances_
        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False).head(25)
        importance.to_csv(METRICS_DIR / "feature_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(
            importance["feature"][::-1],
            importance["importance"][::-1],
        )
        ax.set(
            xlabel="Feature importance",
            title="Top 25 Feature Importances",
        )
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)
    except (AttributeError, ValueError):
        LOGGER.warning(
            "Feature importance is not supported by selected estimator."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_FILE)
    LOGGER.info("Saved model to %s", MODEL_FILE)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
