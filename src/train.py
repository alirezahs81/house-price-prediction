"""Train, compare, select, and persist a house-price regression pipeline."""

import json
import logging
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
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
from src.evaluate import regression_metrics, save_diagnostics
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
            "model__loss": ["huber", "squared_error", "absolute_error"],
            "model__alpha": [0.85, 0.9, 0.95],
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
    tuning_rows: list[dict[str, Any]] = []
    candidate_results: dict[str, dict[str, Any]] = {}

    # Tune each family with the same shuffled five-fold protocol, then
    # compare the selected configurations on identical repeated 5x2 folds.
    # All preprocessing remains inside each cloned pipeline and fold.
    for name, estimator in candidate_models().items():
        LOGGER.info("Tuning and cross-validating %s", name)
        pipeline = make_pipeline(estimator, X_train)
        params = _search_space(name)
        started = time.time()

        if params:
            search = RandomizedSearchCV(
                pipeline,
                params,
                n_iter=8,
                scoring="neg_root_mean_squared_error",
                cv=tuning_cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
                return_train_score=False,
            )
            search.fit(X_train, y_train)
            selected_estimator = search.best_estimator_
            best_params = search.best_params_
            cv_mean = float(-search.best_score_)
            cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

            for index, candidate_params in enumerate(search.cv_results_["params"]):
                tuning_rows.append(
                    {
                        "model": name,
                        "params": candidate_params,
                        "cv_log_rmse_mean": float(
                            -search.cv_results_["mean_test_score"][index]
                        ),
                        "cv_log_rmse_std": float(
                            search.cv_results_["std_test_score"][index]
                        ),
                        "rank": int(search.cv_results_["rank_test_score"][index]),
                    }
                )
        else:
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=tuning_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            selected_estimator = pipeline.fit(X_train, y_train)
            best_params = {}
            cv_mean = float(-scores.mean())
            cv_std = float(scores.std())
            tuning_rows.append(
                {
                    "model": name,
                    "params": {},
                    "cv_log_rmse_mean": cv_mean,
                    "cv_log_rmse_std": cv_std,
                    "rank": 1,
                }
            )

        candidate_results[name] = {
            "estimator": selected_estimator,
            "best_params": best_params,
        }
        tuning_cv_mean = cv_mean
        tuning_cv_std = cv_std
        comparison_scores = cross_val_score(
            selected_estimator,
            X_train,
            y_train,
            cv=model_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        cv_mean = float(-comparison_scores.mean())
        cv_std = float(comparison_scores.std())
        rows.append(
            {
                "model": name,
                "cv_log_rmse_mean": cv_mean,
                "cv_log_rmse_std": cv_std,
                "cv_folds": 10,
                "tuning_cv_log_rmse_mean": tuning_cv_mean,
                "tuning_cv_log_rmse_std": tuning_cv_std,
                "cv_seconds": round(time.time() - started, 2),
            }
        )

    results = pd.DataFrame(rows).sort_values("cv_log_rmse_mean")
    tuning_results = pd.DataFrame(tuning_rows).sort_values(
        "cv_log_rmse_mean"
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    tuning_results.to_csv(REPORTS_DIR / "tuning_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(results["model"], results["cv_log_rmse_mean"])
    ax.set(
        xlabel="Repeated 5x2 CV RMSE (log-price)",
        title="Tuned Candidate Model Comparison",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    best_name = str(results.iloc[0]["model"])
    best_cv_row = results.iloc[0]
    LOGGER.info("Best tuned cross-validated model family: %s", best_name)
    selected = candidate_results[best_name]["estimator"]
    best_params = candidate_results[best_name]["best_params"]

    # The selected model's repeated-CV metrics were calculated for every
    # candidate above, using the same folds and training-only data.
    tuned_cv_log_rmse_mean = float(best_cv_row["cv_log_rmse_mean"])
    tuned_cv_log_rmse_std = float(best_cv_row["cv_log_rmse_std"])

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
    validation_ids = (
        df.loc[X_val.index, "Id"].to_numpy() if "Id" in df.columns else None
    )
    metrics: dict[str, Any] = save_diagnostics(
        validation_actual,
        validation_predictions,
        sample_ids=validation_ids,
    )

    # Compare against a simple training-only median baseline.
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    baseline_predictions = np.maximum(
        np.expm1(baseline.predict(X_val)), 1.0
    )
    baseline_metrics = regression_metrics(validation_actual, baseline_predictions)
    metrics.update(
        {f"baseline_{key}": value for key, value in baseline_metrics.items()}
    )
    metrics["holdout_RMSE_improvement_vs_baseline"] = float(
        1.0 - metrics["RMSE"] / baseline_metrics["RMSE"]
    )
    metrics.update(
        {
            "selected_model": best_name,
            "best_params": best_params,
            "validation_rows": int(len(y_val)),
            "repeated_cv_folds": 10,
            "selected_family_cv_log_rmse_mean": float(
                best_cv_row["cv_log_rmse_mean"]
            ),
            "selected_family_cv_log_rmse_std": float(
                best_cv_row["cv_log_rmse_std"]
            ),
            "in_sample_train_log_rmse": train_log_rmse,
            "validation_log_rmse": validation_log_rmse,
            "tuned_cv_log_rmse_mean": tuned_cv_log_rmse_mean,
            "tuned_cv_log_rmse_std": tuned_cv_log_rmse_std,
            "validation_minus_tuned_cv_log_rmse": (
                validation_log_rmse - tuned_cv_log_rmse_mean
            ),
            "target_transform": "log1p; predictions inverse-transformed with expm1",
            "holdout_used_for_model_selection": False,
            "validation_error_reports": [
                "validation_residuals.csv",
                "worst_validation_errors.csv",
                "validation_error_by_price_band.csv",
                "model_comparison.csv",
                "tuning_comparison.csv",
            ],
            "cv_metric_note": (
                "Each supported family receives the same eight-iteration "
                "randomized search budget using shuffled five-fold CV. The "
                "selected configuration for every family is then compared on "
                "the same repeated five-fold CV (five folds repeated twice), "
                "using training-partition rows only. The untouched final holdout "
                "is excluded from all fitting, tuning, and model selection."
            ),
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
