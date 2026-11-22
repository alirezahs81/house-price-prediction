"""Train, compare, select, and persist a house-price regression pipeline."""

import json
import logging
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
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
from src.nested_cv import nested_cv_model_comparison
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
            "model__min_samples_leaf": [2, 4, 6, 10],
            "model__subsample": [0.65, 0.8, 0.95],
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
    # Nested CV evaluates each model family with its tuning contained inside
    # the outer training fold. The final holdout remains untouched.
    LOGGER.info("Running nested CV for model-family comparison")
    nested_results, nested_tuning_rows, outer_fold_rows = (
        nested_cv_model_comparison(
            X_train=X_train,
            y_train=y_train,
            models=candidate_models(),
            search_spaces={name: _search_space(name) for name in candidate_models()},
            pipeline_factory=make_pipeline,
            outer_splits=5,
            inner_splits=3,
            n_iter=3,
            random_state=RANDOM_STATE,
        )
    )
    rows = nested_results.to_dict(orient="records")
    tuning_rows: list[dict[str, Any]] = list(nested_tuning_rows)
    candidate_results: dict[str, dict[str, Any]] = {}
    tuning_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # After nested evaluation, tune each family on the full training partition
    # to produce final candidate pipelines. Family ranking comes only from
    # the outer-fold scores, not these post-selection tuning scores.
    for name, estimator in candidate_models().items():
        LOGGER.info("Final training-partition tuning for %s", name)
        pipeline = make_pipeline(estimator, X_train)
        params = _search_space(name)

        if params:
            search = RandomizedSearchCV(
                pipeline,
                params,
                n_iter=min(8, len(list(ParameterGrid(params)))),
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
            final_tuning_mean = float(-search.best_score_)
            final_tuning_std = float(
                search.cv_results_["std_test_score"][search.best_index_]
            )
            for index, candidate_params in enumerate(search.cv_results_["params"]):
                tuning_rows.append(
                    {
                        "stage": "final_training_tuning",
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
            selected_estimator = pipeline.fit(X_train, y_train)
            best_params = {}
            family_row = nested_results.loc[nested_results["model"] == name].iloc[0]
            final_tuning_mean = float(family_row["cv_log_rmse_mean"])
            final_tuning_std = float(family_row["cv_log_rmse_std"])
            tuning_rows.append(
                {
                    "stage": "final_training_tuning",
                    "model": name,
                    "params": {},
                    "cv_log_rmse_mean": final_tuning_mean,
                    "cv_log_rmse_std": final_tuning_std,
                    "rank": 1,
                }
            )

        candidate_results[name] = {
            "estimator": selected_estimator,
            "best_params": best_params,
            "final_tuning_cv_log_rmse_mean": final_tuning_mean,
            "final_tuning_cv_log_rmse_std": final_tuning_std,
        }
    outer_fold_results = pd.DataFrame(outer_fold_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outer_fold_results.to_csv(
        REPORTS_DIR / "nested_cv_outer_fold_results.csv", index=False
    )

    results = pd.DataFrame(rows).sort_values("cv_log_rmse_mean")
    tuning_results = pd.DataFrame(tuning_rows).sort_values(
        "cv_log_rmse_mean"
    )
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    tuning_results.to_csv(REPORTS_DIR / "tuning_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(results["model"], results["cv_log_rmse_mean"])
    ax.set(
        xlabel="Nested 5-fold CV RMSE (log-price)",
        title="Nested-CV Model Family Comparison",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    best_name = str(results.iloc[0]["model"])
    best_cv_row = results.iloc[0]
    LOGGER.info("Best nested-CV model family: %s", best_name)
    selected = candidate_results[best_name]["estimator"]
    best_params = candidate_results[best_name]["best_params"]

    # Nested outer-fold scores estimate the full model-selection procedure.
    # Final tuning CV scores are reported separately and are not used to rank
    # model families.
    tuned_cv_log_rmse_mean = float(
        candidate_results[best_name]["final_tuning_cv_log_rmse_mean"]
    )
    tuned_cv_log_rmse_std = float(
        candidate_results[best_name]["final_tuning_cv_log_rmse_std"]
    )

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
            "nested_cv_outer_folds": 5,
            "nested_cv_inner_folds": 3,
            "nested_cv_search_iterations_per_family": 3,
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
                "nested_cv_outer_fold_results.csv",
            ],
            "cv_metric_note": (
                "Model families are compared using nested five-fold CV. In each "
                "outer fold, hyperparameter tuning uses only the outer training "
                "partition with three-fold CV and a three-iteration randomized "
                "search. Outer validation folds are never used for tuning. After "
                "nested evaluation, each family is tuned on the full training "
                "partition for final candidate pipelines; those tuning scores "
                "are reported separately and are not used for family ranking. "
                "The untouched final holdout is excluded from all fitting, "
                "tuning, and model selection."
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
