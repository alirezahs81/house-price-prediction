"""Train, compare, select, and persist a house-price regression pipeline."""
import json, logging, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.config import TRAIN_FILE, MODEL_FILE, MODELS_DIR, REPORTS_DIR, METRICS_DIR, RANDOM_STATE, TARGET
from src.data_loader import load_train
from src.data_validation import validate_training_data
from src.preprocessing import prepare_features, transform_absence_categories, build_preprocessor
from src.modeling import candidate_models, make_pipeline
from src.evaluate import regression_metrics, save_diagnostics
from src.exploratory_analysis import run_eda

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

def train():
    df = load_train(); validate_training_data(df)
    run_eda()
    y = np.log1p(df[TARGET].astype(float))
    X = transform_absence_categories(prepare_features(df))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows=[]; fitted={}
    for name, estimator in candidate_models().items():
        LOGGER.info("Cross-validating %s", name)
        pipe = make_pipeline(estimator, X_train)
        started=time.time()
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
        rows.append({"model":name, "cv_log_rmse_mean":float(-scores.mean()), "cv_log_rmse_std":float(scores.std()), "cv_seconds":round(time.time()-started,2)})
        fitted[name]=pipe
    results=pd.DataFrame(rows).sort_values("cv_log_rmse_mean")
    REPORTS_DIR.mkdir(exist_ok=True); METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR/"model_comparison.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(results["model"], results["cv_log_rmse_mean"])
    ax.set(xlabel="5-fold CV RMSE (log-price)", title="Candidate Model Comparison")
    ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(__import__("src.config", fromlist=["FIGURES_DIR"]).FIGURES_DIR/"model_comparison.png", dpi=150)
    plt.close(fig)
    best_name=str(results.iloc[0]["model"])
    LOGGER.info("Best CV model: %s", best_name)
    # Modest randomized search for a tree ensemble, selected using CV only.
    estimator = candidate_models()[best_name]
    base_pipe=make_pipeline(estimator, X_train)
    params={}
    if best_name == "ExtraTrees":
        params={"model__n_estimators":[250,400], "model__min_samples_leaf":[1,2,3], "model__max_features":[0.7,0.9,1.0]}
    elif best_name == "RandomForest":
        params={"model__n_estimators":[200,350], "model__min_samples_leaf":[1,2,3], "model__max_features":[0.7,0.9]}
    elif best_name == "GradientBoosting":
        params={"model__n_estimators":[150,250,350], "model__learning_rate":[0.03,0.05,0.07], "model__max_depth":[2,3]}
    if params:
        search=RandomizedSearchCV(base_pipe, params, n_iter=min(3, int(np.prod([len(v) for v in params.values()]))),
            scoring="neg_root_mean_squared_error", cv=cv, random_state=RANDOM_STATE, n_jobs=1, refit=True)
        search.fit(X_train,y_train); selected=search.best_estimator_; best_params=search.best_params_
    else:
        selected=base_pipe.fit(X_train,y_train); best_params={}
    val_pred=np.expm1(selected.predict(X_val))
    y_val_original=np.expm1(y_val)
    metrics=save_diagnostics(y_val_original,val_pred)
    metrics.update({"selected_model":best_name, "best_params":best_params, "validation_rows":int(len(y_val))})
    (METRICS_DIR/"validation_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    # Refit the selected hyperparameter configuration on all labeled data.
    final_pipe=selected
    final_pipe.fit(X,y)
    try:
        names = final_pipe.named_steps["preprocessor"].get_feature_names_out()
        importances = final_pipe.named_steps["model"].feature_importances_
        importance = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False).head(25)
        importance.to_csv(REPORTS_DIR/"metrics"/"feature_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(importance["feature"][::-1], importance["importance"][::-1])
        ax.set(xlabel="Feature importance", title="Top 25 Feature Importances")
        fig.tight_layout(); fig.savefig(__import__("src.config", fromlist=["FIGURES_DIR"]).FIGURES_DIR/"feature_importance.png", dpi=150)
        plt.close(fig)
    except (AttributeError, ValueError):
        LOGGER.warning("Feature importance is not supported by selected estimator.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True); joblib.dump(final_pipe, MODEL_FILE)
    LOGGER.info("Saved model to %s", MODEL_FILE)
    return metrics

if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
