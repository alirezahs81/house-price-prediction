"""Public package API for the House Price Prediction project."""

__version__ = "1.0.0"
__author__ = "House Price Prediction Team"
__email__ = "your-email@example.com"
__description__ = "Reproducible Ames Housing regression project"
__license__ = "MIT"

from src.data_loader import load_test, load_train
from src.data_validation import validate_prediction_data, validate_training_data
from src.evaluate import regression_metrics, save_diagnostics
from src.exploratory_analysis import run_eda
from src.feature_engineering import engineer_features
from src.modeling import candidate_models, make_pipeline
from src.predict import predict
from src.preprocessing import (
    build_preprocessor,
    prepare_features,
    transform_absence_categories,
)
from src.train import train

__all__ = [
    "load_train", "load_test", "validate_training_data",
    "validate_prediction_data", "prepare_features",
    "transform_absence_categories", "build_preprocessor",
    "engineer_features", "candidate_models", "make_pipeline", "train",
    "regression_metrics", "save_diagnostics", "predict", "run_eda",
]
