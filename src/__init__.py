"""
House Price Prediction - A reproducible regression project for Ames Housing dataset.

This package provides:
- Data loading and validation
- Exploratory data analysis
- Feature engineering
- Leakage-safe preprocessing
- Multiple model comparison and selection
- Cross-validation and hyperparameter tuning
- Model persistence and prediction CLI
"""

__version__ = "1.0.0"
__author__ = "House Price Prediction Team"
__email__ = "your-email@example.com"
__description__ = "Reproducible Ames Housing regression project"
__license__ = "MIT"

# Public API
from src.data_loader import load_train, load_test
from src.data_validation import validate_training_data, validate_prediction_data
from src.preprocessing import prepare_features, transform_absence_categories, build_preprocessor
from src.feature_engineering import engineer_features
from src.modeling import candidate_models, make_pipeline
from src.train import train
from src.evaluate import regression_metrics, save_diagnostics
from src.predict import predict
from src.exploratory_analysis import run_eda

__all__ = [
    "load_train",
    "load_test",
    "validate_training_data",
    "validate_prediction_data",
    "prepare_features",
    "transform_absence_categories",
    "build_preprocessor",
    "engineer_features",
    "candidate_models",
    "make_pipeline",
    "train",
    "regression_metrics",
    "save_diagnostics",
    "predict",
    "run_eda",
]