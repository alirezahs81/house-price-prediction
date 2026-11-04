"""Input schema and data-quality validation."""
import pandas as pd

def validate_training_data(frame: pd.DataFrame, target: str = "SalePrice") -> None:
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if target not in frame:
        raise ValueError(f"Required target column {target!r} is missing.")
    if frame[target].isna().any():
        raise ValueError(f"Target column {target!r} contains missing values.")
    if (frame[target] <= 0).any():
        raise ValueError("Target values must be strictly positive for log modeling.")
    if "Id" in frame and frame["Id"].duplicated().any():
        raise ValueError("Duplicate Id values found.")

def validate_prediction_data(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Prediction input is empty.")
    if "SalePrice" in frame:
        raise ValueError("Prediction input must not include SalePrice.")
    if "Id" not in frame:
        raise ValueError("Prediction input must include Id.")
    if frame["Id"].isna().any():
        raise ValueError("Id contains missing values.")
