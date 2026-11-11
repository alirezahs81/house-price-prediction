"""Input schema and data-quality validation."""

import numpy as np
import pandas as pd


def validate_training_data(
    frame: pd.DataFrame, target: str = "SalePrice"
) -> None:
    """Raise ValueError when the training frame violates its expected schema."""
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if target not in frame:
        raise ValueError(f"Required target column {target!r} is missing.")

    target_values = pd.to_numeric(frame[target], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(target_values).all():
        raise ValueError(f"Target column {target!r} must contain finite numbers.")
    if (target_values <= 0).any():
        raise ValueError("Target values must be strictly positive for log modeling.")

    if "Id" in frame:
        if frame["Id"].isna().any():
            raise ValueError("Id contains missing values.")
        if frame["Id"].duplicated().any():
            raise ValueError("Duplicate Id values found.")


def validate_prediction_data(frame: pd.DataFrame) -> None:
    """Raise ValueError when prediction input violates its expected schema."""
    if frame.empty:
        raise ValueError("Prediction input is empty.")
    if "SalePrice" in frame:
        raise ValueError("Prediction input must not include SalePrice.")
    if "Id" not in frame:
        raise ValueError("Prediction input must include Id.")
    if frame["Id"].isna().any():
        raise ValueError("Id contains missing values.")
    if frame["Id"].duplicated().any():
        raise ValueError("Prediction input contains duplicate Id values.")
