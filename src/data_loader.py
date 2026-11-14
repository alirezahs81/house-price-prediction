"""Dataset loading utilities."""

from pathlib import Path

import pandas as pd

from src.config import TARGET, TEST_FILE, TRAIN_FILE


def load_train(path: str | Path = TRAIN_FILE) -> pd.DataFrame:
    """Load training data and verify that the target column exists."""
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Training data must contain target column {TARGET!r}.")
    return frame


def load_test(path: str | Path = TEST_FILE) -> pd.DataFrame:
    """Load competition test features."""
    return pd.read_csv(path)
