"""Dataset loading utilities."""
import pandas as pd
from src.config import TRAIN_FILE, TEST_FILE, TARGET

def load_train(path=TRAIN_FILE):
    """Load training data and verify the target exists."""
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Training data must contain target column {TARGET!r}.")
    return frame

def load_test(path=TEST_FILE):
    """Load competition test features."""
    return pd.read_csv(path)
