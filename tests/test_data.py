import numpy as np
import pandas as pd
import pytest

from src.config import TEST_FILE, TRAIN_FILE
from src.data_loader import load_test, load_train
from src.data_validation import validate_prediction_data, validate_training_data


def test_dataset_files_load():
    assert TRAIN_FILE.exists() and TEST_FILE.exists()
    assert len(load_train()) == 1460
    assert len(load_test()) == 1459


def test_validation_rejects_missing_target():
    with pytest.raises(ValueError):
        validate_training_data(pd.DataFrame({"Id": [1]}))


def test_training_validation_rejects_invalid_targets_and_duplicate_ids():
    with pytest.raises(ValueError, match="finite"):
        validate_training_data(
            pd.DataFrame({"Id": [1, 2], "SalePrice": [100000, np.inf]})
        )
    with pytest.raises(ValueError, match="Duplicate Id"):
        validate_training_data(
            pd.DataFrame({"Id": [1, 1], "SalePrice": [100000, 120000]})
        )


def test_prediction_schema():
    validate_prediction_data(pd.DataFrame({"Id": [1], "LotArea": [8000]}))
    with pytest.raises(ValueError):
        validate_prediction_data(pd.DataFrame({"LotArea": [8000]}))
