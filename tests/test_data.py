import pandas as pd
import pytest
from src.config import TRAIN_FILE, TEST_FILE
from src.data_loader import load_train, load_test
from src.data_validation import validate_training_data, validate_prediction_data

def test_dataset_files_load():
    assert TRAIN_FILE.exists() and TEST_FILE.exists()
    assert len(load_train()) == 1460
    assert len(load_test()) == 1459

def test_validation_rejects_missing_target():
    with pytest.raises(ValueError):
        validate_training_data(pd.DataFrame({"Id":[1]}))

def test_prediction_schema():
    validate_prediction_data(pd.DataFrame({"Id":[1], "LotArea":[8000]}))
    with pytest.raises(ValueError):
        validate_prediction_data(pd.DataFrame({"LotArea":[8000]}))
