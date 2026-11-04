"""Tests for prediction module."""
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

from src.predict import predict
from src.train import train
from src.config import MODEL_FILE, TEST_FILE, PREDICTION_FILE


def test_predict_creates_valid_output(tmp_path):
    """Test that predict creates valid CSV output with Id and SalePrice."""
    # First train a minimal model
    from src.config import TRAIN_FILE
    from src.data_loader import load_train
    from src.data_validation import validate_training_data
    from src.preprocessing import prepare_features, transform_absence_categories
    from src.modeling import make_pipeline
    from sklearn.linear_model import Ridge
    import numpy as np
    
    df = load_train()
    validate_training_data(df)
    
    y = np.log1p(df["SalePrice"].astype(float))
    X = transform_absence_categories(prepare_features(df))
    
    model = make_pipeline(Ridge(alpha=1.0), X).fit(X, y)
    
    # Save model to temp location
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    
    # Create test input
    test_df = df.head(10)[["Id"] + [c for c in df.columns if c not in ["SalePrice"]]]
    input_path = tmp_path / "test_input.csv"
    test_df.to_csv(input_path, index=False)
    
    # Run prediction
    output_path = tmp_path / "predictions.csv"
    result = predict(input_path=input_path, output_path=output_path, model_path=model_path)
    
    # Validate output
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Id", "SalePrice"]
    assert len(result) == len(test_df)
    assert result["SalePrice"].notna().all()
    assert (result["SalePrice"] > 0).all()
    assert np.isfinite(result["SalePrice"]).all()


def test_predict_validates_input():
    """Test that predict validates input data correctly."""
    from src.data_validation import validate_prediction_data
    
    # Valid input
    valid_df = pd.DataFrame({"Id": [1, 2], "LotArea": [10000, 12000]})
    validate_prediction_data(valid_df)  # Should not raise
    
    # Missing Id
    with pytest.raises(ValueError, match="must include Id"):
        validate_prediction_data(pd.DataFrame({"LotArea": [10000]}))
    
    # Contains SalePrice
    with pytest.raises(ValueError, match="must not include SalePrice"):
        validate_prediction_data(pd.DataFrame({"Id": [1], "SalePrice": [100000]}))
    
    # Empty dataframe
    with pytest.raises(ValueError, match="empty"):
        validate_prediction_data(pd.DataFrame())


def test_predict_handles_missing_model():
    """Test that predict fails gracefully with missing model."""
    with pytest.raises(FileNotFoundError):
        predict(model_path="/nonexistent/model.joblib")


def test_prediction_output_format():
    """Test prediction output matches Kaggle submission format."""
    # Load existing predictions if available
    if PREDICTION_FILE.exists():
        preds = pd.read_csv(PREDICTION_FILE)
        assert list(preds.columns) == ["Id", "SalePrice"]
        assert preds["Id"].notna().all()
        assert preds["SalePrice"].notna().all()
        assert (preds["SalePrice"] > 0).all()