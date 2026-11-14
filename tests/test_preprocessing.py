import numpy as np
import pandas as pd

from src.preprocessing import (
    build_preprocessor,
    prepare_features,
    transform_absence_categories,
)


def test_missing_category_semantics():
    frame = pd.DataFrame(
        {
            "Id": [1, 2],
            "SalePrice": [100000, 120000],
            "PoolQC": [None, "Ex"],
            "LotArea": [100, 200],
        }
    )
    X = transform_absence_categories(prepare_features(frame))
    assert X["PoolQC"].iloc[0] == "None"
    assert "SalePrice" not in X
    assert "Id" not in X


def test_preprocessor_handles_unseen_categories():
    X_train = pd.DataFrame(
        {"num": [1.0, None, 3.0], "cat": ["a", None, "b"]}
    )
    X_validation = pd.DataFrame({"num": [10000.0], "cat": ["unseen"]})
    preprocessor = build_preprocessor(X_train)
    train_transformed = preprocessor.fit_transform(X_train)
    validation_transformed = preprocessor.transform(X_validation)

    assert train_transformed.shape[0] == 3
    assert validation_transformed.shape[0] == 1
    assert np.isfinite(validation_transformed).all()

    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps[
        "imputer"
    ]
    assert numeric_imputer.statistics_[0] == 2.0



def test_target_and_identifier_do_not_change_model_features():
    frame = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "SalePrice": [100000, 150000, 200000],
            "LotArea": [5000, 7000, 9000],
            "OverallQual": [4, 6, 8],
        }
    )
    changed_target_and_ids = frame.assign(
        Id=[101, 102, 103],
        SalePrice=[900000, 800000, 700000],
    )

    original_features = prepare_features(frame)
    changed_features = prepare_features(changed_target_and_ids)

    pd.testing.assert_frame_equal(original_features, changed_features)
    assert "SalePrice" not in original_features
    assert "Id" not in original_features
