import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import prepare_features


def test_engineered_features():
    df = pd.DataFrame(
        {
            "TotalBsmtSF": [1000],
            "1stFlrSF": [800],
            "2ndFlrSF": [200],
            "FullBath": [2],
            "HalfBath": [1],
            "BsmtFullBath": [1],
            "BsmtHalfBath": [0],
            "YearBuilt": [2000],
            "YearRemodAdd": [2010],
            "YrSold": [2020],
            "GarageArea": [400],
            "OverallQual": [7],
            "GrLivArea": [1000],
        }
    )
    out = engineer_features(df)
    assert out["TotalSF"].iloc[0] == 2000
    assert out["TotalBathrooms"].iloc[0] == 3.5
    assert out["HouseAgeAtSale"].iloc[0] == 20
    assert out["HasGarage"].iloc[0] == 1


def test_target_and_identifier_never_enter_model_features():
    frame = pd.DataFrame(
        {
            "Id": [1, 2],
            "LotArea": [8000, 9000],
            "SalePrice": [100000, 900000],
        }
    )
    changed_target = frame.assign(SalePrice=[9999999, 1])
    features = prepare_features(frame)
    changed_features = prepare_features(changed_target)

    assert "SalePrice" not in features.columns
    assert "Id" not in features.columns
    pd.testing.assert_frame_equal(features, changed_features)


def test_engineered_features_are_independent_of_target_and_identifier():
    frame = pd.DataFrame(
        {
            "Id": [10, 20],
            "SalePrice": [100_000, 200_000],
            "OverallQual": [5, 8],
            "GrLivArea": [1200, 1800],
            "GarageArea": [0, 500],
        }
    )
    changed = frame.assign(
        Id=[999, 888],
        SalePrice=[9_000_000, 1],
    )

    original_features = prepare_features(frame)
    changed_features = prepare_features(changed)

    pd.testing.assert_frame_equal(original_features, changed_features)
    assert "Id" not in original_features.columns
    assert "SalePrice" not in original_features.columns
