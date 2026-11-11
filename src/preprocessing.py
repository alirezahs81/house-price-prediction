"""Leakage-safe scikit-learn preprocessing pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import engineer_features

ABSENCE_CATEGORIES = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and remove identifiers and target from model inputs."""
    data = engineer_features(frame)
    return data.drop(columns=["SalePrice", "Id"], errors="ignore")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a transformer inferred from the supplied training feature frame."""
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_absence_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Represent semantically absent Ames Housing categories consistently."""
    data = frame.copy()
    for column in ABSENCE_CATEGORIES:
        if column in data.columns:
            data[column] = data[column].astype("object").fillna("None")
    return data
