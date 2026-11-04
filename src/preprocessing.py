"""Leakage-safe sklearn preprocessing pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.feature_engineering import engineer_features

# These missing category values usually mean the feature is absent in Ames Housing.
ABSENCE_CATEGORIES = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
 "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
 "PoolQC", "Fence", "MiscFeature", "MasVnrType"]

def prepare_features(frame):
    """Engineer features and remove identifiers/target from model inputs."""
    data = engineer_features(frame)
    return data.drop(columns=["SalePrice", "Id"], errors="ignore")

def build_preprocessor(X):
    """Build a transformer inferred from the supplied training feature frame.
    
    Assumes absence categories have already been handled by transform_absence_categories().
    """
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                             ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric),
                              ("categorical", categorical_pipe, categorical)],
                             remainder="drop", verbose_feature_names_out=False)

def transform_absence_categories(frame):
    """Apply consistent semantic missing-category handling before the pipeline."""
    data = frame.copy()
    for col in ABSENCE_CATEGORIES:
        if col in data.columns:
            data[col] = data[col].fillna("None")
    return data
