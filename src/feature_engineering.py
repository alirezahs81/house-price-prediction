"""Deterministic, target-independent housing feature engineering."""
import pandas as pd

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with useful derived features; safe for train and test."""
    data = frame.copy()
    def numeric(name):
        return pd.to_numeric(data[name], errors="coerce") if name in data else pd.Series(0, index=data.index, dtype=float)
    data["TotalSF"] = numeric("TotalBsmtSF").fillna(0) + numeric("1stFlrSF").fillna(0) + numeric("2ndFlrSF").fillna(0)
    data["TotalBathrooms"] = (numeric("FullBath").fillna(0) + 0.5 * numeric("HalfBath").fillna(0)
                              + numeric("BsmtFullBath").fillna(0) + 0.5 * numeric("BsmtHalfBath").fillna(0))
    if "YrSold" in data and "YearBuilt" in data:
        data["HouseAgeAtSale"] = (numeric("YrSold") - numeric("YearBuilt")).clip(lower=0)
    if "YrSold" in data and "YearRemodAdd" in data:
        data["YearsSinceRemodel"] = (numeric("YrSold") - numeric("YearRemodAdd")).clip(lower=0)
    data["HasGarage"] = (numeric("GarageArea").fillna(0) > 0).astype(int)
    data["QualityTimesLivingArea"] = numeric("OverallQual").fillna(0) * numeric("GrLivArea").fillna(0)
    return data
