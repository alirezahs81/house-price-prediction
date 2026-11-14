"""Deterministic, target-independent housing feature engineering."""

import pandas as pd


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with useful derived features safe for train and test."""
    data = frame.copy()

    def numeric(name: str) -> pd.Series:
        if name in data.columns:
            return pd.to_numeric(data[name], errors="coerce")
        return pd.Series(0.0, index=data.index, dtype="float64")

    data["TotalSF"] = (
        numeric("TotalBsmtSF").fillna(0)
        + numeric("1stFlrSF").fillna(0)
        + numeric("2ndFlrSF").fillna(0)
    )
    data["TotalBathrooms"] = (
        numeric("FullBath").fillna(0)
        + 0.5 * numeric("HalfBath").fillna(0)
        + numeric("BsmtFullBath").fillna(0)
        + 0.5 * numeric("BsmtHalfBath").fillna(0)
    )

    if "YrSold" in data.columns and "YearBuilt" in data.columns:
        data["HouseAgeAtSale"] = (
            numeric("YrSold") - numeric("YearBuilt")
        ).clip(lower=0)
    if "YrSold" in data.columns and "YearRemodAdd" in data.columns:
        data["YearsSinceRemodel"] = (
            numeric("YrSold") - numeric("YearRemodAdd")
        ).clip(lower=0)

    data["HasGarage"] = (numeric("GarageArea").fillna(0) > 0).astype(int)
    data["QualityTimesLivingArea"] = (
        numeric("OverallQual").fillna(0) * numeric("GrLivArea").fillna(0)
    )
    return data
