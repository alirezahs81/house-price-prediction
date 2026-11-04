import pandas as pd
from src.feature_engineering import engineer_features

def test_engineered_features():
    df=pd.DataFrame({"TotalBsmtSF":[1000],"1stFlrSF":[800],"2ndFlrSF":[200],
        "FullBath":[2],"HalfBath":[1],"BsmtFullBath":[1],"BsmtHalfBath":[0],
        "YearBuilt":[2000],"YearRemodAdd":[2010],"YrSold":[2020],
        "GarageArea":[400],"OverallQual":[7],"GrLivArea":[1000]})
    out=engineer_features(df)
    assert out["TotalSF"].iloc[0]==2000
    assert out["TotalBathrooms"].iloc[0]==3.5
    assert out["HouseAgeAtSale"].iloc[0]==20
    assert out["HasGarage"].iloc[0]==1
