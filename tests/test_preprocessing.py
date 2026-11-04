import pandas as pd
from src.preprocessing import prepare_features, transform_absence_categories, build_preprocessor

def test_missing_category_semantics():
    frame=pd.DataFrame({"Id":[1,2], "SalePrice":[100000,120000], "PoolQC":[None,"Ex"], "LotArea":[100,200]})
    X=transform_absence_categories(prepare_features(frame))
    assert X["PoolQC"].iloc[0]=="None"
    assert "SalePrice" not in X and "Id" not in X

def test_preprocessor_handles_unseen_categories():
    X=pd.DataFrame({"num":[1.0,None,3.0],"cat":["a",None,"b"]})
    pre=build_preprocessor(X)
    out=pre.fit_transform(X)
    assert out.shape[0]==3
