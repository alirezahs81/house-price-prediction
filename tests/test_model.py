import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.modeling import make_pipeline
from src.preprocessing import prepare_features, transform_absence_categories

def test_small_model_fit_and_prediction():
    df=pd.DataFrame({"Id":range(1,21),"LotArea":np.arange(20)+100,"OverallQual":[4,5]*10,
                     "Neighborhood":["A","B"]*10,"SalePrice":np.arange(20)*1000+50000})
    X=transform_absence_categories(prepare_features(df)); y=np.log1p(df["SalePrice"])
    model=make_pipeline(Ridge(),X).fit(X,y)
    pred=np.expm1(model.predict(X))
    assert np.isfinite(pred).all() and (pred>0).all()
    assert len(pred)==len(df)
