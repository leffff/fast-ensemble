from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from fast_ensemble.stacking import StackingTransformer
from fast_ensemble.wrappers import (
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    LGBMClassifierWrapper,
    LGBMRegressorWrapper,
    XGBClassifierWrapper,
    XGBRegressorWrapper,
)

stack = StackingTransformer(
    models=[
        (
            "catboost",
            CatBoostRegressorWrapper(CatBoostRegressor(verbose=0), use_best_model=True),
        ),
        ("xgboost", XGBRegressorWrapper(XGBRegressor(), use_best_model=True)),
        ("lgmb", LGBMRegressorWrapper(LGBMRegressor(), use_best_model=True)),
        ("boosting", GradientBoostingRegressor()),
    ],
    main_metric=mean_squared_error,
    regression=True,
)

X, y = make_regression(n_targets=1)
print(y)

stack.fit_transform(X, y).to_csv("wrapper_output_3.csv")
