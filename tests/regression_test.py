from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from fast_ensemble.stacking import StackingTransformer

stack = StackingTransformer(
    models=[
        ("catboost", CatBoostRegressor(verbose=0)),
        ("xgboost", XGBRegressor()),
        ("lgmb", LGBMRegressor()),
        ("boosting", GradientBoostingRegressor()),
    ],
    main_metric=mean_squared_error,
    regression=True,
)

X, y = make_regression(n_targets=1)
print(y)

print(stack.fit_transform(X, y))
