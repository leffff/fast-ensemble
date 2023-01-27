from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor

from fast_ensemble.stacking import StackingTransformer
from fast_ensemble.wrappers import (
    CatBoostRegressorWrapper,
    LGBMRegressorWrapper,
    XGBRegressorWrapper,
)

stack = StackingTransformer(
    models=[
        (
            "catboost",
            CatBoostRegressorWrapper(
                CatBoostRegressor(verbose=0),
                use_best_model=True,
                early_stopping_rounds=100,
            ),
        ),
        (
            "xgboost",
            XGBRegressorWrapper(
                XGBRegressor(), use_best_model=True, early_stopping_rounds=100
            ),
        ),
        (
            "lgmb",
            LGBMRegressorWrapper(
                LGBMRegressor(), use_best_model=True, early_stopping_rounds=100
            ),
        ),
        ("boosting", GradientBoostingRegressor()),
    ],
    main_metric=mean_squared_error,
    regression=True,
    n_folds=5,
    random_state=None,
    shuffle=False,
    verbose=True,
)

stack_2 = StackingTransformer(
    models=[
        ("Dumb Regressor", DummyRegressor()),
    ],
    main_metric=mean_squared_error,
    regression=True,
    n_folds=5,
    random_state=None,
    shuffle=False,
    verbose=True,
)

X, y = make_regression(n_targets=1)

stack.fit_transform(X, y)
stack_2.fit_transform(X, y)

stack.merge(stack_2)

print(stack.get_models())

print(stack.get_scores(prettified=True))

print(stack.models)

print(stack.names)

print(stack.n_models)

print(stack.n_folds)