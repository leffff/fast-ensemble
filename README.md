# fast-ensemble
Library for effecient and convenient high level table model ensembling

---
Usage Example:

```python
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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

X, y = make_regression(n_targets=1)

X_trans = stack.fit_transform(X, y)
```
