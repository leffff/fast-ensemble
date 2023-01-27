[![PyPI version](https://img.shields.io/pypi/v/fast-ensemble.svg?colorB=4cc61e)](https://pypi.org/project/fast-ensemble/) 
[![PyPI license](https://img.shields.io/pypi/l/fast-ensemble.svg)](https://github.com/leffff/fast-ensemble/blob/main/LICENSE)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/fast-ensemble.svg)](https://pypi.python.org/pypi/fast-ensemble/)

# fast-ensemble
Scikit-learn-style library for effecient and convenient high level table model ensembling

---
## Usage Example:

Initialize Stack
```python
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from fast_ensemble import StackingTransformer
from fast_ensemble import (
    CatBoostRegressorWrapper,
    LGBMRegressorWrapper,
    XGBRegressorWrapper,
)

stack_1 = StackingTransformer(
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
    stratified=True,
    stratification_bins=7
)
```
And another one

```python
stack_2 = StackingTransformer(
    models=[
        ("Dummy Regressor", DummyRegressor()),
    ],
    main_metric=mean_squared_error,
    regression=True,
    n_folds=5,
    random_state=None,
    shuffle=False,
    verbose=True,
    stratified=True,
    stratification_bins=7
)
```
Train your stacks (and get transformed dataframes)
```python
X, y = make_regression(n_targets=1)

X_1_trans = stack_1.fit_transform(X, y)
X_2_trans = stack_2.fit_transform(X, y)
```
Want to merge 2 stacks for convenience? Here you go!
```python
stack_1.merge(stack_2)

stack_1.get_scores(prettified=True)

       catboost       xgboost          lgmb      boosting  Dummy Regressor
0   9852.055535  23389.781003   8872.055479  13130.504063    21344.359900
1  14259.407424  20177.587908  12040.548492  14088.529604    28620.260635
2  16393.254421  24267.409682   9503.011118  15067.349045    33377.287468
3  12694.791124  16349.931831   7188.301326  10675.853608    29510.019041
4  17505.264716  12158.834533  10273.547605   9621.041119    39099.670810
```




