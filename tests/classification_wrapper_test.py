import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from fast_ensemble.stacking import StackingTransformer
from fast_ensemble.wrappers import (
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    LGBMClassifierWrapper,
    LGBMRegressorWrapper,
    XGBClassifierWrapper,
    XGBRegressorWrapper,
)


def metric(target, preds):

    return accuracy_score(target, np.argmax(preds, 1))


stack = StackingTransformer(
    models=[
        (
            "catboost",
            CatBoostClassifierWrapper(
                CatBoostClassifier(verbose=0), use_best_model=True, early_stopping_rounds=100
            ),
        ),
        ("xgboost", XGBClassifierWrapper(XGBClassifier(), use_best_model=True, early_stopping_rounds=100)),
        ("lgmb", LGBMClassifierWrapper(LGBMClassifier(), use_best_model=True, early_stopping_rounds=100)),
        ("boosting", GradientBoostingClassifier()),
    ],
    main_metric=metric,
    regression=False,
)

X, y = load_iris(return_X_y=True)
print(y)

stack.fit_transform(X, y).to_csv("wrapper_output_4.csv")
