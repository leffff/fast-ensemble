import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from fast_ensemble.stacking import StackingTransformer


def metric(target, preds):
    return accuracy_score(target, np.argmax(preds, 1))


stack = StackingTransformer(
    models=[
        ("catboost", CatBoostClassifier(verbose=0)),
        ("xgboost", XGBClassifier()),
        ("lgmb", LGBMClassifier()),
        ("boosting", GradientBoostingClassifier()),
    ],
    regression=False,
    main_metric=metric,
)

X, y = load_iris(return_X_y=True)

stack.fit_transform(X, y)
