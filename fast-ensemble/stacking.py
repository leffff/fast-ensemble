from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from errors import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold
from utils import average_preds, to_pandas


class WaveStackingTransformer:
    """
    Stacking ensembling model
    """

    def __init__(
        self,
        models,
        metric,
        n_folds: int = 4,
        random_state: int = None,
        shuffle: bool = False,
        verbose: bool = True,
        regression: bool = True,
    ):

        self.models = [i[1] for i in models]
        self.n_models = len(self.models)
        self.names = [i[0] for i in models]
        self.model_dict = {}
        self.model_scores_dict = {}

        self.n_folds = n_folds
        self.metric = metric
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.regression = regression

        self.fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.array, list],
        y: Union[pd.DataFrame, pd.Series, np.array, list],
    ):

        self.model_dict = {}
        self.model_scores_dict = {}

        X, y = to_pandas(X), to_pandas(y)

        self.n_labels = len(np.unique(y))

        kf = KFold(
            n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle
        )

        # Form each model
        for model_i in range(self.n_models):
            # For each fold
            sub_models = []
            sub_scores = []

            for fold, (train_index, test_index) in enumerate(kf.split(X)):
                sub_model = self.models[fold]

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                sub_model.fit(X_train, y_train)
                sub_models.append(sub_model)

                if self.regression:
                    sub_score = self.metric(sub_model.predict(X_test), y_test)
                else:
                    sub_score = self.metric(sub_model.predict_proba(X_test), y_test)

                sub_scores.append(sub_score)

                if self.verbose:
                    print(f"Fold: {fold}, Score: {sub_score}")

            self.model_dict[self.names[model_i]] = sub_models
            self.model_scores_dict[self.names[model_i]] = sub_scores

        self.fitted = True
        return self

    def transform(
        self, X: Union[pd.DataFrame, pd.Series, np.array, list]
    ) -> pd.DataFrame:
        preds = dict()
        for model_name in self.model_dict:
            models = self.model_dict[model_name]
            model_preds = []

            for fold_model in models:
                model_preds.append(fold_model.predict(X))

            mean_preds = average_preds(model_preds)
            for i in range(self.n_labels):
                preds[f"{model_name}_{i}"] = mean_preds[:, i]

        return pd.DataFrame(preds)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, pd.Series, np.array, list],
        y: Union[pd.DataFrame, pd.Series, np.array, list],
    ) -> pd.DataFrame:

        self.fit(X, y)
        transformations = self.transform(X)

        return transformations

    def get_scores(self, prettified: bool = False) -> [np.ndarray, pd.DataFrame]:
        if not self.fitted:
            raise NotFittedError()

        if prettified:
            return pd.DataFrame(data=self.model_scores_dict)

        return np.array(list(self.model_scores_dict.values()))

    def get_models(self) -> [np.ndarray, pd.DataFrame]:
        if not self.fitted:
            raise NotFittedError()

        return self.model_dict.values()
