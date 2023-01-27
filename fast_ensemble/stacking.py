from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from fast_ensemble.errors import NotFittedError, NameIntersectionError, FoldMismatchError
from fast_ensemble.utils import average_preds, to_pandas
from fast_ensemble.wrappers import (
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    LGBMClassifierWrapper,
    LGBMRegressorWrapper,
    XGBClassifierWrapper,
    XGBRegressorWrapper,
)


class StackingTransformer:
    """
    Stacking ensemble model
    """

    def __init__(
        self,
        models: list,
        main_metric,
        n_folds: int = 5,
        random_state: int = None,
        shuffle: bool = False,
        stratified: bool = True,
        verbose: bool = True,
        regression: bool = True,
        stratification_bins: int = 10,
        custom_bins: np.ndarray = None
    ):

        self.models = [i[1] for i in models]
        self.n_models = len(self.models)
        self.names = [i[0] for i in models]
        self.model_dict = {}
        self.model_scores_dict = {}

        self.n_folds = n_folds
        self.main_metric = main_metric
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratified = stratified
        self.verbose = verbose
        self.regression = regression
        self.stratification_bins = stratification_bins
        self.custom_bins = custom_bins

        self.fitted = False

    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.array, list],
        y: Union[pd.DataFrame, pd.Series, np.array, list],
    ) -> Any:

        self.model_dict = {}
        self.model_scores_dict = {}

        X = to_pandas(X)

        self.__get_n_labels(y)

        if self.stratified:
            kf = StratifiedKFold(
                n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle
            )
        else:
            kf = KFold(
                n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle
            )

        # Form each model
        for model_i in range(self.n_models):
            if self.verbose:
                print(self.names[model_i])

            # For each fold
            sub_models = []
            sub_scores = []

            if self.regression:
                if isinstance(self.custom_bins, np.ndarray):
                    digitized = np.digitize(y, self.custom_bins)
                else:
                    bins = np.linspace(y.min(), y.max(), self.stratification_bins)
                    digitized = np.digitize(y, bins)

                split = kf.split(X, digitized)
            else:
                split = kf.split(X, y)

            for fold, (train_index, test_index) in enumerate(split):
                sub_model = self.models[model_i]

                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if type(sub_model) in [
                    CatBoostRegressorWrapper,
                    CatBoostClassifierWrapper,
                    XGBRegressorWrapper,
                    XGBClassifierWrapper,
                    LGBMClassifierWrapper,
                    LGBMRegressorWrapper,
                ]:

                    sub_model = sub_model.fit(
                        X_train, y_train, eval_set=(X_test, y_test)
                    )
                else:
                    sub_model = sub_model.fit(X_train, y_train)

                sub_models.append(sub_model)

                if self.regression:
                    sub_score = self.main_metric(y_test, sub_model.predict(X_test))
                else:
                    sub_score = self.main_metric(
                        y_test, sub_model.predict_proba(X_test)
                    )

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
        all_preds = []

        for model_name in self.model_dict:
            models = self.model_dict[model_name]
            model_preds = []

            for fold_model in models:
                if self.regression:
                    fold_pred = fold_model.predict(X)
                else:
                    fold_pred = fold_model.predict_proba(X)

                if len(fold_pred.shape) == 1:
                    model_preds.append(fold_pred.reshape(-1, self.n_labels))
                else:
                    model_preds.append(fold_pred)

            mean_preds = average_preds(model_preds)
            all_preds.append(mean_preds)

        return self.__prettify_preds(all_preds)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, pd.Series, np.array, list],
        y: Union[pd.DataFrame, pd.Series, np.array, list],
    ) -> pd.DataFrame:

        self.model_dict = {}
        self.model_scores_dict = {}

        X = to_pandas(X)

        self.__get_n_labels(y)

        if self.stratified:
            kf = StratifiedKFold(
                n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle
            )
        else:
            kf = KFold(
                n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle
            )

        all_preds = []
        # Form each model
        for model_i in range(self.n_models):
            if self.verbose:
                print(self.names[model_i])

            # For each fold
            sub_models = []
            sub_scores = []
            sub_preds = np.zeros((X.shape[0], self.n_labels))

            if self.regression:
                if isinstance(self.custom_bins, np.ndarray):
                    digitized = np.digitize(y, self.custom_bins)
                else:
                    bins = np.linspace(y.min(), y.max(), self.stratification_bins)
                    digitized = np.digitize(y, bins)

                split = kf.split(X, digitized)
            else:
                split = kf.split(X, y)

            for fold, (train_index, test_index) in enumerate(split):
                sub_model = self.models[model_i]

                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if type(sub_model) in [
                    CatBoostRegressorWrapper,
                    CatBoostClassifierWrapper,
                    XGBRegressorWrapper,
                    XGBClassifierWrapper,
                    LGBMClassifierWrapper,
                    LGBMRegressorWrapper,
                ]:

                    sub_model = sub_model.fit(
                        X_train, y_train, eval_set=(X_test, y_test)
                    )
                else:
                    sub_model = sub_model.fit(X_train, y_train)

                sub_models.append(sub_model)

                if self.regression:
                    preds = sub_model.predict(X_test)
                    sub_score = self.main_metric(y_test, preds)
                else:
                    preds = sub_model.predict_proba(X_test)
                    sub_score = self.main_metric(y_test, preds)

                sub_preds[test_index] = preds.reshape((-1, self.n_labels))

                sub_scores.append(sub_score)

                if self.verbose:
                    print(f"Fold: {fold}, Score: {sub_score}")

            self.model_dict[self.names[model_i]] = sub_models
            self.model_scores_dict[self.names[model_i]] = sub_scores
            all_preds.append(sub_preds)

        self.fitted = True

        return self.__prettify_preds(all_preds)

    def get_scores(self, prettified: bool = True) -> [np.ndarray, pd.DataFrame]:
        if not self.fitted:
            raise NotFittedError()

        if prettified:
            return pd.DataFrame(data=self.model_scores_dict)

        return np.array(list(self.model_scores_dict.values()))

    def get_models(self) -> dict:
        if not self.fitted:
            raise NotFittedError()

        return self.model_dict

    def merge(self, other_stack: "StackingTransformer") -> None:
        other_names = other_stack.names

        intersecting_names = set(self.names).intersection(set(other_names))

        if len(intersecting_names) > 0:
            raise NameIntersectionError(intersecting_names)

        if self.n_folds != other_stack.n_folds:
            raise FoldMismatchError(expected_n_folds=self.n_folds, received_n_folds=other_stack.n_folds)

        self.names.extend(other_names)

        other_model_dict = other_stack.model_dict
        for key in other_model_dict:
            self.model_dict[key] = other_model_dict[key]

        self.models.extend(other_stack.models)

        self.n_models += other_stack.n_models

        other_model_scores_dict = other_stack.model_scores_dict
        for key in other_model_scores_dict:
            self.model_scores_dict[key] = other_model_scores_dict[key]


    def __prettify_preds(self, preds: list) -> pd.DataFrame:
        transformations = pd.DataFrame(
            np.hstack(preds),
            columns=[
                f"{model_name}_{i}"
                for i in range(self.n_labels)
                for model_name in self.names
            ],
        )

        return transformations

    def __get_n_labels(self, y) -> None:
        if len(y.shape) == 1:
            if not self.regression:
                self.n_labels = len(np.unique(y))
            else:
                self.n_labels = 1
        else:
            self.n_labels = y.shape[1]
