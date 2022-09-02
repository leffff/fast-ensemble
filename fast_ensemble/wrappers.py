from fast_ensemble.errors import NotFittedError


class BaseRegressorWrapper:
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None
    ):
        self.base_estimator = base_estimator
        self.use_best_model = use_best_model
        self.early_stopping_rounds = early_stopping_rounds
        self.fitted = False

    def fit(self, X, y, eval_set: tuple = None):
        self.base_estimator.fit(X, y)

        self.fitted = True

        return self.base_estimator

    def get_model(self):
        return self.base_estimator

    def get_iterations(self):
        return self.base_estimator.n_estimators

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict(X, ntree_limit=self.get_iterations())


class BaseClassifierWrapper(BaseRegressorWrapper):
    def __init__(self, base_estimator, use_best_model: bool = False, early_stopping_rounds: int = None):
        super().__init__(base_estimator, use_best_model, early_stopping_rounds)
        self.base_estimator = base_estimator
        self.use_best_model = use_best_model
        self.fitted = False

    def predict_proba(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict_proba(X, ntree_limit=self.get_iterations())


class CatBoostWrapper:
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None
    ):
        self.base_estimator = base_estimator
        self.use_best_model = use_best_model
        self.early_stopping_rounds = early_stopping_rounds
        self.fitted = False

    def get_iterations(self):
        if not self.fitted:
            raise NotFittedError()

        if self.use_best_model:
            return self.base_estimator.model.get_best_iteration()

        return self.base_estimator.n_estimators


class CatBoostClassifierWrapper(BaseClassifierWrapper, CatBoostWrapper):
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None,
        cat_features: list = None,
        text_features: list = None,
        embedding_features: list = None
    ):
        super().__init__(base_estimator, use_best_model, early_stopping_rounds)
        self.cat_features = cat_features
        self.text_features = text_features
        self.embedding_features = embedding_features

    def fit(self, X, y, eval_set: tuple = None):
        self.base_estimator.fit(
            X,
            y,
            eval_set=eval_set,
            use_best_model=self.use_best_model,
            cat_features=self.cat_features,
            text_features=self.text_features,
            embedding_features=self.embedding_features,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        self.fitted = True

        return self.base_estimator


class CatBoostRegressorWrapper(BaseRegressorWrapper, CatBoostWrapper):
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None,
        cat_features: list = None
    ):
        super().__init__(base_estimator, use_best_model, early_stopping_rounds)
        self.cat_features = cat_features

    def fit(self, X, y, eval_set: tuple = None):
        self.base_estimator.fit(
            X,
            y,
            eval_set=eval_set,
            use_best_model=self.use_best_model,
            cat_features=self.cat_features,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        self.fitted = True

        return self.base_estimator


class XGBWrapper:
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None
    ):
        self.base_estimator = base_estimator
        self.use_best_model = use_best_model
        self.early_stopping_rounds = early_stopping_rounds
        self.fitted = False

    def get_iterations(self):
        if not self.fitted:
            raise NotFittedError()

        if self.use_best_model:
            return self.base_estimator.get_booster().best_ntree_limit

        return self.base_estimator.n_estimators

    def fit(self, X, y, eval_set: tuple = None):
        self.base_estimator.fit(
            X, y, eval_set=[eval_set], early_stopping_rounds=self.early_stopping_rounds
        )

        self.fitted = True

        return self.base_estimator


class XGBClassifierWrapper(BaseClassifierWrapper, XGBWrapper):
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None
    ):
        super().__init__(base_estimator, use_best_model, early_stopping_rounds)


class XGBRegressorWrapper(BaseRegressorWrapper, XGBWrapper):
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None
    ):
        super().__init__(base_estimator, use_best_model, early_stopping_rounds)


class LGBMWrapper:
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None
    ):
        self.base_estimator = base_estimator
        self.use_best_model = use_best_model
        self.early_stopping_rounds = early_stopping_rounds
        self.fitted = False

    def get_iterations(self):
        if not self.fitted:
            raise NotFittedError()

        if self.use_best_model:
            return self.base_estimator.best_iteration

        return self.base_estimator.n_estimators_

    def fit(self, X, y, eval_set: tuple = None):
        self.base_estimator.fit(
            X, y, eval_set=[eval_set], early_stopping_rounds=self.early_stopping_rounds
        )

        self.fitted = True

        return self.base_estimator


class LGBMClassifierWrapper(BaseClassifierWrapper, LGBMWrapper):
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None
    ):
        super().__init__(base_estimator, use_best_model, early_stopping_rounds)


class LGBMRegressorWrapper(BaseRegressorWrapper, LGBMWrapper):
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        early_stopping_rounds: int = None
    ):
        super().__init__(base_estimator, use_best_model, early_stopping_rounds)
