from fast_ensemble.errors import NotFittedError


class BaseWrapper:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.fitted = False

    def fit(self, X, y, eval_set: tuple = None, early_stopping_rounds: int = None):
        self.base_estimator.fit(X, y)

        self.fitted = True

        return self.base_estimator

    def get_model(self):
        return self.base_estimator

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict(X)


class CatBoostClassifierWrapper(BaseWrapper):
    def __init__(
        self,
        base_estimator,
        use_best_model: bool = False,
        cat_features: list = None,
        text_features: list = None,
        embedding_features: list = None,
    ):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model
        self.cat_features = cat_features
        self.text_features = text_features
        self.embedding_features = embedding_features

    def fit(self, X, y, eval_set: tuple = None, early_stopping_rounds: int = None):
        self.base_estimator.fit(
            X,
            y,
            eval_set=eval_set,
            use_best_model=self.use_best_model,
            cat_features=self.cat_features,
            text_features=self.text_features,
            embedding_features=self.embedding_features,
            early_stopping_rounds=early_stopping_rounds,
        )

        self.fitted = True

        return self.base_estimator

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict_proba(X)


class CatBoostRegressorWrapper(BaseWrapper):
    def __init__(
        self, base_estimator, use_best_model: bool = False, cat_features: list = None
    ):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model
        self.cat_features = cat_features

    def fit(self, X, y, eval_set: tuple = None, early_stopping_rounds: int = None):
        self.base_estimator.fit(
            X,
            y,
            eval_set=eval_set,
            use_best_model=self.use_best_model,
            cat_features=self.cat_features,
            early_stopping_rounds=early_stopping_rounds,
        )

        self.fitted = True

        return self.base_estimator


class XGBClassifierWrapper(BaseWrapper):
    def __init__(self, base_estimator, use_best_model: bool = False):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model

    def fit(self, X, y, eval_set: tuple = None, early_stopping_rounds: int = None):
        self.base_estimator.fit(
            X, y, eval_set=[eval_set], early_stopping_rounds=early_stopping_rounds
        )

        self.fitted = True

        return self.base_estimator

    def get_iterations(self):
        if not self.fitted:
            raise NotFittedError()

        if self.use_best_model:
            return self.base_estimator.get_booster().best_ntree_limit

        return self.base_estimator.n_estimators

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict(X, ntree_limit=self.get_iterations())

    def predict_proba(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict_proba(X, ntree_limit=self.get_iterations())


class XGBRegressorWrapper(BaseWrapper):
    def __init__(self, base_estimator, use_best_model: bool = False):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model

    def fit(self, X, y, eval_set: tuple = None, early_stopping_rounds: int = None):
        self.base_estimator.fit(
            X, y, eval_set=[eval_set], early_stopping_rounds=early_stopping_rounds
        )

        self.fitted = True

        return self.base_estimator

    def get_iterations(self):
        if not self.fitted:
            raise NotFittedError()

        if self.use_best_model:
            return self.base_estimator.get_booster().best_ntree_limit

        return self.base_estimator.n_estimators

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict(X, ntree_limit=self.get_iterations())


class LGBMClassifierWrapper(BaseWrapper):
    def __init__(self, base_estimator, use_best_model: bool = False):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model

    def fit(self, X, y, eval_set: tuple = None, early_stopping_rounds: int = None):
        self.base_estimator.fit(
            X, y, eval_set=[eval_set], early_stopping_rounds=early_stopping_rounds
        )

        self.fitted = True

        return self.base_estimator

    def get_iterations(self):
        if not self.fitted:
            raise NotFittedError()

        if self.use_best_model:
            return self.base_estimator.best_iteration

        return self.base_estimator.n_estimators_

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict(X, ntree_limit=self.get_iterations())

    def predict_proba(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict_proba(X, ntree_limit=self.get_iterations())


class LGBMRegressorWrapper(BaseWrapper):
    def __init__(self, base_estimator, use_best_model: bool = False):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model

    def fit(self, X, y, eval_set: tuple = None, early_stopping_rounds: int = None):
        self.base_estimator.fit(
            X, y, eval_set=[eval_set], early_stopping_rounds=early_stopping_rounds
        )

        self.fitted = True

        return self.base_estimator

    def get_iterations(self):
        if not self.fitted:
            raise NotFittedError()

        if self.use_best_model:
            return self.base_estimator.best_iteration

        return self.base_estimator.n_estimators_

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError()

        return self.base_estimator.predict(X, ntree_limit=self.get_iterations())
