class BaseWrapper:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y, eval_set):
        self.base_estimator.fit(X, y)

        return self.base_estimator

    def predict(self, X):
        return self.base_estimator.predict(X)


class CatBoostClassifierWrapper(BaseWrapper):
    def __init__(self, base_estimator, use_best_model, cat_features, text_features, embedding_features):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model
        self.cat_features = cat_features
        self.text_features = text_features
        self.embedding_features = embedding_features

    def fit(self, X, y, eval_set):
        self.base_estimator.fit(X,
                                y,
                                eval_set=eval_set,
                                use_best_model=self.use_best_model,
                                cat_features=self.cat_features,
                                text_features=self.text_features,
                                embedding_features=self.embedding_features)

        return self.base_estimator

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)


class CatBoostRegressorWrapper(BaseWrapper):
    def __init__(self, base_estimator, use_best_model, cat_features):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model
        self.cat_features = cat_features

    def fit(self, X, y, eval_set):
        self.base_estimator.fit(X,
                                y,
                                eval_set=eval_set,
                                use_best_model=self.use_best_model,
                                cat_features=self.cat_features)

        return self.base_estimator


class XGBClassifierWrapper(BaseWrapper):
    def __init__(self, base_estimator, use_best_model):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model

    def fit(self, X, y, eval_set):
        self.base_estimator.fit(X,
                                y,
                                eval_set=eval_set,
                                )

        self.best_iteration = self.base_estimator.get_booster().best_ntree_limit

        return self.base_estimator

    def predict(self, X):
        return self.base_estimator.predict(X, ntree_limit=self.best_iteration)

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X, ntree_limit=self.best_iteration)


class XGBRegressorWrapper(BaseWrapper):
    def __init__(self, base_estimator, use_best_model, cat_features):
        super().__init__(base_estimator)
        self.use_best_model = use_best_model
        self.cat_features = cat_features

    def fit(self, X, y, eval_set):
        self.base_estimator.fit(X,
                                y,
                                eval_set=eval_set,
                                )

        self.best_iteration = self.base_estimator.get_booster().best_ntree_limit

        return self.base_estimator

    def predict(self, X):
        return self.base_estimator.predict(X, ntree_limit=self.best_iteration)
