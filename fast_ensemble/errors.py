class NotFittedError(Exception):
    def __init__(
        self,
        message="This model has not been fitted yet. Use .fit() to train the model.",
    ):
        self.message = message
        super().__init__(self.message)


class NameIntersectionError(Exception):
    def __init__(
        self,
        names,
    ):
        self.message = f"Unable to merge with other stack due to intersection in following names: {names}"
        super().__init__(self.message)


class FoldMismatchError(Exception):
    def __init__(
        self,
        expected_n_folds,
        received_n_folds,
    ):
        self.message = f"Unable to merge with other stack due to inconsistent amount of folds. Expected {expected_n_folds}, received {received_n_folds}"
        super().__init__(self.message)