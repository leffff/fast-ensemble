from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd


def to_pandas(X: Any) -> pd.DataFrame:
    dtype = type(X)

    if dtype == pd.DataFrame:
        return X

    elif dtype == pd.Series:
        return pd.DataFrame(X)

    elif dtype == np.ndarray:
        return pd.DataFrame(X)

    elif dtype == list:
        return pd.DataFrame(X)

    else:
        raise ValueError(
            f"Wrong dtype. Expected pd.DataFrame, pd.Series, np.ndarray or list, found {dtype}"
        )


def average_preds(preds_array):
    return sum(preds_array) / len(preds_array)
