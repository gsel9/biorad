"""
Utility functions for predictive modeling experiments.
"""


import pandas as pd
import numpy as np


def load_target(path_to_target: str, index_col: int = 0, classify: bool = True):
    """

    """
    var = pd.read_csv(path_to_target, index_col=index_col)
    if classify:
        return np.squeeze(var.values).astype(np.int32)

    return np.squeeze(var.values).astype(np.float32)


def load_predictors(path_to_data: str, index_col: int = 0, regex: str = None):
    """

    """
    data = pd.read_csv(path_to_data, index_col=index_col)
    if regex is None:
        return np.array(data.values, dtype=np.float32)

    target_features = data.filter(regex=regex).columns

    return np.array(data.loc[:, target_features].values, dtype=np.float32)
