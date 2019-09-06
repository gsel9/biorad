# -*- coding: utf-8 -*-
#
# formatting.py
#

"""
Utility functions for type checking, formatting of data and estimators.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np


def check_train_test(X_train, X_test):
    """Formatting checking of training and test predictor sets.

    Args:
        X_train (array-like): Training predictor data.
        X_test (array-like): Test predictor data.

    Returns:
        (tuple): Checked training and test predictor sets.

    """
    # Check training set.
    if np.ndim(X_train) > 2:
        if np.ndim(np.squeeze(X_train)) > 2:
            raise RuntimeError('X train ndim {}'.format(np.ndim(X_train)))
        else:
            X_train = np.squeeze(X_train)
    if np.ndim(X_train) < 2:
        if np.ndim(X_train.reshape(-1, 1)) == 2:
            X_train = X_train.reshape(-1, 1)
        else:
            raise RuntimeError('X train ndim {}'.format(np.ndim(X_train)))
    # Check test set.
    if np.ndim(X_test) > 2:
        if np.ndim(np.squeeze(X_test)) > 2:
            raise RuntimeError('X test ndim {}'.format(np.ndim(X_train)))
        else:
            X_test = np.squeeze(X_test)
    if np.ndim(X_test) < 2:
        if np.ndim(X_test.reshape(-1, 1)) == 2:
            X_test = X_test.reshape(-1, 1)
        else:
            raise RuntimeError('X test ndim {}'.format(np.ndim(X_test)))
    return (
        np.array(X_train, dtype=float), np.array(X_test, dtype=float),
    )
