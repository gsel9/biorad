# -*- coding: utf-8 -*-
#
# utils.py
#

"""
Utility module for the head and neck project.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import logging
import radiomics

import numpy as np

from numba import jit
from sklearn.preprocessing import StandardScaler


def listdir(path_to_dir, skip_tail=('.csv'), skip_head=('.')):

    labels = []
    for label in os.listdir(path_to_dir):
        if not label.endswith(skip_tail) and not label.startswith(skip_head):
            labels.append(str(label))

    return labels


def train_test_z_scores(X_train, X_test):
    """Compute Z-scores for training and test sets.

    Args:
        X_train (array-like): Training set.
        X_test (array-like): Test set.

    Returns:
        (tuple): Standard score values for training and test set.

    """

    # NOTE: Avoid information leakage and make train/test sets comparable.
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std


class BootstrapOutOfBag:

    def __init__(self, n_splits=10, random_state=None):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, **kwargs):

        rand_gen = np.random.RandomState(self.random_state)

        nrows, _ = np.shape(X)
        sample_indicators = np.arange(nrows)
        for _ in range(self.n_splits):
            train_idx = rand_gen.choice(
                sample_indicators, size=nrows, replace=True
            )
            test_idx = np.array(
                list(set(sample_indicators) - set(train_idx)), dtype=int
            )
            yield train_idx, test_idx


def check_support(support):

    if np.ndim(support) > 1:
        return np.squeeze(support)

    if not isinstance(support, np.ndarray):
        return np.array(support, dtype=int)


@jit
def point632plus(train_score, test_score, r_marked, test_score_marked):

    point632 = 0.368 * train_score + 0.632 * test_score
    frac = (0.368 * 0.632 * r_marked) / (1 - 0.368 * r_marked)

    return point632 + (test_score_marked - train_score) * frac


@jit
def relative_overfit_rate(train_score, test_score, gamma):

    if test_score > train_score and gamma > train_score:
        return (test_score - train_score) / (gamma - train_score)
    else:
        return 0


# NOTE: Need only use sum(y_) if y_ is binary.
def no_info_rate(y_true, y_pred):

    # NB: Only applicable to a dichotomous classification problem.
    p_one = np.sum(y_true == 1) / np.size(y_true)
    q_one = np.sum(y_pred == 1) / np.size(y_pred)

    return p_one * (1 - q_one) + (1 - p_one) * q_one


def point632plus_score(y_true, y_pred, train_score, test_score):

    gamma = no_info_rate(y_true, y_pred)
    # To account for gamma <= train_score/train_score < gamma <= test_score
    # in which case R can fall outside of [0, 1].
    test_score_marked = min(test_score, gamma)

    # Adjusted R.
    r_marked = relative_overfit_rate(train_score, test_score, gamma)

    # Compute .632+ train score.
    return point632plus(train_score, test_score, r_marked, test_score_marked)


def scale_fit_predict632(*args, score_func=None, **kwargs):

    model, X_train, X_test, y_train, y_test = args

    # Compute Z scores.
    X_train_std, X_test_std = train_test_z_scores(X_train, X_test)

    model.fit(X_train_std, y_train)

    # Aggregate model predictions.
    y_train_pred = model.predict(X_train_std)
    train_score = score_func(y_train, y_train_pred)

    y_test_pred = model.predict(X_test_std)
    test_score = score_func(y_test, y_test_pred)

    # Compute train and test .632+ scores.
    train_632_score = point632plus_score(
        y_train, y_train_pred, train_score, test_score
    )
    test_632_score = point632plus_score(
        y_test, y_test_pred, train_score, test_score
    )
    return train_632_score, test_632_score
