# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
Wrapped feature selection algorithms providing API compatible with model
comparison framework.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import mifs

import numpy as np
import pandas as pd

from utils import fwutils

from scipy import stats
from ReliefF import ReliefF
from sklearn import feature_selection
from sklearn.base import TransformerMixin

from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector


def permutation_selection(
        X_train, X_test, y_train, y_test,
        score_func,
        model,
        num_rounds,
        random_state,
        **kwargs
    ):
    """Perform feature selection by feature permutation importance.

    Args:
        X_train (array-like): Training predictor set.
        X_test (array-like): Test predictor set.
        y_train (array-like): Training target set.
        y_test (array-like): Test target set.
        score_func (function):
        model ():
        num_rounds (int): The number of permutations per feature.
        random_state (int):

    Returns:
        (tuple): Training subset, test subset and selected features support
            indicators.

    """
    # Z-score transformation.
    X_train_std, X_test_std = fwutils.train_test_z_scores(X_train, X_test)

    model.fit(X_train_std, y_train)

    imp = feature_permutation_importance(
        X_test_std, y_test, score_func, model, num_rounds, random_state,
        **kwargs
    )
    # Return features contributing to model performance as support.
    return X_train_std, X_test_std, np.where(imp > 0)


def feature_permutation_importance(
        X, y, score_func, model, num_rounds, random_state, **kwargs
    ):
    """Feature importance by random feature permutations.

    Args:
        X (array-like): Predictor observations.
        y (array-like): Ground truths.
        score_func (function):
        model ():
        num_rounds (int): The number of permutations per feature.
        random_state (int):

    Returns:
        (array-like): Average feature permutation importances.

    """
    rgen = np.random.RandomState(random_state)

    # Baseline performance.
    baseline = score_func(y, model.predict(X))

    _, num_features = np.shape(X)
    importance = np.zeros(num_features, dtype=float)
    for round_idx in range(num_rounds):
        for col_idx in range(num_features):
            # Store original feature permutation.
            x_orig = X[:, col_idx].copy()
            # Break association between x and y by random permutation.
            rgen.shuffle(X[:, col_idx])
            new_score = score_func(y, model.predict(X))
            # Reinsert original feature prior to permutation.
            X[:, col_idx] = x_orig
            # Feature is likely important if new score < baseline.
            importance[col_idx] += baseline - new_score

    return importance / num_rounds


def wilcoxon_selection(
        X_train, X_test, y_train, y_test, thresh=0.05, **kwargs
    ):
    """Perform feature selection by the Wilcoxon signed-rank.

    Args:
        X_train (array-like): Training predictor set.
        X_test (array-like): Test predictor set.
        y_train (array-like): Training target set.
        y_test (array-like): Test target set.
        thresh (float):

    Returns:
        (tuple): Training subset, test subset and selected features support
            indicators.

    """
    # Z-score transformation.
    X_train_std, X_test_std = fwutils.train_test_z_scores(X_train, X_test)

    support = wilcoxon_signed_rank(
        X_train_std, y_train, thresh=thresh, **kwargs
    )
    return X_train_std, X_test_std, support


def wilcoxon_signed_rank(X, y, thresh=0.05, **kwargs):
    """The Wilcoxon signed-rank test to determine if two dependent samples were
    selected from populations having the same distribution. Includes

    H0: The distribution of the differences x - y is symmetric about zero.
    H1: The distribution of the differences x - y is not symmetric about zero.

    Args:
        X (array-like): Predictor observations.
        y (array-like): Ground truths.
        thresh (float):

    Returns:
        (numpy.ndarray): Support indicators.

    """
    _, ncols = np.shape(X)

    support, pvals = [], []
    for num in range(ncols):
        # If p-value > thresh: same distribution.
        _, pval = stats.wilcoxon(X[:, num], y)
        # Bonferroni correction.
        if pval <= thresh / ncols:
            support.append(num)

    return np.array(support, dtype=int)


def relieff_selection(
        X_train, X_test, y_train, y_test, num_neighbors, num_features, **kwargs
    ):
    """A wrapper for the ReliefF feature selection algorithm.

    Args:
        X_train (array-like): Training predictor set.
        X_test (array-like): Test predictor set.
        y_train (array-like): Training target set.
        y_test (array-like): Test target set.
        num_neighbors (int): The number of neighbors to consider when assigning
            feature importance scores.
        num_features (): The number of features to select.

    Returns:
        (tuple): Training subset, test subset and selected features support
            indicators.

    """
    # Z-score transformation.
    X_train_std, X_test_std = fwutils.train_test_z_scores(X_train, X_test)

    selector = ReliefF(n_neighbors=num_neighbors)
    selector.fit(X_train_std, y_train)

    return X_train_std, X_test_std, selector.top_features[:num_features]


# NOTE:
# * Cloned from: https://github.com/danielhomola/mifs
# * Requirements: TODO
def mrmr_selection(X_train, X_test, y_train, y_test, k, num_features, **kwargs):
    """Minimum redundancy maximum relevancy.

    Args:
        X_train (array-like): Training predictor set.
        X_test (array-like): Test predictor set.
        y_train (array-like): Training target set.
        y_test (array-like): Test target set.

    Returns:
        (tuple): Training subset, test subset and selected features support
            indicators.

    """
    # Z-score transformation.
    X_train_std, X_test_std = fwutils.train_test_z_scores(X_train, X_test)
    # If 'auto': n_features is determined based on the amount of mutual
    # information the previously selected features share with y.
    selector = mifs.MutualInformationFeatureSelector(
        method='MRMR', k=5, n_features=num_features, categorical=True,
    )
    selector.fit(X_train_std, y_train)

    return X_train_std, X_test_std, np.where(selector.support_ == True)
