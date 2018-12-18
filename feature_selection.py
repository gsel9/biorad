# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import utils

import numpy as np
import pandas as pd

from scipy import stats
from ReliefF import ReliefF
from sklearn import feature_selection
from multiprocessing import cpu_count

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from model_selection import permutation_test_score
from sklearn.ensemble import RandomForestClassifier

from mlxtend.feature_selection import SequentialFeatureSelector


SEED = 0
METRIC = roc_auc_score


def variance_threshold(data, alpha=0.05):
    """A wrapper of scikit-learn VarianceThreshold."""

    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = feature_selection.VarianceThreshold(threshold=alpha)
    # NB: Cannot filter variance from standardized data.
    selector.fit(X_train, y_train)
    support = _check_support(selector.get_support(indices=True), X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def mutual_info(data, n_neighbors=3, thresh=0.05):
    """A wrapper of scikit-learn mutual information feature selector."""

    X_train, X_test, y_train, y_test = data

    global SEED

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    mut_info = feature_selection.mutual_info_classif(
        X_train_std, y_train, n_neighbors=n_neighbors, random_state=SEED
    )
    # NOTE: Retain features contributing above threshold to model performance.
    support = _check_support((np.argwhere(mut_info > thresh)), X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def relieff(data, n_neighbors=20, k=10):
    """A wrapper of the ReliefF algorithm.

    Args:
        n_neighbors (int): The number of neighbors to consider when assigning
            feature importance scores.

    """
    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = ReliefF(n_neighbors=n_neighbors)
    selector.fit(X_train_std, y_train)

    support = _check_support(selector.top_features[:k], X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def forward_floating(data, scoring=None, model=None, k=3, cv=10):
    """A wrapper of mlxtend Sequential Forward Floating Selection algorithm.

    """
    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    # NOTE: Nested calls not supported by multiprocessing => joblib converts
    # into sequential code (thus, default n_jobs=1).
    #n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()
    n_jobs = 1

    selector = SequentialFeatureSelector(
        model, k_features=k, forward=True, floating=True, scoring='roc_auc',
        cv=cv, n_jobs=n_jobs
    )
    selector.fit(X_train_std, y_train)

    support = _check_support(selector.k_feature_idx_, X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def permutation_test_score(
        estimator,
        X, y,
        n_permutations, n_splits=10, random_state=None, n_jobs=1, scoring=None,
    ):
    """Evaluate the significance of an Out-of-Bag validated score with
    permutations.

    """
    sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    _, scores, pval = permutation_test_score(
        estimator, X, y, cv=sampler, scoring=scoring
    )
    return scores, pval


def _feature_importance_permutation(X, y, model, score_func, num_rounds, seed):

    rgen = np.random.RandomState(seed)

    _, nfeatures = np.shape(X)
    baseline = score_func(y, model.predict(X))

    avg_imp = np.zeros(nfeatures, dtype=float)
    rep_avg_imp = np.zeros((nfeatures, num_rounds))
    for round_idx in range(num_rounds):
        for col_idx in range(nfeatures):

            save_col = X[:, col_idx].copy()
            rgen.shuffle(X[:, col_idx])

            new_score = score_func(y, model.predict(X))
            X[:, col_idx] = save_col
            importance = baseline - new_score
            avg_imp[col_idx] += importance
            rep_avg_imp[col_idx, round_idx] = importance

    return avg_imp / num_rounds, rep_avg_imp


def permutation_importance(data, model=None, thresh=0, nreps=5):
    """A wrapper of mlxtend feature importance permutation algorithm.

    """

    X_train, X_test, y_train, y_test = data

    global METRIC, SEED

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    model.fit(X_train_std, y_train)

    imp, _  = _feature_importance_permutation(
        model=model, X=X_test_std, y=y_test, score_func=METRIC, seed=SEED,
        num_rounds=nreps
    )
    support = _check_support(np.where(imp > thresh), X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def _check_support(support, X):

    if not isinstance(support, np.ndarray):
        support = np.array(support, dtype=int)

    # NB: Default mechanism includes all features if none were selected.
    if len(support) < 1:
        support = np.arange(X.shape[1], dtype=int)
    else:
        if np.ndim(support) > 1:
            support = np.squeeze(support)
        if np.ndim(support) < 1:
            support = support[np.newaxis]
        if np.ndim(support) != 1:
            raise RuntimeError(
                'Invalid dimension {} to support.'.format(np.ndim(support))
            )
    return support


def _check_feature_subset(X_train, X_test, support):

    # Support should be a non-empty vector.
    _X_train, _X_test = X_train[:, support],  X_test[:, support]

    if np.ndim(_X_train) > 2:
        if np.ndim(np.squeeze(_X_train)) > 2:
            raise RuntimeError('X train ndim {}'.format(np.ndim(_X_train)))
        else:
            _X_train = np.squeeze(_X_train)

    if np.ndim(_X_test) > 2:
        if np.ndim(np.squeeze(_X_test)) > 2:
            raise RuntimeError('X test ndim {}'.format(np.ndim(_X_train)))
        else:
            _X_test = np.squeeze(_X_test)

    if np.ndim(_X_train) < 2:
        if np.ndim(_X_train.reshape(-1, 1)) == 2:
            _X_train = _X_train.reshape(-1, 1)
        else:
            raise RuntimeError('X train ndim {}'.format(np.ndim(_X_train)))

    if np.ndim(_X_test) < 2:
        if np.ndim(_X_test.reshape(-1, 1)) == 2:
            _X_test = _X_test.reshape(-1, 1)
        else:
            raise RuntimeError('X test ndim {}'.format(np.ndim(_X_test)))

    return (
        np.array(_X_train, dtype=float), np.array(_X_test, dtype=float),
        support
    )


# https://www.uio.no/studier/emner/matnat/math/STK1000/h17/wilcoxon-rank-sum-test.pdf
def wilcoxon_signed_rank(X, y, thresh=0.05):
    """A nonparametric test to determine whether two dependent samples were
    selected from populations having the same distribution.

    """
    _, ncols = np.shape(X)

    # (H0): The difference between pairs is symmetrically distributted around
    # zero. A p_value < 0.05 => (H1).
    support = []
    for num in range(ncols):
        _, pval = stats.wilcoxon(X[:, num], y)
        # NOTE: Bonferroni correction.
        if pval <= thresh / ncols:
            support.append(num)

    return np.zeros(support, dtype = int)


if __name__ == '__main__':
