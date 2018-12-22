# -*- coding: utf-8 -*-
#
# utils.py
#

"""
Model comparison backend utility functions.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np

from numba import jit
from sklearn.preprocessing import StandardScaler


class Selector:
    """Representation of a feature selection procedure.

    Args:
        name (str): Name of feature selection procedure.
        funt (function): Function executing the selection procedure.
        params (dict): Parameters passed to the selection procedure.

    Returns:
        (tuple): Training subset, test subset and selected features support
            indicators.

    """

    def __init__(self, name, func, params):

        self.name = name
        self.func = func
        self.params = params

    def __call__(self, *args **kwargs):

        X_train_std, X_test_std, _support = self.func(
            *args, **self.params, **kwargs
        )
        support = self._check_support(_support, X_train_std)

        return self._check_feature_subset(X_train_std, X_test_std, support)

    @staticmethod
    def _check_support(support, X):
        # Formatting of indicators subset.

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

    @staticmethod
    def _check_feature_subset(X_train, X_test, support):
        # Formatting training and test subsets.

        # Support should be a non-empty vector (ensured in _check_support).
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


class BootstrapOutOfBag:
    """A bootstrap Out-of-Bag resampler.

    Args:
        n_splits (int): The number of resamplings to perform.
        random_state (int): Seed for the pseudo-random number generator.

    """

    def __init__(self, n_splits=10, random_state=None):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, **kwargs):
        """Generates Out-of-Bag samples.

        Args:
            X (array-like): The predictor data.
            y (array-like): The target data.

        Returns:
            (genrator): An iterable with X and y sample indicators.

        """
        rgen = np.random.RandomState(self.random_state)

        nrows, _ = np.shape(X)
        sample_indicators = np.arange(nrows)
        for _ in range(self.n_splits):
            # Sample with replacement.
            train_idx = rgen.choice(
                sample_indicators, size=nrows, replace=True
            )
            # Oberervations not part of training set defines test set.
            test_idx = np.array(
                list(set(sample_indicators) - set(train_idx)), dtype=int
            )
            yield train_idx, test_idx


def check_support(support):
    """Format selected support."""
    if np.ndim(support) > 1:
        return np.squeeze(support)

    if not isinstance(support, np.ndarray):
        return np.array(support, dtype=int)


# This approach is more robust towards situations where no features are
# selected, but may result in very small feature subsets. Increasing feature
# robustness.
def select_support(features):
    """Select feature indicators according to maximum number of votes."""
    max_counts = np.max(features)
    return np.squeeze(np.where(features == max_counts)), max_counts


def select_hparams(hparams):
    """Select hyperparameters according most frequently occuring settings."""
    # NOTE: Returns original input if all parameters have equal number of votes.
    try:
        return max(hparams, key=hparams.count)
    except:
        return hparams


def train_test_z_scores(X_train, X_test):
    """Apply Z-score transformation to features in training and test sets.

    Args:
        X_train (array-like): Training set.
        X_test (array-like): Test set.

    Returns:
        (tuple): Transformed training and test set.

    """
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    # Apply training params in transforming test set: renders test set
    # comparable to training set while preventing information bleeding.
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std


def scale_fit_predict632(model, X_train, X_test, y_train, y_test, score_func):
    """Assess model performance on Z-score transformed training and test sets.

    Args:
        model (object): An untrained model with `fit()` and `predict` methods.
        X_train (array-like): Training set.
        X_test (array-like): Test set.
        y_train (array-like): Training set ground truths.
        y_test (array-like): Test set ground truths.
        score_func (function): A score function for model performance
            evaluation.

    Returns:
        (float):

    """
    # Compute Z scores.
    X_train_std, X_test_std = train_test_z_scores(X_train, X_test)

    model.fit(X_train_std, y_train)

    # Aggregate model predictions.
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)

    train_score = score_func(y_train, y_train_pred)
    test_score = score_func(y_test, y_test_pred)

    # Compute .632+ scores.
    train_632_score = point632plus_score(
        y_train, y_train_pred, train_score, test_score
    )
    test_632_score = point632plus_score(
        y_test, y_test_pred, train_score, test_score
    )
    return train_632_score, test_632_score


def point632plus_score(y_true, y_pred, train_score, test_score):
    """Compute .632+ score for binary classification.

    Args:
        y_true (array-like): Ground truths.
        y_pred (array-like): Predictions.
        train_score (float): Resubstitution score.
        test_score (float): True score.

    Returns:
        (float): The .632+ score value.

    """
    gamma = _no_info_rate_binary(y_true, y_pred)
    # Calculate adjusted parameters as described in Efron & Tibshiranir paper.
    test_score_marked = min(test_score, gamma)
    r_marked = relative_overfit_rate(train_score, test_score, gamma)

    return point632plus(train_score, test_score, r_marked, test_score_marked)


@jit
def point632plus(train_score, test_score, r_marked, test_score_marked):
    """Calculate the .632+ score from parameters.

    Args:
        train_score (float): The resubstitution score.
        test_score (float): The true score.
        r_marked (float): Adjusted relative overfitting rate.
        test_score_marked (float):

    Returns:
        (float): The .632+ score value.

    """
    point632 = 0.368 * train_score + 0.632 * test_score
    frac = (0.368 * 0.632 * r_marked) / (1 - 0.368 * r_marked)

    return point632 + (test_score_marked - train_score) * frac


@jit
def relative_overfit_rate(train_score, test_score, gamma):
    """Calculate the relative overfitting rate from parameters.

    Args:
        train_score (float): The resubstitution score.
        test_score (float): The true score.
        gamma (float): The no information rate.

    Returns:
        (float): The relative overfitting rate value.

    """
    if test_score > train_score and gamma > train_score:
        return (test_score - train_score) / (gamma - train_score)
    else:
        return 0


def _no_info_rate_binary(y_true, y_pred):
    # NB: Only applicable to a dichotomous classification problem.
    p_one = np.sum(y_true) / np.size(y_true)
    q_one = np.sum(y_pred) / np.size(y_pred)

    return p_one * (1 - q_one) + (1 - p_one) * q_one
