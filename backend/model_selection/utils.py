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


def scale_fit_predict632(X_train, X_test, y_train, y_test, score_func, model):
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
    # NOTE: Error handling mechanism.
    try:
        model.fit(X_train_std, y_train)
    except:
        return None, None

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


def check_estimator(hparams, estimator, support=None, random_state=None):

    # In case num components > support.
    if support is not None and 'n_components' in hparams:
        # NOTE: Adjusting to Python counting logic.
        if len(support) - 1 < hparams['n_components']:
            hparams['n_components'] = len(support) - 1

    # If stochastic algorithm.
    try:
        model = estimator(**hparams, random_state=random_state)
    except:
        model = estimator(**hparams)

    return model
