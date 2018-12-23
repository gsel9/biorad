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
def select_support(features, method='max', thresh=0.05):
    """Select feature indicators according to maximum number of votes."""

    max_counts = np.max(features)
    if method == 'max':
        return np.squeeze(np.where(features == max_counts)), max_counts
    elif method == 'frac':
        fracs = features / max_counts
        return np.squeeze(np.where(fracs >= thresh)), max_counts


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


def check_estimator(hparams, estimator, support=None, random_state=None):
    """Ensure correct estimator setup.

    Args:
        hparams (dict):
        estimator (object):
        support (optional, array-like): Adjust for subpsace methods expecting
            more components than there are available features.
        random_state (int):

    Returns:
        (object): Instantiated model.

    """

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
