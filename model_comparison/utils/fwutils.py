# -*- coding: utf-8 -*-
#
# utils.py
#

"""
Model comparison framework utility functions.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


class BootstrapOutOfBag:
    """A bootstrap Out-of-Bag resampler.

    Args:
        n_splits (int): The number of resamplings to perform.
        random_state (int): Seed for the pseudo-random number generator.

    """

    def __init__(self, n_splits, random_state):

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


def balance_data(X, y, random_state):
    """Balance distribution of target classes with the Synthetic Minority
    Oversmapling Technique proposed by ...

    Args:
        X (array-like):
        y (array-like):
        random_state (int):

    Returns:
        (tuple): Balanced predictor and target data sets.

    """

    balancer = SMOTE(random_state=random_state)
    return balancer.fit_sample(X, y)


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
    X_train, X_test = check_train_test(X_train, X_test)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    # Apply training params in transforming test set: renders test set
    # comparable to training set while preventing information bleeding.
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std


def check_estimator(estimator, hparams, support_size=None, random_state=None):
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
    if support_size is not None and 'n_components' in hparams:
        # NOTE: Adjusting to Python counting logic.
        if support_size - 1 < hparams['n_components']:
            hparams['n_components'] = support_size - 1

    # If stochastic algorithm.
    try:
        model = estimator(**hparams, random_state=random_state)
    except:
        model = estimator(**hparams)

    return model
