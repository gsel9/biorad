# -*- coding: utf-8 -*-
#
# formatting.py
#

"""
Utility functions for type checking, formatting of data and estimators.

Notes:
* Need to select hparams across different experiments? Keep function?
* No need to select support across different runs? Remove function?

To Dos:
* Include check_train_test() in model selection functions
* Include check_support() in FS estimator.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np

from numba import int32
from numba import jitclass
from numba import jit

from collections import OrderedDict

from imblearn import over_sampling
from sklearn.preprocessing import StandardScaler


def pipelines_from_configs(selector_configs, estimator_configs):
    # Iterate through configs and build pipelines in <dict>.
    pass


def _pipeline_from_config(config):
    pass


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


def check_subset(X):
    """Formatting and type checking of feature subset.

    Args:
        X (array-like):

    Returns:
        (array-like): Formatted feature subset.

    """

    # Check training set.
    if np.ndim(X) > 2:
        if np.ndim(np.squeeze(X)) > 2:
            raise RuntimeError('X train ndim {}'.format(np.ndim(X)))
        else:
            X = np.squeeze(X)
    if np.ndim(X) < 2:
        if np.ndim(X.reshape(-1, 1)) == 2:
            X = X.reshape(-1, 1)
        else:
            raise RuntimeError('X train ndim {}'.format(np.ndim(X_train)))

    return np.array(X, dtype=float)



def check_support(support, X):
    """Formatting of feature subset indicators.

    Args:
        support (array-like): Feature subset indicators.
        X (array-like): Original feature matrix.

    Returns:
        (array-like): Formatted indicators of feature subset.

    """
    if not isinstance(support, np.ndarray):
        support = np.array(support, dtype=int)

    # NB: Include all features if none were selected.
    if len(support) - 1 < 1:
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


# TODO: Moving to separate module.
@jit
def select_hparams(hparams):
    """Select hyperparameters according most frequently occuring settings."""
    # NOTE: Returns original input if all parameters have equal number of votes.
    try:
        return max(hparams, key=hparams.count)
    except:
        return hparams


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
