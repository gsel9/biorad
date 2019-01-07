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

TEMP:
# Parameter search space
space = {}
# Random number between 50 and 100
space['kbest__percentile'] = hp.uniform('kbest__percentile', 50, 100)
# Random number between 0 and 1
#space['clf__l1_ratio'] = hp.uniform('clf__l1_ratio', 0.0, 1.0)
# Log-uniform between 1e-9 and 1e-4
#space['clf__alpha'] = hp.loguniform('clf__alpha', -9*np.log(10), -4*np.log(10))
# Random integer in 20:5:80
#space['clf__n_iter'] = 20 + 5 * hp.randint('clf__n_iter', 12)
# Random number between 50 and 100
space['clf__class_weight'] = hp.choice('clf__class_weight', [None,]) #'balanced']),
space['clf__n_estimators'] = scope.int(hp.quniform('clf__clf__n_estimators', 20, 500, 5))
# Discrete uniform distribution
space['clf__max_leaf_nodes'] = scope.int(hp.quniform('clf__max_leaf_nodes', 30, 150, 1))
# Discrete uniform distribution
space['clf__min_samples_leaf'] = scope.int(hp.quniform('clf__min_samples_leaf', 20, 500, 5))

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

import numpy as np

from numba import int32
from numba import jitclass
from numba import jit

from collections import OrderedDict

from imblearn import over_sampling

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin


class PipeEstimator(BaseEstimator, MetaEstimatorMixin):
    """A scikit-learn estimator wrapper enabling intermediate configuration of
    the wrapped model between pipeline steps.

    In particular, configrations are necessary with respect to:
    - Adjusting the number of components to keep in decomposition methods
      should the previous pipeline transformer reduce the feature set resulting
      in an expected number of cmoponents exceed the number of available
      features.

    Args:
        model ():

    """

    def __init__(self, model=None, params=None):

        super().__init__()

        self.model = model
        self.params = params

    def __name__(self):

        return self.model.__name__

    def set_params(self, **params):
        """Update estimator hyperparamter configuration.

        Kwargs:
            params (dict): Hyperparameter settings.

        """

        # Update hyperparameters and estimator configuration.
        self.params = params
        self.model.set_params(**self.params)

        return self

    def get_params(self, deep=True):
        """Returns hyperparameter configurations."""

        return self.model.get_params(deep=deep)

    def fit(self, X, y=None, **kwargs):

        self._set_estimator_config(X, y=y, **kwargs)
        self.model.fit(X, y, **kwargs)

        return self

    def predict(self, X):

        return self.model.predict(X)

    def _set_estimator_config(self, X, y=None, **kwargs):
        # Validate model configuration.

        # Update hyperparameter settings.
        if 'n_components' in self.params:
            if self.params['n_components'] > X.shape[1]:
                self.params['n_components'] = X.shape[1]

        # Update hyperparameters.
        self.model.set_params(**self.params)

        return self


def pipelines_from_configs(selector_configs, estimator_configs):
    """

    Args:
        selector_configs (dict):
        estimator_configs (dict):

    Returns:
        (dict):

    """
    pipes_and_params = OrderedDict()
    for classifier_name, clf_setup in estimator_configs.items():
        for selector_name, sel_setup in selector_configs.items():
            pipe_label = '{}_{}'.format(selector_name, classifier_name)
            # Joining two lists of selector and estimator pipe elements.
            pipe_elem = [*sel_setup['selector'], *clf_setup['estimator']]
             # Joining two dicts of selector and estimator parameters.
            param_space = {**sel_setup['params'], **clf_setup['params']}
            # Format for model comparison experiments.
            pipes_and_params[pipe_label] = (Pipeline(pipe_elem), param_space)
    return pipes_and_params


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
