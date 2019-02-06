# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
Feature selection algorithms providing compatible with model comparison
framework, scikti-learn `Pipeline` objects and hyperopt `fmin` function.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import warnings

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import MetaEstimatorMixin


class BaseSelector(BaseEstimator, TransformerMixin):
    """Representation of a feature selection algorithm.

    Args:
        error_handling (str, array-like): Determines feature selection error
            handling mechanism.

    """

    VALID_ERROR_MECHANISMS = ['all', 'nan']

    def __init__(self, error_handling='nan'):

        super().__init__()

        self.error_handling = error_handling

        if not self.error_handling in self.VALID_ERROR_MECHANISMS:
            raise ValueError('Invalid error handling mechanism {}'
                             ''.format(self.error_handling))

        # NOTE: Attribute set with instance.
        self.support = None

    @staticmethod
    def check_subset(X):
        """Formatting and type checking of feature subset.

        Args:
            X (array-like):

        Returns:
            (array-like): Formatted feature subset.

        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)

        if np.ndim(X) > 2:
            warnings.warn('Squeezeing X with shape: {}'.format(np.shape(X)))
            X = np.squeeze(X)

        # Scikit-learn recommendation: Reshape data with array.reshape(-1, 1)
        # if data has a single feature, or array.reshape(1, -1) if it contains
        # a single sample.
        if np.ndim(X) < 2:
            nrows, ncols = np.shape(X)
            if nrows == 1:
                warnings.warn('Reshaping X with shape: {}'.format(np.shape(X)))
                X = X.reshape(1, -1)
            if ncols == 1:
                warnings.warn('Reshaping X with shape: {}'.format(np.shape(X)))
                X = X.reshape(-1, 1)

        if not np.ndim(X) == 2:
            raise RuntimeError('Error X ndim {}'.format(np.ndim(X_train)))

        return X

    def check_support(self, support, X):
        """Formatting of feature subset indicators.

        Args:
            support (array-like): Feature subset indicators.
            X (array-like): Original feature matrix.

        Returns:
            (array-like): Formatted indicators of feature subset.

        """
        if not isinstance(support, np.ndarray):
            support = np.array(support, dtype=int)

        # Check if support is empty. Fall back to error mechanism if so.
        if np.size(support) < 1:
            warnings.warn('Error mechanism: {}'.format(self.error_handling))
            if self.error_handling == 'all':
                support = np.arange(X.shape[1], dtype=int)
            elif self.error_handling == 'nan':
                return np.nan
            else:
                raise RuntimeError('Cannot format support: {}'.format(support))

        # Ensure correct support dimensionality.
        if np.ndim(support) > 1:
            support = np.squeeze(support)
        if np.ndim(support) < 1:
            support = support[np.newaxis]

        # Sanity check (breaks if error handling is `return NaN`).
        if self.error_handling == 'all':
            assert np.ndim(support) == 1

        return support

    def transform(self, X):
        """

        """

        # Method is shared by all subclasses as a required pipeline signature.
        if self.support is np.nan:
            return X
        else:
            return self.check_subset(X[:, self.support])


class BaseEstimator(BaseEstimator, MetaEstimatorMixin):
    """A wrapper for scikit-learn estimators. Enables intermediate
    configuration of the wrapped classifier model between pipeline steps.

    In particular, configrations are necessary with respect to:
    - Adjusting the number of components to keep in decomposition methods
      should the previous pipeline transformer reduce the feature set resulting
      in an expected number of cmoponents exceed the number of available
      features.

    Args:
        model ():

    """

    def __init__(self, mode=None, model=None):

        super().__init__()

        self._mode = mode
        self._model = model

    def set_params(self, **params):
        """Update estimator hyperparamter configuration.

        Kwargs:
            params (dict): Hyperparameter settings.

        """
        self._model.set_params(**params)

        return self

    def get_params(self, deep=True):
        """Returns hyperparameter configurations.

        """
        return self._model.get_params(deep=deep)

    def fit(self, X, y=None, **kwargs):
        """

        """
        self._check_config(X)

        self._model.fit(X, y, **kwargs)

        return self

    def predict(self, X):
        """

        """
        y_pred = np.squeeze(self._model.predict(X))

        if self._mode == 'classification':
            return np.array(y_pred, dtype=int)
        elif self._mode == 'regression':
            return np.array(y_pred, dtype=float)
        else:
            raise ValueError('Invalid estimator mode: {}'.format(self._mode))

    def _check_config(self, X, y=None, **kwargs):
        # Validate model configuration by updating hyperparameter settings.

        if 'n_components' in self.get_params():
            if self._model.n_components > X.shape[1]:
                self._model.n_components = X.shape[1]

        return self
