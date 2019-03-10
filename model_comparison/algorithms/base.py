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

from copy import copy

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin

# TEMP:
try:
    from . import sffs
except:
    import sffs


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
            warnings.warn('Squeezing X of shape {np.shape(X)}')
            X = np.squeeze(X)

        # Scikit-learn recommendation:
        # Reshape data with array.reshape(-1, 1) if data has a single feature,
        # or array.reshape(1, -1) if it contains a single sample.
        if np.ndim(X) < 2:
            nrows, ncols = np.shape(X)
            if nrows == 1:
                warnings.warn('Reshaping X from {np.shape(X)}')
                X = X.reshape(1, -1)
            if ncols == 1:
                warnings.warn('Reshaping X from {np.shape(X)}')
                X = X.reshape(-1, 1)

        if np.ndim(X) != 2:
            raise RuntimeError('Invalid dim of X: {}Error X ndim {np.ndim(X)}')

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

        # Ensure correct dimensionality of support.
        if np.ndim(support) > 1:
            support = np.squeeze(support)
        if np.ndim(support) < 1:
            support = support[np.newaxis]

        return support

    def transform(self, X):
        """Extract the selected feature subset from the predictor matrix.

        Args:
            (array-like): Original predictor matrix.

        Returns:
            (array-like): The predictor subset matrix.

        """
        return self.check_subset(X[:, self.support])


class BaseClassifier(BaseEstimator, ClassifierMixin):
    """A wrapper for scikit-learn estimators. Enables intermediate
    configuration of the wrapped classifier model between pipeline steps.

    Args:
        model ():

    """

    def __init__(
        self,
        model=None,
        with_selection: bool=None,
        scoring=None,
        cv: int=None,
        forward: bool=None,
        floating: bool=None,
    ):

        super().__init__()

        self.model = model
        self.with_selection = with_selection
        self.scoring = scoring
        self.cv = cv
        self.forward = forward
        self.floating = floating

        # NOTE: Attribute set with instance.
        self.support = None
        self.num_features = None

    def set_params(self, **params):
        """Update estimator hyperparamter configuration.

        Kwargs:
            params (dict): Hyperparameter settings.

        """
        params = self._check_config(params)
        self.model.set_params(**params)

        return self

    def get_params(self, deep=True):
        """Returns hyperparameter configurations.

        """
        return self.model.get_params(deep=deep)

    # TODO: Make a decorator for wrapper algorithms accepting a model.
    def fit(self, X, y=None, **kwargs):
        """Train classifier with optional sequential feature selection to
        reduce in the input feature space.

        """
        self._check_params(X, y)
        # TODO: Fix
        if self.with_selection:
            # NOTE: Cannot deepcopy Cython objects.
            model = copy(self.model)
            selector = sffs.SequentialFeatureSelector(
                estimator=model,
                k_features=self.num_features,
                forward=self.forward,
                floating=self.floating,
                scoring=self.scoring,
                cv=self.cv
            )
            selector.fit(X, y)
            self.support = np.array(selector.k_feature_idx_, dtype=int)
            # Check hyperparameter setup with the reduced feature set.
            self._check_params(X[:, self.support], y)
            self.model.fit(X[:, self.support], y, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)

        return self

    def predict(self, X):
        """Generate model prediction.

        """
        if self.with_selection:
            y_pred = np.squeeze(self.model.predict(X[:, self.support]))
        else:
            y_pred = np.squeeze(self.model.predict(X))

        return np.array(y_pred, dtype=int)

    def _check_config(self, params):
        # Validate model hyperparameter configuration settings for updating.
        _params = {}
        for key in params.keys():
            if params[key] is not None:
                if 'gamma' in key:
                    if params['gamma'] == 'value':
                        _params['gamma'] = params['gamma_value']
                    else:
                        _params['gamma'] = 'auto'
                elif 'num_features' in key:
                    self.num_features = params[key]
                else:
                    _params[key] = params[key]
            else:
                pass

        return _params

    def _check_params(self, X, y=None):
        # Validate model hyperparamter configuration for training.

        _, num_cols = np.shape(X)

        # TEMP: Used only in sequential selection. May be removed after
        # detaching.
        # NOTE: Adjust the number of features to select according to the
        # dimensionality of X.
        if self.num_features is None:
            return None

        if self.num_features < 1:
            self.num_features = 1
        elif self.num_features > num_cols:
            self.num_features = int(num_cols - 1)
        else:
            self.num_features = int(self.num_features)

        # NOTE: Adjust the number of components according to the
        # dimensionality of X.
        if 'n_components' in self.get_params():
            if self.model.n_components > num_cols:
                self.model.n_components = int(num_cols - 1)
            elif self.model.n_components < 1:
                self.model.n_components = 1
            else:
                self.model.n_components = int(num_cols - 1)

        return self
