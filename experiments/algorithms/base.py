# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
Base classes for wrappers of feature selection and classification algorithms
ensuring unified API for model comparison experiments.
"""

__author__ = 'Severin Langberg'
__contact__ = 'langberg91@gmail.com'


import warnings

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.utils import check_X_y


class BaseSelector(BaseEstimator, TransformerMixin):
    """Representation of a feature selection algorithm.

    """

    def __init__(self):

        super().__init__()

        # NOTE: Attribute set with instance.
        self.support = None

    @staticmethod
    def check_support(support):
        """Formatting of feature subset indicators.

        Args:
            support (array-like): Feature subset indicators represented by either a
                boolean array or an array of integers referencing feature numbers.

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

    @staticmethod
    def check_X_subset(X):
        """Formatting and type checking of feature subset.

        Args:
            X (array-like):

        Returns:
            (array-like): Formatted feature subset.

        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)

        if np.ndim(X) > 2:
            warnings.warn(f'Squeezing X of shape {np.shape(X)}.')
            X = np.squeeze(X)

        # Sanity check.
        if np.ndim(X) != 2:
            raise RuntimeError(f'Invalid dim of X: {np.ndim(X)}.')

        # Scikit-learn recommendation: Reshape data with reshape(-1, 1) if data
        # has a single feature, or reshape(1, -1) if data has a single sample.
        if X.shape[0] == 1:
            warnings.warn(f'Reshaping X from {np.shape(X)}.')
            X = X.reshape(1, -1)
        if X.shape[1] == 1:
            warnings.warn(f'Reshaping X from {np.shape(X)}.')
            X = X.reshape(-1, 1)

        return X

    @staticmethod
    def check_X_y(X, y):
        # A wrapper around the sklearn formatter function.
        return check_X_y(X, y)

    def transform(self, X):
        """Extract the selected feature subset from the predictor matrix.

        Args:
            (array-like): Original predictor matrix.

        Returns:
            (array-like): The predictor subset matrix.

        """
        return self.check_X_subset(X[:, self.support])


class BaseClassifier(BaseEstimator, ClassifierMixin):
    """A wrapper for scikit-learn estimators. Enables intermediate
    configuration of the wrapped classifier model between pipeline steps.

    Args:
        model ():

    """

    def __init__(self, model=None, random_state=0):

        super().__init__()

        self.model = model
        self.random_state = random_state

        # NOTE: Threshold to binarize continous predictions.
        self.binary_thresh = None

    def set_params(self, **params):
        """Update estimator hyperparamter configuration.

        Kwargs:
            params (dict): Hyperparameter settings.

        """
        params = self.check_params(**params)
        self.model.set_params(**params)
        if hasattr(self.model, 'random_state'):
            self.model.random_state = self.random_state

        return self

    def get_params(self, deep=True):
        """Returns hyperparameter configurations.

        """
        return self.model.get_params(deep=deep)

    def fit(self, X, y=None, **kwargs):
        """Train classifier with optional sequential feature selection to
        reduce in the input feature space.

        """
        self.check_model_config(X, y)
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """Generate model prediction.

        """
        y_pred = np.squeeze(self.model.predict(X))
        if self.binary_thresh:
            return np.array(y_pred > self.binary_thresh, dtype=int)
        return np.array(y_pred, dtype=int)

    def check_model_config(self, X, y=None):
        """Validate model configuration."""

        _, num_cols = np.shape(X)
        # Check in number of components to use exceeds number of features.
        if 'n_components' in self.get_params():
            if self.model.n_components > num_cols:
                self.model.n_components = int(num_cols - 1)

        return self

    def check_params(self, **params):
        _params = {}
        for key, value in params.items():
            if value == 'none':
                _params[key] = None
            elif key == 'binarization':
                self.binary_thresh = value
            else:
                _params[key] = value

        return _params
