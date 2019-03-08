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

from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin

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
        # NOTE: Sanity check breaks if error handling returns NaN.
        if self.error_handling != 'nan':
            assert np.ndim(support) == 1

        return support

    def transform(self, X):
        """Extract the selected feature subset from the predictor matrix.

        Args:
            (array-like): Original predictor matrix.

        Returns:
            (array-like): The predictor subset matrix.

        """
        if self.support is np.nan:
            return X

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

    def fit(self, X, y=None, **kwargs):
        """Train classifier with optional sequential feature selection to
        reduce in the input feature space.

        """
        self._check_params(X, y)
        if self.with_selection:
            model = deepcopy(self.model)
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

        _, ncols = np.shape(X)

        # NOTE: Limited to the number of selected features from previous steps.
        if 'n_components' in self.get_params():
            if self.model.n_components > ncols:
                self.model.n_components = int(ncols - 1)
            elif self.model.n_components < 1:
                self.model.n_components = 1
            else:
                raise RuntimeError('Invalid value of columns, {ncols}, in X')

        # NOTE: Limited to the number of selected features from previous steps.
        if self.num_features is None:
            return
        elif self.num_features < 1:
            self.num_features = 1
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self
