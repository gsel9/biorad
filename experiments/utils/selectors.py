# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
Wrapped feature selection algorithms providing API compatible with model
comparison framework.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import mifs

import numpy as np
import pandas as pd

import utils

from scipy import stats
from ReliefF import ReliefF
from sklearn import feature_selection

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector

from sklearn.base import BaseEstimator, TransformerMixin


class BaseSelector:

    def __init__(self, z_scoring, error_handling='return_all'):

        self.z_scoring = z_scoring
        self.error_handling = error_handling

        # Apply Z-score transformation to data.
        if self.z_scoring:
            self._scaler = StandardScaler()
        else:
            self._scaler = None

    @property
    def support(self):

        if self._support is None:
            return

        return np.array(self._support, dtype=int)

    # TODO:
    @staticmethod
    def _check_X_y(X, y):

        return X, y


class PermutationSelection(BaseSelector, BaseEstimator, TransformerMixin):
    """Perform feature selection by feature permutation importance.

    Args:
        error_handling (str, array-like):
        score_func (function):
        model ():
        num_rounds (int): The number of permutations per feature.
        random_state (int):

    """

    def __init__(
        self,
        score_func,
        num_rounds,
        test_size,
        model,
        hparams=None,
        z_scoring=True,
        error_handling='return_all',
        random_state=None,
    ):

        super().__init__(z_scoring, error_handling)

        self.score_func = score_func
        self.num_rounds = num_rounds
        self.test_size = test_size
        self.model = model
        self.hparams = hparams
        self.z_scoring = z_scoring
        self.random_state = random_state

        self.rgen = np.random.RandomState(self.random_state)

        # Set model hyperparameters.
        if self.hparams is not None:
            self.model.set_params(**self.hparams)

        self._support = None

    def fit(self, X, y):

        self.X, self.y = self._check_X_y(X, y)

        # Perform train-test splitting.
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        # Perform Z-score transformation.
        if self.z_scoring and self._scaler is not None:
            X_train = self._scaler.fit_transform(X_train)
            X_test = self._scaler.transform(X_test)

        # NB: Error handling.
        try:
            self.model.fit(X_train, y_train)
        except:
            pass

        avg_imp = self.feature_permutation_importance(X_test, y_test)
        # Return features contributing to model performance as support.
        _support = np.where(avg_imp > 0)
        # Formatting and error handling.
        self.support = utils.formatting.check_support(_support, self.X)

        return self

    def transform(self):

        return self.support.formatting.check_subset(self.X[:, self.support])

    def _feature_permutation_importance(self, X, y):
        # Returns average feature permutation importance.

        # Baseline performance.
        baseline = self.score_func(y, self.model.predict(X))

        _, num_features = np.shape(X)
        importance = np.zeros(num_features, dtype=float)
        for round_idx in range(self.num_rounds):
            for col_idx in range(num_features):
                # Store original feature permutation.
                x_orig = X[:, col_idx].copy()
                # Break association between x and y by random permutation.
                self.rgen.shuffle(X[:, col_idx])
                new_score = self.score_func(y, self.model.predict(X))
                # Reinsert original feature prior to permutation.
                X[:, col_idx] = x_orig
                # Feature is likely important if new score < baseline.
                importance[col_idx] += baseline - new_score

        return importance / num_rounds


class WilcoxonSelection(BaseSelector, BaseEstimator, TransformerMixin):
    """Perform feature selection by Wilcoxon signed-rank test.

    Args:
        bf_correction (bool): Determine to apply Bonferroni correction for
            multiple testing.

    """

    def __init__(
        self,
        z_scoring,
        thresh=0.05,
        bf_correction=True,
        error_handling='return_all'
    ):

        super().__init__(z_scoring, error_handling)

        self.thresh = thresh
        self.z_scoring = z_scoring
        self.bf_correction = bf_correction

    def fit(self, X, y=None):

        self.X, self.y = self._check_X_y(X, y)

        # Perform Z-score transformation.
        if self.z_scoring and self._scaler is not None:
            self.X = self._scaler.fit_transform(self.X)

        _support = self.wilcoxon_signed_rank()
        # Formatting and error handling.
        self.support = utils.formatting.check_support(_support, self.X)

        return self

    def transform(self):

        return self.support.formatting.check_subset(self.X[:, self.support])

    def wilcoxon_signed_rank(self):
        """The Wilcoxon signed-rank test to determine if two dependent samples were
        selected from populations having the same distribution. Includes

        H0: The distribution of the differences x - y is symmetric about zero.
        H1: The distribution of the differences x - y is not symmetric about zero.

        Args:
            X (array-like): Predictor observations.
            y (array-like): Ground truths.
            thresh (float):

        Returns:
            (numpy.ndarray): Support indicators.

        """
        _, ncols = np.shape(self.X)

        support, pvals = [], []
        # Apply Bonferroni correction.
        if bf_correction:
            for num in range(ncols):
                # If p-value > thresh: same distribution.
                _, pval = stats.wilcoxon(self.X[:, num], self.y)
                    if pval <= self.thresh / ncols:
                        support.append(num)
        else:
            for num in range(ncols):
                # If p-value > thresh: same distribution.
                _, pval = stats.wilcoxon(self.X[:, num], self.y)
                    if pval <= self.thresh:
                        support.append(num)

        return np.array(support, dtype=int)


class ReliefFSelection(BaseSelector, BaseEstimator, TransformerMixin):

    def __init__(
        self,
        z_scoring,
        num_neighbors,
        num_features,
        error_handling='return_all'
    ):

        super().__init__(z_scoring, error_handling)

        self.z_scoring = z_scoring
        self.num_neighbors = num_neighbors
        self.num_features = num_features

    def fit(self, X, y=None):

        self.X, self.y = self._check_X_y(X, y)

        # Perform Z-score transformation.
        if self.z_scoring and self._scaler is not None:
            self.X = self._scaler.fit_transform(self.X)

        selector = ReliefF(n_neighbors=self.num_neighbors)
        selector.fit(self.X, self.y)

        _support = selector.top_features[:self.num_features]

        self.support = utils.formatting.check_support(_support, self.X)

        return self

    def transform(self):

        return self.support.formatting.check_subset(self.X[:, self.support])


class MRMRSelection(BaseSelector, BaseEstimator, TransformerMixin):
    """Perform feature selection with the minimum redundancy maximum relevancy
    algortihm.

    Args:
        k (int):
        num_features (): If `auto`, the number of is determined based on the
            amount of mutual information the previously selected features share
            with the target classes.

    """

    # NOTE:
    # * Cloned from: https://github.com/danielhomola/mifs
    # * Requirements: TODO

    def __init__(
        self,
        z_scoring,
        k,
        num_features,
        error_handling='return_all'
    ):

        super().__init__(z_scoring, error_handling)

        self.z_scoring = z_scoring
        self.k = k
        self.num_features = num_features

    def fit(self, X, y=None):

        self.X, self.y = self._check_X_y(X, y)

        # Perform Z-score transformation.
        if self.z_scoring and self._scaler is not None:
            self.X = self._scaler.fit_transform(self.X)

        # If 'auto': n_features is determined based on the amount of mutual
        # information the previously selected features share with y.
        selector = mifs.MutualInformationFeatureSelector(
            method='MRMR',
            k=self.k,
            n_features=self.num_features,
            categorical=True,
        )
        selector.fit(X_train_std, y_train)

        _support = np.where(selector.support_ == True)

        self.support = utils.formatting.check_support(_support, self.X)

        return self

    def transform(self):

        return self.support.formatting.check_subset(self.X[:, self.support])
