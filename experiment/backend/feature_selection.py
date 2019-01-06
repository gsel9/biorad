# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
Wrapped feature selection algorithms providing API compatible with model
comparison framework and scikti-learn Pipeline objects.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import mifs

import numpy as np
import pandas as pd

from scipy import stats
from ReliefF import ReliefF

from sklearn.utils import check_X_y
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# TransformerMixin incorporates fit_transform().
# BaseEstimator provides grid-searchable hyperparameters.
from sklearn.base import BaseEstimator, TransformerMixin


class BaseSelector:
    """Base representation of a feature selection algorithm.

    Args:
        error_handling (str, array-like): Determines feature selection error
            handling mechanism.

    """

    VALID_ERROR_MECHANISMS = ['return_all', 'return_NaN']

    def __init__(self, error_handling='return_all'):

        self.error_handling = error_handling

        self.X = None
        self.y = None

        if not self.error_handling in self.VALID_ERROR_MECHANISMS:
            raise ValueError('Invalid error handling mechanism {}'
                             ''.format(self.error_handling))

    @property
    def support(self):
        """Returns numpy.ndarray of feature support indicators."""

        if self._support is None:
            return

        return np.array(self._support, dtype=int)

    @staticmethod
    def check_subset(X):
        """Formatting and type checking of feature subset.

        Args:
            X (array-like):

        Returns:
            (array-like): Formatted feature subset.

        """
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

    @staticmethod
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
            if self.error_handling == 'return_all':
                support = np.arange(X.shape[1], dtype=int)
            elif self.error_handling == 'return_NaN':
                support = np.nan
            else:
                pass
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


class PermutationSelection(BaseSelector, BaseEstimator, TransformerMixin):
    """Perform feature selection by feature permutation importance.

    Args:
        error_handling (str, array-like):
        score_func (function):
        model (): The wrapped learning algorithm.
        model_params (dict): The model hyperparameter configuration.
        num_rounds (int): The number of permutations per feature.
        random_state (int):

    """

    def __init__(
        self,
        score_func=None,
        num_rounds=10,
        test_size=None,
        model=None,
        model_params=None,
        error_handling='return_all',
        random_state=None,
    ):

        super().__init__(error_handling)

        self.score_func = score_func
        self.num_rounds = num_rounds
        self.test_size = test_size
        self.model = model
        self.model_params = model_params
        self.random_state = random_state

        self.rgen = np.random.RandomState(self.random_state)

        # Set model hyperparameters.
        if self.model_params is not None:
            self.model.set_params(**self.model_params)
        # If stochastic algorithm.
        try:
            self.model.random_state = self.random_state
        except:
            pass

        self._support = None

    def __name__(self):

        return 'PermutationSelection'

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        return check_X_y(X, y)

    def fit(self, X, y):

        self.X, self.y = self._check_X_y(X, y)

        # Perform train-test splitting.
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        # Relying on error handling mechanism.
        try:
            selector.fit(X_train, y_train)
        except:
            return self.check_support([])

        avg_imp = self.feature_permutation_importance(X_test, y_test)
        # Return features contributing to model performance as support.
        self.support = self.check_support(np.where(avg_imp > 0), self.X)

        return self

    def transform(self):

        return self.check_subset(self.X[:, self.support])

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
        thresh=0.05,
        bf_correction=True,
        error_handling='return_all'
    ):

        super().__init__(error_handling)

        self.thresh = thresh
        self.bf_correction = bf_correction

    def __name__(self):

        return 'WilcoxonSelection'

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        return check_X_y(X, y)

    def fit(self, X, y=None):

        self.X, self.y = self._check_X_y(X, y)

        # Formatting and error handling.
        self.support = self.check_support(self.wilcoxon_signed_rank(), self.X)

        return self

    def transform(self):

        return self.check_subset(self.X[:, self.support])

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


# pip install ReliefF
class ReliefFSelection(BaseSelector, BaseEstimator, TransformerMixin):
    """

    The algorithm is notably sensitive to feature interactions [1], [2]. It is
    recommended that each feature is scaled to the interval [0 1].

    Args:
        num_neighbors (int)): Controls the locality of the estimates. The
            proposed default value is ten [3], [4].
        num_features (int)

    References:
        [1]: Kira, Kenji and Rendell, Larry (1992). The Feature Selection
             Problem: Traditional Methods and a New Algorithm. AAAI-92
             Proceedings.
        [2]: Kira, Kenji and Rendell, Larry (1992) A Practical Approach to
             Feature Selection, Proceedings of the Ninth International Workshop
             on Machine Learning, p249-256.
        [3]: Kononenko, I.: 1994, ‘Estimating  attributes:  analysis  and
             extensions  of  Relief’. In:  L. De Raedt and F. Bergadano (eds.):
             Machine Learning: ECML-94. pp. 171–182, Springer Verlag.
        [4]: M. Robnik-Sikonja, Kononenko, I.: 1994,2003, ‘Theoretical and
             Empirical Analysis of ReliefF and RReliefF‘. In: Machine Learning
             Journal 53, p23-69.

    """

    def __init__(
        self,
        num_neighbors=10,
        num_features=None,
        error_handling='return_all'
    ):

        super().__init__(error_handling)

        self.num_neighbors = num_neighbors
        self.num_features = num_features

        self._scaler = MinMaxScaler()

    def __name__(self):

        return 'ReliefFSelection'

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        X, y = check_X_y(X, y)
        # Scaling to [0, 1] range as recommended for this algorithm.
        X = self._scaler.fit_transform(X)

        return X, y

    def fit(self, X, y=None):

        # NOTE: Includes scaling to [0, 1] range.
        self.X, self.y = self._check_X_y(X, y)

        selector = ReliefF(n_neighbors=self.num_neighbors)
        # Rely on error handling mechanism.
        try:
            selector.fit(self.X, self.y)
        except:
            return self.check_support([])

        _support = selector.top_features[:self.num_features]

        self.support = self.check_support(_support, self.X)

        return self

    def transform(self):

        return self.check_subset(self.X[:, self.support])


# NOTE:
# * Cloned from: https://github.com/danielhomola/mifs
# * Use conda to install bottleneck=1.2.1 and pip to install local mifs clone.
class MRMRSelection(BaseSelector, BaseEstimator, TransformerMixin):
    """Perform feature selection with the minimum redundancy maximum relevancy
    algortihm.

    Args:
        k (int):
        num_features (): If `auto`, the number of is determined based on the
            amount of mutual information the previously selected features share
            with the target classes.
        error_handling (str, {`return_all`, `return_NaN`}):

    """

    def __init__(
        self,
        k=None,
        num_features=None,
        error_handling='return_all'
    ):

        super().__init__(error_handling)

        self.k = k
        self.num_features = num_features

    def __name__(self):

        return 'MRMRSelection'

    def fit(self, X, y=None):

        self.X, self.y = self._check_X_y(X, y)

        # If 'auto': n_features is determined based on the amount of mutual
        # information the previously selected features share with y.
        selector = mifs.MutualInformationFeatureSelector(
            method='MRMR',
            k=self.k,
            n_features=self.num_features,
            categorical=True,
        )
        # Rely on error handling mechanism.
        try:
            selector.fit(self.X, self.y)
        except:
            return self.check_support([])

        self.support = self.check_support(
            np.where(selector.support_ == True), self.X
        )
        return self

    def transform(self):

        return self.check_subset(self.X[:, self.support])


if __name__ == '__main__':

    pass
