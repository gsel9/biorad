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


import mifs
import warnings

import numpy as np
import pandas as pd

from scipy import stats
from ReliefF import ReliefF

from sklearn.utils import check_X_y
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from sklearn.base import BaseEstimator, TransformerMixin


class BaseSelector(BaseEstimator, TransformerMixin):
    """Representation of a feature selection algorithm.

    Args:
        error_handling (str, array-like): Determines feature selection error
            handling mechanism.

    """

    VALID_ERROR_MECHANISMS = ['all', 'nan']

    def __init__(self, error_handling='nan'):

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

        # Recommendation from sklearn: Reshape data with array.reshape(-1, 1)
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

        # Method is shared by all subclasses as a required pipeline signature.
        if self.support is np.nan:
            return X
        else:
            return self.check_subset(X[:, self.support])


class PermutationSelection(BaseSelector):
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
        model=None,
        test_size=None,
        score_func=None,
        num_rounds=None,
        error_handling='all',
        random_state=None
    ):

        super().__init__(error_handling)

        self.model = model
        self.test_size = test_size
        self.num_rounds = num_rounds
        self.score_func = score_func
        self.random_state = random_state

        # NOTE: Attributes set with instanceself.
        self.rgen = None
        self.support = None
        self.all_params = None

    def __name__(self):

        return 'PermutationSelection'

    @property
    def num_sel_features(self):
        """Returns the size of the reduced feature set."""

        return np.size(self.support)

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        return check_X_y(X, y)

    # Are procedure params passed along? Required to sett random state?
    def set_params(self, **params):
        """Update model hyperparameters."""

        if 'random_state' in params.keys():
            self.random_state = params['random_state']
        if 'num_features' in params.keys():
            self.num_features = params['num_features']

        """
        # Separate model and procedure parameters:
        model_params, procedure_params = {}, {}
        for key in params.keys():
            if key in self.model.get_params():
                model_params[key] = params[key]
            else:
                procedure_params[key] = params[key]

        # Set model psecific params and store all procedure params.
        self.model.set_params(**model_params)
        """
        self.model.set_params(**params)

        return self

    def get_params(self, deep=True):
        """Return model hyperparameters."""

        return self.model.get_params(deep=deep)

    def fit(self, X, y, **kwargs):

        X, y = self._check_X_y(X, y)

        if self.rgen is None:
            self.rgen = np.random.RandomState(self.random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        try:
            self.model.fit(X_train, y_train)
            avg_imp = self._feature_permutation_importance(X_test, y_test)
            # Need to retain features with zero importance to maintain level of
            # baseline score.
            _support = np.where(avg_imp >= 0)
        except:
            warnings.warn('Failed selecting features with {}'
                          ''.format(self.__name__))
            _support = []

        self.support = self.check_support(_support, X)

        return self

    def _feature_permutation_importance(self, X, y):
        # Returns average feature permutation importance.

        # Baseline performance.
        baseline = self.score_func(y, self.model.predict(X))

        _, num_features = np.shape(X)
        importance = np.zeros(num_features, dtype=float)
        for round_idx in range(self.num_rounds):
            for col_idx in range(num_features):
                # Store original feature permutation.
                X_orig = X[:, col_idx].copy()
                # Break association between x and y by random permutation.
                self.rgen.shuffle(X[:, col_idx])
                new_score = self.score_func(y, self.model.predict(X))
                # Reinsert original feature prior to permutation.
                X[:, col_idx] = X_orig
                # Feature is likely important if new score < baseline.
                importance[col_idx] += baseline - new_score
        return importance / self.num_rounds


class WilcoxonSelection(BaseSelector):
    """Perform feature selection by Wilcoxon signed-rank test.

    Args:
        bf_correction (bool): Determine to apply Bonferroni correction for
            multiple testing.

    """

    def __init__(
        self,
        thresh=0.05,
        bf_correction=True,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.thresh = thresh
        self.bf_correction = bf_correction

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):

        return 'WilcoxonSelection'

    @property
    def num_sel_features(self):
        """Returns the size of the reduced feature set."""

        return np.size(self.support)

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)
        try:
            _support = self.wilcoxon_signed_rank(X, y)
        except:
            warnings.warn('Failed support with {}.'.format(self.__name__))
            _support = []

        self.support = self.check_support(_support, X)

        return self

    def wilcoxon_signed_rank(self, X, y):
        """The Wilcoxon signed-rank test to determine if two dependent samples were
        selected from populations having the same distribution. Includes

        H0: The distribution of the differences x - y is symmetric about zero.
        H1: The distribution of the differences x - y is not symmetric about zero.

        If p-value > thresh => same distribution.

        Args:
            X (array-like): Predictor observations.
            y (array-like): Ground truths.
            thresh (float):

        Returns:
            (numpy.ndarray): Support indicators.

        """
        _, ncols = np.shape(X)

        support = []
        if self.bf_correction:
            for num in range(ncols):
                _, pval = stats.wilcoxon(X[:, num], y)
                if pval <= self.thresh / ncols:
                    support.append(num)
        else:
            for num in range(ncols):
                _, pval = stats.wilcoxon(X[:, num], y)
                if pval <= self.thresh:
                    support.append(num)

        return np.array(support, dtype=int)


# pip install ReliefF
class ReliefFSelection(BaseSelector):
    """

    Args:
        num_neighbors (int)): Controls the locality of the estimates. The
            recommended default value is ten [3], [4].
        num_features (int)

    Note:
    - The algorithm is notably sensitive to feature interactions [1], [2].
    - It is recommended that each feature is scaled to the interval [0, 1].

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
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.num_neighbors = num_neighbors
        self.num_features = num_features

        # NOTE: Attributes set with instance.
        self.support = None
        self.scaler = None

    def __name__(self):

        return 'ReliefFSelection'

    @property
    def num_sel_features(self):
        """Returns the size of the reduced feature set."""

        return np.size(self.support)

    def _check_X_y(self, X, y):
        # A wrapper around sklearn formatter.

        X, y = check_X_y(X, y)
        # Scaling to [0, 1] range as recommended for this algorithm.
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)

        return X, y

    def fit(self, X, y=None, **kwargs):

        # NOTE: Includes scaling to [0, 1] range.
        X, y = self._check_X_y(X, y)
        # Hyperparameter adjustments.
        self._check_params(X)
        # Fallback error handling mechanism.
        try:
            selector = ReliefF(n_neighbors=int(self.num_neighbors))
            selector.fit(X, y)
            # Select the predefined number of features from ReliefF ranking.
            _support = selector.top_features[:int(self.num_features)]
            # Sanity check.
            assert len(_support) == self.num_features
        except:
            warnings.warn('Failed support with {}.'.format(self.__name__))
            _support = []

        self.support = self.check_support(_support, X)

        return self

    def _check_params(self, X):

        # Satisfying check in sklearn KDTree (binary tree).
        nrows, _ = np.shape(X)
        if self.num_neighbors > nrows:
            self.num_neighbors = nrows - 1

        return self


# NOTE:
# * Cloned from: https://github.com/danielhomola/mifs
# * Use conda to install bottleneck=1.2.1 and pip to install local mifs clone.
class MRMRSelection(BaseSelector):
    """Perform feature selection with the minimum redundancy maximum relevancy
    algortihm.

    Args:
        k (int): Note that k > 0, but must be smaller than the smallest number
            of observations for each individual class.
        num_features (): If `auto`, the number of is determined based on the
            amount of mutual information the previously selected features share
            with the target classes.
        error_handling (str, {`return_all`, `return_NaN`}):

    """

    def __init__(self, k=1, num_features=None, error_handling='all'):

        super().__init__(error_handling)

        self.k = k
        self.num_features = num_features

        # NOTE: Attributes set with instance.
        self.support = None

    def __name__(self):

        return 'MRMRSelection'

    @property
    def num_sel_features(self):
        """Returns the size of the reduced feature set."""

        return np.size(self.support)

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)
        # Hyperparameter adjustments.
        self._check_params(y)
        # Fallback error handling mechanism.
        try:
            selector = mifs.MutualInformationFeatureSelector(
                method='MRMR',
                k=int(self.k),
                n_features=int(self.num_features),
                categorical=True,
            )
            selector.fit(X, y)
            # Extract features from the mask array of selected features.
            _support = np.where(selector.support_ == True)
            # Sanity check.
            assert len(_support) == self.num_features
        except:
            warnings.warn('Failed support with {}.'.format(self.__name__))
            _support = []

        self.support = self.check_support(_support, X)

        return self

    def _check_params(self, y):

        # From MIFS source code: k > 0, but smaller than the
        # smallest class.
        min_class_count = np.min(np.bincount(y))
        if self.k > min_class_count:
            self.k = min_class_count
        if self.k < 1:
            self.k = 1

        return self


class Screening(BaseSelector):
    """Based on the Gain equation propsed by Vallieres (2015)."""

    def __init__(
        self,
        var_thresh=1e-4,
        mi_thresh=0.01,
        alpha=0.05,
        error_handling='all',
        random_state=None
    ):

        super().__init__(error_handling)

        self.var_thresh = var_thresh
        self.mi_thresh = mi_thresh
        self.alpha = alpha
        self.random_state = random_state

        # NOTE: Attributes set with instance.
        self.support = None

    def __name__(self):

        return 'GainSelection'

    # NOTE:
    # * Requires target vector. Should build on other basis than
    #   transformer mix in?
    # * Vallieres Gain equation: https://github.com/mvallieres/radiomics/blob/master/MultivariableModeling/featureSetReduction.m
    def fit(self, X, y, **kwargs):
        """

        If p-value > thresh:
            Samples are uncorrelated (fail to reject H0).
        else:
            Samples are correlated (reject H0).

        """
        X, y = self._check_X_y(X, y)
        # Checking for correlations is the most tedious task.
        try:
            _support = self._filter_low_variance(X[:, _support])
            _support = self._filter_mutual_info(X[:, _support])
            _support = self._filter_correalted(X[:, _support])
        except:
            warnings.warn('Failed support with {}.'.format(self.__name__))
            _support = []

        self.support = self.check_support(_support, X)

        return self

    def _filter_low_variance(self, X):
        # Remove features with mutial information.
        minfo = mutual_info_classif(X, y)
        return np.squeeze(np.where(minfo > self.mi_thresh))

    def _filter_mutual_info(self, X):
        # Remove features with low variance.
        var_filter = VarianceThreshold(threshold=self.var_thresh)
        var_filter.fit(X)
        return var_filter.get_support(get_indices=True)

    def _filter_correalted(self, X):
        # Remove features without significant correlation to target.

        # Separate into ordinal-oridinal/nominal-ordinal
        pass


if __name__ == '__main__':

    # TODO: Write test on selecting exactly num features.
    pass
