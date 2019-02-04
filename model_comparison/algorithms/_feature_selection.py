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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split





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


# NOTE:
# * Cloned from: https://github.com/danielhomola/mifs
# * Use conda to install bottleneck V1.2.1 and pip to install local mifs clone.
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

    # ERROR: All NaN-slices encountered.
    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)
        # Shifting to all positive values.
        X = X + abs(np.min(X)) + 1
        # Hyperparameter adjustments.
        self._check_params(y)
        # Fallback error handling mechanism.
        #try:
        selector = mifs.MutualInformationFeatureSelector(
            method='MRMR',
            k=int(self.k),
            n_features=int(self.num_features),
            categorical=True,
        )
        selector.fit(X, y)
        # Extract features from the mask array of selected features.
        _support = np.squeeze(np.where(selector.support_))
            # Sanity check.
        #except:
        #    warnings.warn('Failed support with {}.'.format(self.__name__))
        #    _support = []
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


class FeatureScreening(BaseSelector):

    def __init__(
        self,
        chi2_num_features=None,
        f_classif_num_features=None,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.chi2_num_features = chi2_num_features
        self.f_classif_num_features = f_classif_num_features

        # NOTE: Attributes set with instance.
        self.support = None

    @property
    def num_sel_features(self):
        """Returns the size of the reduced feature set."""

        return np.size(self.support)

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        return check_X_y(X, y)

    # NB: Z-score transform converts categorical features to continous.
    # Current best: F-ANOVA
    def fit(self, X, y=None, **kwargs):

        # QUESTIONS:
        # * Not do any form of feature screening? How to justify doing an
        #   initial feature selection prior to modelling based on an algorithm
        #   unknown if selects the optimal features wrt. a specific procedure?

        # TODO:
        # * Try modeling without prior feature selection.
        # * Can justify use of e.g. F-ANOVA or MRMR prior to modeling procedure?

        X, y = self._check_X_y(X, y)

        #mut_infos = mutual_info_classif(X, y, discrete_features=False)
        #_support = np.argsort(mut_infos)[::-1][:self.chi2_num_features]

        # TODO:
        #selector = mifs.MutualInformationFeatureSelector(
        #    method='MRMR',
        #    k=self.f_classif_num_features,
        #    n_features='auto',
        #    categorical=False,
        #)
        #selector.fit(X, y)
        # Extract features from the mask array of selected features.
        #_support = np.squeeze(np.where(selector.support_))

        # TODO:
        #selector = DGUFS(
        #    num_features=self.chi2_num_features,
        #    num_clusters=self.f_classif_num_features,
        #    alpha=0.5,
        #    beta=0.9,
        #    max_iter=50
        #)
        #selector.fit(X)
        #_support = selector.support

        # Assumes normally distributed (Z-score transformation shifts feature
        # distributions towards Gaussian).
        #selector = SelectKBest(f_classif, k=self.f_classif_num_features)
        #selector.fit(X, y)
        #_support = np.array(selector.get_support(indices=True), dtype=int)

        #self.support = self.check_support(_support, X)
        self.support = np.arange(X.shape[1], dtype=int)

        return self

    @staticmethod
    def _categorical_continous(X, thresh=5):

        _, ncols = np.shape(X)

        categorical, continous = [], []
        for col_num in range(ncols):
            if len(np.unique(X[:, col_num])) < thresh:
                categorical.append(col_num)
            else:
                continous.append(col_num)

        return X[:, categorical], X[:, continous]

    def _check_params(self, X_cat, X_cont):
        # Parapmeter adjustments.
        if self.chi2_num_features >= X_cat.shape[1]:
            self.chi2_num_features = X_cat.shape[1] - 1
        elif self.chi2_num_features < 2:
            self.chi2_num_features = 2

        if self.f_classif_num_features < 2:
            self.f_classif_num_features = 2
        if self.f_classif_num_features >= X_cont.shape[1]:
            self.f_classif_num_features = X_cont.shape[1] - 1

        return self


if __name__ == '__main__':

    # TODO: Write test on selecting exactly num features.
    pass
