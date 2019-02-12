# -*- coding: utf-8 -*-
#
# selectors.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

import mifs
import warnings

import numpy as np

from . import base
from copy import deepcopy

from ReliefF import ReliefF
from scipy.stats import ranksums
from skfeature.function.similarity_based.fisher_score import fisher_score

from sklearn.svm import SVC
from sklearn.utils import check_X_y
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter


SEED = 0


class SequentialSelection(base.BaseSelector):
    """

    """

    NAME = 'SequentialSelection'

    def __init__(
        self,
        model=None,
        model_name=None,
        num_features=None,
        scoring='roc_auc',
        cv=0,
        forward=True,
        floating=False,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.model = model
        self.model_name = model_name
        self.num_features = num_features
        self.scoring = scoring
        self.cv = cv
        self.forward = forward
        self.floating = floating

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):

        global SEED

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameter(num_features)

        return config

    def set_params(self, **params):
        """Update estimator hyperparamter configuration.

        Kwargs:
            params (dict): Hyperparameter settings.

        """

        self.num_features = params['num_features']

        return self

    def set_model_params(self, **params):
        params = self._check_config(params)
        self.model.set_params(**params)

        return self

    def _check_config(self, params):
        # Validate model configuration by updating hyperparameter settings.

        _params = {}
        for key in params.keys():
            if params[key] is not None:
                if 'gamma' in key:
                    if params['gamma'] == 'value':
                        _params['gamma'] = params['gamma_value']
                    else:
                        _params['gamma'] = 'auto'
                else:
                    _params[key] = params[key]
            else:
                pass

        return _params

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):
        X, y = self._check_X_y(X, y)

        self._check_params(X, y)

        #try:
        selector = SequentialFeatureSelector(
            estimator=self.model,
            k_features=self.num_features,
            forward=self.forward,
            floating=self.floating,
            scoring=self.scoring,
            cv=self.cv
        )
        selector.fit(X, y)
        _support = selector.k_feature_idx_
        #except:
        #    warnings.warn('Failed support with {}.'.format(self.__name__))
        #    _support = []

        self.support = self.check_support(_support, X)

        return self

    def _check_params(self, X, y):

        _, ncols = np.shape(X)
        if self.num_features < 1:
            self.num_features = int(self.num_features)
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self


class ANOVAFvalueSelection(base.BaseSelector):
    """

    """

    NAME = 'ANOVAFvalueSelection'

    def __init__(
        self,
        num_features=None,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.num_features = num_features

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):
        """Returns the ANOVA F-value hyperparameter configuration space."""

        global SEED

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameter(num_features)

        return config

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)

        self._check_params(X, y)
        #try:
        selector = SelectKBest(f_classif, k=self.num_features)
        selector.fit(X, y)
        _support = selector.get_support(indices=True)
        #except:
        #    warnings.warn('Failed support with {}.'.format(self.__name__))
        #    _support = []

        self.support = self.check_support(_support, X)

        return self

    def _check_params(self, X, y):

        _, ncols = np.shape(X)

        if self.num_features < 1:
            self.num_features = int(self.num_features)
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self


class FScoreSelection(base.BaseSelector):
    """

    """

    NAME = 'FScoreSelection'

    def __init__(
        self,
        num_features=None,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.num_features = num_features

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):
        """Returns the Fisher score hyperparameter configuration space."""

        global SEED

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameter(num_features)

        return config

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)

        def _fisher_score(X, y):

            scores = fisher_score(X, y)
            return np.argsort(scores, 0)[::-1]

        self._check_params(X, y)
        #try:
        _support = _fisher_score(X, y)[:self.num_features]
        #except:
        #    warnings.warn('Failed support with {}.'.format(self.__name__))
        #    _support = []

        self.support = self.check_support(_support, X)

        return self

    def _check_params(self, X, y):

        _, ncols = np.shape(X)

        if self.num_features < 1:
            self.num_features = int(self.num_features)
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self


class WilcoxonSelection(base.BaseSelector):
    """Perform feature selection by Wilcoxon rank sum test.

    Args:
        bf_correction (bool): Apply Bonferroni correction for
            multiple testing correction.

    """

    NAME = 'WilcoxonSelection'

    def __init__(
        self,
        num_features=None,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.num_features = num_features

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):
        """Returns the Wilcoxon selection hyperparameter configuration space.
        """

        global SEED

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameter(num_features)

        return config

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)
        #try:
        p_values = self.wilcoxon_rank_sum(X, y)
        _support = np.argsort(p_values)[:self.num_features]
        #except:
        #    warnings.warn('Failed support with {}.'.format(self.__name__))
        #    _support = []

        self.support = self.check_support(_support, X)

        return self

    def wilcoxon_rank_sum(self, X, y):
        """The Wilcoxon rank sum test to determine if two measurements are
        drawn from the same distribution.

        Args:
            X (array-like): Predictor matrix.
            y (array-like): Target variable.

        Returns:
            (numpy.ndarray): Support indicators.

        """
        _, ncols = np.shape(X)

        p_values = []
        for num in range(ncols):
            _, p_value = ranksums(X[:, num], y)
            p_values.append(num)

        return np.array(p_values, dtype=float)


# NOTE:
# * Cloned from: https://github.com/danielhomola/mifs
# * Use conda to install bottleneck V1.2.1 and pip to install local mifs clone.
class MRMRSelection(base.BaseSelector):
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

    NAME = 'MRMRSelection'

    def __init__(
        self,
        num_neighbors=None,
        num_features=None,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.num_neighbors = num_neighbors
        self.num_features = num_features

        # NOTE: Attributes set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):
        """Returns the MRMR hyperparameter configuration space."""

        global SEED

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=10, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

    def fit(self, X, y=None, **kwargs):
        """

        """
        # Ensures all elements of X > 0.
        X, y = self._check_X_y(X, y)

        self._check_params(X, y)
        #try:
        # NOTE: Categorical refers to the target variable data type.
        selector = mifs.MutualInformationFeatureSelector(
            method='MRMR',
            categorical=True,
            k=self.num_neighbors,
            n_features=self.num_features,
        )
        selector.fit(X, y)
        # Check for all NaNs.
        if np.all(np.isnan(selector.support_)):
            _support = []
        else:
            _support = np.squeeze(np.where(selector.support_))
        #except:
        #    warnings.warn('Failed support with {}.'.format(self.__name__))
        #    _support = []
        self.support = self.check_support(_support, X)

        return self

    def _check_params(self, X, y):

        # From MIFS source code: k > 0, but smaller than the
        # smallest class.
        min_class_count = np.min(np.bincount(y))
        if self.num_neighbors > min_class_count:
            self.num_neighbors = int(min_class_count)
        if self.num_neighbors < 1:
            self.num_neighbors = 1

        _, ncols = np.shape(X)

        if self.num_features < 1:
            self.num_features = int(self.num_features)
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        X, y = check_X_y(X, y)
        X_nonegative = X + np.abs(np.min(X)) + 1

        return X_nonegative, y



# pip install ReliefF from https://github.com/gitter-badger/ReliefF
class ReliefFSelection(base.BaseSelector):
    """

    Args:
        num_neighbors (int)): Controls the locality of the estimates. The
            recommended default value is ten [3], [4].
        num_features (int)

    Note:
    - The algorithm is notably sensitive to feature interactions [1], [2].
    - It is recommended that each feature is scaled to the interval [0, 1].

    Robnik-Sikonja and Kononenko (2003) showed that ReliefF’sestimates of
    informative attribute are deteriorating with increasing number of nearest
    neighbors in parity domain. Robnik-Sikonja and Kononenko also supports
    Dalaka et al., 2000 with ten neighbors.

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

    NAME = 'ReliefFSelection'

    def __init__(
        self,
        num_neighbors=None,
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

        return self.NAME

    @property
    def config_space(self):
        """Returns the ReliefF hyperparameter configuration space."""

        global SEED

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=10, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

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

        self._check_params(X, y)
        #try:
        selector = ReliefF(
            n_neighbors=self.num_neighbors,
            n_features_to_keep=self.num_features
        )
        selector.fit(X, y)
        _support = selector.top_features[:self.num_features]
        #except:
        #    warnings.warn('Failed to select support with {}'.format(self.NAME))
        #    _support = []
        self.support = self.check_support(_support, X)

        return self

    def _check_params(self, X, y):

        # Satisfying check in sklearn KDTree (binary tree).
        nrows, ncols = np.shape(X)
        if self.num_neighbors > nrows:
            self.num_neighbors = int(nrows - 1)
        else:
            self.num_neighbors = int(self.num_neighbors)

        if self.num_features < 1:
            self.num_features = int(self.num_features)
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self


class MutualInformationSelection(base.BaseSelector):

    NAME = 'MutualInformationSelection'

    def __init__(
        self,
        num_neighbors=None,
        num_features=None,
        random_state=None,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.num_features = num_features
        self.num_neighbors = num_neighbors
        self.random_state = random_state

        # NOTE: Attributes set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):
        """Returns the MI hyperparameter configuration space."""

        global SEED

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=10, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

    def fit(self, X, y, **kwargs):
        """

        """
        X, y = self._check_X_y(X, y)

        def _mutual_info_classif(X, y):

            return mutual_info_classif(
                X, y,
                n_neighbors=self.num_neighbors,
                random_state=self.random_state
            )

        self._check_params(X, y)
        #try:
        selector = SelectKBest(_mutual_info_classif, k=self.num_features)
        selector.fit(X, y)
        _support = selector.get_support(indices=True)
        #except:
        #    warnings.warn('Failed support with {}.'.format(self.__name__))
        #    _support = []
        self.support = self.check_support(_support, X)

        return self

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        return check_X_y(X, y)

    def _check_params(self, X, y):

        # Satisfying check in sklearn KDTree (binary tree).
        nrows, ncols = np.shape(X)
        if self.num_neighbors > nrows:
            self.num_neighbors = int(nrows - 1)
        else:
            self.num_neighbors = int(self.num_neighbors)

        if self.num_features < 1:
            self.num_features = int(self.num_features)
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self


class Chi2Selection(base.BaseSelector):

    NAME = 'Chi2Selection'

    def __init__(
        self,
        num_features=None,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.num_features = num_features

        # NOTE: Attributes set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):
        """Returns the MI hyperparameter configuration space."""

        global SEED

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y, **kwargs):
        """

        """
        # Ensures all elements of X > 0 for Chi2 test.
        X, y = self._check_X_y(X, y)

        self._check_params(X, y)
        #try:
        selector = SelectKBest(chi2, k=self.num_features)
        selector.fit(X, y)
        _support = selector.get_support(indices=True)
        #except:
        #    warnings.warn('Failed support with {}.'.format(self.__name__))
        #    _support = []
        self.support = self.check_support(_support, X)

        return self

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        X, y = check_X_y(X, y)
        X_nonegative = X + np.abs(np.min(X)) + 1

        return X_nonegative, y

    def _check_params(self, X, y):

        _, ncols = np.shape(X)

        if self.num_features < 1:
            self.num_features = int(self.num_features)
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self
