# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
Wrappers for feature selection algorithms ensuring unified API for model
comparison experiments.

Notes:
* Permutation importance is sensitive towards correlated features.
* ReliefF requires scaling of features to unit length.

"""

__author__ = 'Severin Langberg'
__contact__ = 'langberg91@gmail.com'

from mifs import MutualInformationFeatureSelector

import numpy as np

from scipy.stats import rankdata

from skrebate import ReliefF
from skrebate import MultiSURF

from sklearn.utils import check_X_y
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from skfeature.function.similarity_based.fisher_score import fisher_score

from . import base


class DummySelection:

    NAME = 'DummySelection'

    def __init__(self):

        self.support = None

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        pass

    def fit(self, X, y=None):

        self.support = np.arange(X.shape[1], dtype=np.int32)
        return self

    def transform(self, X):
        return X


class FisherScoreSelection(base.BaseSelector):
    """

    """

    NAME = 'FisherScoreSelection'

    def __init__(self, num_features: int = None, random_state: int = 0):

        super().__init__()

        self.num_features = num_features
        self.random_state = random_state

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):
        return self.NAME

    @property
    def config_space(self):
        """Returns the Fisher score hyperparameter configuration space."""

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y=None):
        """Perform feature selection.

        """
        X, y = self.check_X_y(X, y)
        self.check_params(X)

        _support = self.fisher_score_selection(X, y)[:self.num_features]
        self.support = self.check_support(_support)
        return self

    def check_params(self, X):

        _, ncols = np.shape(X)
        if self.num_features > ncols:
            self.num_features = int(ncols - 1)

        return self

    @staticmethod
    def fisher_score_selection(X, y):
        # Wrapping skfeature Fisher score.
        scores = fisher_score(X, y)
        return np.argsort(scores, 0)[::-1]


class WilcoxonSelection(base.BaseSelector):
    """Perform feature selection by Wilcoxon rank sum test.

    Args:
        bf_correction (bool): Apply Bonferroni correction for
            multiple testing correction.

    """

    NAME = 'WilcoxonSelection'

    def __init__(self, num_features: int = None, random_state: int = 0):

        super().__init__()

        self.num_features = num_features
        self.random_state = random_state

        # NOTE: Attribute set with instance.
        self.duplicates_in_X = False
        self.support = None

    def __name__(self):
        return self.NAME

    @property
    def config_space(self):
        """Returns the Wilcoxon selection hyperparameter configuration space.
        """

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y=None):
        """Perform feature selection.

        """
        X, y = self.check_X_y(X, y)

        scores = self.wilcoxon_selection(X, y)

        _support = np.argsort(scores)[:self.num_features]
        self.support = self.check_support(_support)

        return self

    @staticmethod
    def wilcoxon_selection(X, y):

        if np.size(np.unique(y)) != 2:
            raise ValueError(f'Dependent variable should be binary. Recieved'
                             f'{np.unique(y)} classes.')

        N, ncols = np.shape(X)
        n0, n1 = np.bincount(y)

        scores = []
        for num in range(ncols):
            r = rankdata(X[:, num])
            r0 = rankdata(X[y == 0, num])
            r1 = rankdata(X[y == 1, num])

            mu_r = np.mean(r)
            mu_r0 = np.mean(r0)
            mu_r1 = np.mean(r1)

            num = n0 * (mu_r0 - mu_r) ** 2 + n1 * (mu_r1 - mu_r) ** 2
            denom = sum((r0 - mu_r) ** 2) + sum((r1 - mu_r) ** 2)
            scores.append(num / denom)

        scores = np.array(scores, dtype=np.float64)

        return (N - 1) * scores


class ReliefFSelection(base.BaseSelector):
    """A wrapper for the scikit-rebate implementation of the ReliefF
    algorithm.

    Args:
        num_neighbors (int): Controls the locality of the estimates. The
            recommended default value is ten [3], [4].
        num_features (int):

    Notes:
    - There are no missing values in the dependent variable.
    - For ReliefF, the setting of k is <= to the number of instances that have
      the least frequent class label.
    - Algorithm is not stochastic (no randoms state required).

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
        [4]: M. Robnik-Sikonja, Kononenko, I.: 1994, 2003, ‘Theoretical and
             Empirical Analysis of ReliefF and RReliefF‘. In: Machine Learning
             Journal 53, p23-69.

    """

    NAME = 'ReliefFSelection'

    def __init__(
        self,
        num_neighbors: int = None,
        num_features: int = None,
        random_state: int = 0
    ):

        super().__init__()

        self.num_neighbors = num_neighbors
        self.num_features = num_features
        self.random_state = random_state

        # NOTE: Attributes set with instance.
        self.support = None
        self.scaler = None

    def __name__(self):
        return self.NAME

    @property
    def config_space(self):
        """Returns the ReliefF hyperparameter configuration space."""

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=2, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

    def fit(self, X, y=None, **kwargs):

        X, y = self.check_X_y(X, y)
        self.check_params(X, y)

        selector = ReliefF(
            n_neighbors=self.num_neighbors,
            n_features_to_select=self.num_features,
        )
        selector.fit(X, y)

        _support = selector.top_features_[:self.num_features]
        self.support = self.check_support(_support)

        return self

    def check_params(self, X, y):

        minority = np.argmin(np.bincount(y))
        nrows = sum(y == minority)
        if self.num_neighbors > nrows:
            self.num_neighbors = int(nrows)

        _, ncols = np.shape(X)
        if self.num_features > ncols:
            self.num_features = int(ncols - 1)

        return self


class MultiSURFSelection(base.BaseSelector):

    NAME = 'MultiSURFSelection'

    def __init__(
        self,
        num_features: int = None,
        random_state: int = 0
    ):

        super().__init__()

        self.num_features = num_features
        self.random_state = random_state

        # NOTE: Attributes set with instance.
        self.support = None
        self.scaler = None

    def __name__(self):
        return self.NAME

    @property
    def config_space(self):
        """Returns the ReliefF hyperparameter configuration space."""

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y=None, **kwargs):

        X, y = self.check_X_y(X, y)
        self.check_params(X, y)

        selector = MultiSURF(
            n_features_to_select=self.num_features,
        )
        selector.fit(X, y)

        _support = selector.top_features_[:self.num_features]
        self.support = self.check_support(_support)

        return self

    def check_params(self, X, y):

        _, ncols = np.shape(X)
        if self.num_features > ncols:
            self.num_features = int(ncols - 1)

        return self


class MutualInformationSelection(base.BaseSelector):

    NAME = 'MutualInformationSelection'

    def __init__(
        self,
        num_neighbors: int=None,
        num_features: int=None,
        random_state: int=0
    ):
        super().__init__()

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

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=10, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

    def fit(self, X, y, **kwargs):
        """
        """
        X, y = self.check_X_y(X, y)
        self.check_params(X, y)

        selector = SelectKBest(self.mutual_info_selection, k=self.num_features)
        selector.fit(X, y)

        _support = selector.get_support(indices=True)
        self.support = self.check_support(_support)

        return self

    def check_params(self, X, y):

        # Satisfying check in sklearn KDTree.
        
        k_thresh = min(np.bincount(y))
        if self.num_neighbors > k_thresh:
            self.num_neighbors = int(k_thresh)

        _, ncols = np.shape(X)
        if self.num_features > ncols:
            self.num_features = int(ncols - 1)

        return self

    def mutual_info_selection(self, X, y):
        # Wrapping sklearn mutual info clf enabling parameter config.
        return mutual_info_classif(
            X, y,
            discrete_features=False,
            n_neighbors=self.num_neighbors,
            random_state=self.random_state
)


class JointMutualInformationSelection(base.BaseSelector):

    NAME = 'JointMutualInformationSelection'

    def __init__(
        self,
        num_neighbors: int = None,
        num_features: int = None,
        random_state: int = 0
    ):
        super().__init__()

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

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=10, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

    def fit(self, X, y, **kwargs):
        """

        """
        X, y = self.check_X_y(X, y)
        self.check_params(X, y)

        selector = MutualInformationFeatureSelector(
            method='JMI',
            k=self.num_neighbors,
            n_features=self.num_features,
            categorical=True,
        )
        selector.fit(X, y)
        self.support = self.check_support(selector.support_)

        return self

    def check_params(self, X, y):

        # Satisfying check in sklearn KDTree.
        
        k_thresh = min(np.bincount(y))
        if self.num_neighbors > k_thresh:
            self.num_neighbors = int(k_thresh - 1)

        _, ncols = np.shape(X)
        if self.num_features > ncols:
            self.num_features = int(ncols - 1)

        return self

    def mutual_info_selection(self, X, y):
        # Wrapping sklearn mutual info clf enabling parameter config.
        return mutual_info_classif(
            X, y,
            discrete_features=False,
            n_neighbors=self.num_neighbors,
            random_state=self.random_state
        )


class ChiSquareSelection(base.BaseSelector):
    """
    Notes:
        - Requires non-negative feature values. Features, x, are shifted by
          x := x + abs(min(x)) + 1.

    """

    NAME = 'ChiSquareSelection'

    def __init__(self, num_features: int=None, random_state: int=0):

        super().__init__()

        self.num_features = num_features
        self.random_state = random_state

        # NOTE: Attributes set with instance.
        self.support = None

    def __name__(self):
        return self.NAME

    @property
    def config_space(self):
        """Returns the MI hyperparameter configuration space."""

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y, **kwargs):
        """

        """
        # Ensures all elements of X > 0 for Chi2 test.
        X, y = self._check_X_y(X, y)
        self.check_params(X, y)

        selector = SelectKBest(chi2, k=self.num_features)
        selector.fit(X, y)

        _support = selector.get_support(indices=True)
        self.support = self.check_support(_support)

        return self

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.
        X, y = check_X_y(X, y)
        # NOTE: Requires all features to be non-negative.
        X_nonegative = X + np.abs(np.min(X, axis=0)) + 1

        return X_nonegative, y

    def check_params(self, X, y):

        _, ncols = np.shape(X)
        if self.num_features > ncols:
            self.num_features = int(ncols - 1)

        return self


if __name__ == '__main__':
    pass
