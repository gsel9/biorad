# -*- coding: utf-8 -*-
#
# selectors.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

import warnings

import numpy as np

from . import base

from ReliefF import ReliefF

from sklearn.utils import check_X_y
from sklearn.preprocessing import MinMaxScaler

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter


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
    def hparam_space(self):
        """Return the ReliefF hyperparameter space."""

        # NOTE: This algorithm is not stochastic and its performance does not
        # varying depending on a random number generator.
        hparam_space = (
            UniformIntegerHyperparameter(
                '{}__num_neighbors'.format(self.NAME),
                lower=10,
                upper=100,
                default_value=20
            ),
            UniformIntegerHyperparameter(
                '{}__num_features'.format(self.NAME),
                lower=2,
                upper=50,
                default_value=20
            )
        )
        return hparam_space

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

        self._check_params(X)
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

    def _check_params(self, X):

        # Satisfying check in sklearn KDTree (binary tree).
        nrows, _ = np.shape(X)
        if self.num_neighbors > nrows:
            self.num_neighbors = int(nrows - 1)
        else:
            self.num_neighbors = int(self.num_neighbors)

        if self.num_features < 1:
            self.num_features = int(self.num_features)
        else:
            self.num_features = int(self.num_features)

        return self


class MutualInformationSelection(base.BaseSelector):

    NAME = 'MutualInformationSelection'

    def __init__(self, num_features=None, error_handling='all'):

        super().__init__(error_handling)

        self.num_features = num_features

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    def fit(self, X, y, **kwargs):

        X, y = self._check_X_y(X, y)
        try:
            selector = SelectKBest(
                score_func=mutual_info_classif, k=self.num_features
            )
            selector.fit(X, y)
            _support = selector.get_support(indices=True)
        except:
            warnings.warn('Failed support with {}.'.format(self.__name__))
            _support = []

        self.support = self.check_support(_support, X)

        return self

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        return check_X_y(X, y)
