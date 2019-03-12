# -*- coding: utf-8 -*-
#
# selectors.py
#

"""

Notes:
* Permutation importance is sensitive towards correlated features.
* ReliefF requires scaling of features to unit length.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

import mifs
import warnings

import numpy as np
import pandas as pd

from copy import deepcopy
from datetime import datetime

from ReliefF import ReliefF
from scipy.stats import ranksums
from scipy.stats import ttest_ind

from sklearn.svm import SVC
from sklearn.utils import check_X_y
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter


from skfeature.function.similarity_based.fisher_score import fisher_score

try:
    from . import base
except:
    import base


# TODO: Refactor to Cython.
class GeneralizedFisherScore(base.BaseSelector):

    SEED = 0
    NAME = 'GeneralizedFisherScore'

    def __init__(
        self,
        feature_pairs: int=None,
        num_classes: int=None,
        gamma: float=None,
        max_iter=10,
        learning_rate=0.01,
        error_handling: str='all'
    ):
        super().__init__(error_handling)

        self.feature_pairs = feature_pairs
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.gamma = gamma

        # NOTE: Attribute set with instance.
        self.support = None
        self.omega = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):
        """Returns the ANOVA F-value hyperparameter configuration space."""

        feature_pairs = UniformIntegerHyperparameter(
            'feature_pairs', lower=2, upper=50, default_value=20
        )
        gamma = UniformFloatHyperparameter(
            'gamma', lower=1e-6, upper=20, default_value=0.5
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters((feature_pairs, gamma))

        return config

    def fit(self, X, y=None, **kwargs):

        if self.num_classes is None:
            self.num_classes = np.unique(y)

        if self.gamma <= 0:
            raise ValueError('Gamma parameter should be > 0!')

        num_rows, num_cols = np.shape(X)
        if self.feature_pairs > num_cols:
            print(f'Cannot compare {num_cols} feature pairs.')

        # Initializing.
        V = 1 / num_rows * np.ones((num_rows, self.num_classes), dtype=np.float32)
        t = 1
        # The most violated constraint.
        self.omega = [self.violated_constraints(num_rows, num_cols, X, V)]

        H = self.construct_H(num_rows, X, y)
        # Iteratively solves a multiple kernel learning problem.
        for _ in range(self.max_iter):
            # Initialize kernel weights.
            lambdas = 1 / t * np.ones(len(self.omega), dtype=np.float32)
            # Alternating optimization by multivariante ridge regression and
            # projected gradient descent.
            for _ in range(10):
                V = self.update_V(num_rows, num_cols, lambdas, X, H)
                lambdas = self.update_lambdas(num_cols, lambdas, X, V)

            self.omega.append(self.violated_constraints(num_rows, num_cols, X, V))
            t = t + 1

        _support = np.zeros(num_cols, dtype=bool)
        for array in self.omega:
            _support[np.where(array != 0)] = True

        self.support = self.check_support(_support, X)
        # TEMP:
        print('FS complete!')
        return self


    def violated_constraints(self, num_rows, num_cols, X, V):
        """
        See equation 26.
        """
        s = []
        for col_num in range(num_cols):
            x_dot_V = np.dot(X[:, col_num], V)
            Vt_dot_xt = np.dot(np.transpose(V), np.transpose(X[:, col_num]))
            s.append(np.dot(x_dot_V, Vt_dot_xt))

        p = np.zeros(num_cols, dtype=int)
        idx = np.argsort(s)[::-1][:self.feature_pairs]
        p[idx] = 1

        return p

    def construct_H(self, num_rows, X, y):

        H = np.ones((num_rows, self.num_classes), dtype=np.float32)
        for num in range(self.num_classes):
            idx = np.squeeze(np.where(y == num))
            num_hits = np.size(idx)
            H[:, num] = H[:, num] * -1.0 * np.sqrt(num_hits / num_rows)
            H[idx, num] = H[idx, num] * np.sqrt(num_rows / num_hits) - np.sqrt(num_hits / num_rows)

        assert np.shape(H) == (num_rows, self.num_classes)

        return H

    # NOTE: May replace dot(x.T, x) by arbitrary kernel function.
    def update_V(self, num_rows, num_cols, lambdas, X, H):
        """

        Returns:
            (array-like): The (n x c) replacement for the previous V matrix.

        """
        # Constructs (n x n) identity matrix.
        I = np.identity(num_rows)

        output = 0
        for t, p_t in enumerate(self.omega):
            core = 0
            for num_col in range(num_cols):
                x = np.copy(X[:, num_col])[np.newaxis]
                core = core + p_t[num_col] * np.dot(np.transpose(x), x) + I
            output = output + lambdas[t] * core
        output = np.dot(np.linalg.inv(1 / self.gamma * output), H)

        return output

    def update_lambdas(self, num_cols, lambdas, X, V):

        output = 0
        for t, p_t in enumerate(self.omega):
            core = 0
            for num_col in range(num_cols):
                x = np.copy(X[:, num_col])[np.newaxis]
                core = core + p_t[num_col] * np.dot(np.dot(np.transpose(x), x), V)
            lambda_gradient = 1 / (2 * self.gamma) * np.trace(np.dot(np.transpose(V), core))
            # Update lambda by gradient descent.
            lambdas[t] = lambdas[t] + self.learning_rate * lambda_gradient
        # Make lambdas sum to 1.
        lambdas = lambdas / np.sum(lambdas)

        return lambdas

class StudentTTestSelection(base.BaseSelector):

    SEED = 0
    NAME = 'StudentTTestSelection'

    def __init__(
        self,
        num_features: int=None,
        error_handling: str='all'
    ):
        super().__init__(error_handling)

        self.num_features = num_features

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def _check_params(self, X, y):

        _, ncols = np.shape(X)

        if self.num_features < 1:
            self.num_features = int(self.num_features)
        elif self.num_features > ncols:
            self.num_features = int(ncols - 1)
        else:
            self.num_features = int(self.num_features)

        return self

    @property
    def config_space(self):
        """Returns the ANOVA F-value hyperparameter configuration space."""

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=100, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)

        def _ttest_ind(X, y):
            # Wrapping scipy stats t-test enabling parameter configuration.
            return ttest_ind(X, y, equal_var=False)

        self._check_params(X, y)
        selector = SelectKBest(_ttest_ind, k=self.num_features)
        selector.fit(X, y)

        _support = selector.get_support(indices=True)
        self.support = self.check_support(_support, X)

        return self


class CorrelationSelection(base.BaseSelector):
    """Base representation of correlation based feature selection."""

    SEED = 0
    NAME = 'CorrelationSelection'

    def __init__(
        self,
        method=None,
        num_features=None,
        error_handling='all'
    ):

        super().__init__(error_handling)

        self.method = method
        self.num_features = num_features

        # NOTE: Attribute set with instance.
        self.support = None

    def __name__(self):

        return self.NAME

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    @property
    def config_space(self):
        """Returns the ANOVA F-value hyperparameter configuration space."""

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)

        cols = np.arange(X.shape[1], dtype=int)
        df_X = pd.DataFrame(X, columns=cols)

        # Create and select the upper triangle of correlation matrix.
        corr_matrix = df_X.corr(method=self.method).abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        # Remove columns with only NaNs and replace lower triangle NaNs with 0.
        upper_clean = upper.dropna(axis=1, how='all').replace(np.nan, 0)
        feature_corr = np.max(upper_clean.values, axis=0)
        _support = np.argsort(feature_corr)[:self.num_features]

        self.support = self.check_support(_support, X)

        return self


class CorrelationEnsembleSelection(base.BaseSelector):

    SEED = 0
    NAME = 'CorrelationEnsembleSelection'

    def __init__(
        self,
        pearson_num_features=None,
        kendall_num_features=None,
        spearman_num_features=None,
        error_handling='all'
    ):
        super().__init__(error_handling)

        self.pearson_num_features = pearson_num_features
        self.kendall_num_features = kendall_num_features
        self.spearman_num_features = spearman_num_features

        # NOTE: Attribute set with instance.
        self.model = FeatureUnion(
            [
                ('pearson', CorrelationSelection(method='pearson')),
                ('kendall', CorrelationSelection(method='kendall')),
                ('spearman', CorrelationSelection(method='spearman'))
            ]
        )

    def __name__(self):

        return self.NAME

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    @property
    def config_space(self):
        """Returns the CorrelationEnsembleSelection hyperparameter configuration space."""

        pearson_thresh = UniformIntegerHyperparameter(
            'pearson__num_features', lower=2, upper=50, default_value=20
        )
        kendall_thresh = UniformIntegerHyperparameter(
            'kendall__num_features', lower=2, upper=50, default_value=20
        )
        spearman_thresh = UniformIntegerHyperparameter(
            'spearman__num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters(
            (pearson_thresh, kendall_thresh, spearman_thresh)
        )
        return config

    def set_params(self, **kwargs):

        self.model.set_params(**kwargs)

        return self

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)

        self.model.fit(X)

        return self

    def transform(self, X, **kwargs):

        return self.model.transform(X)

    def _check_params(self, X, y):

        _, ncols = np.shape(X)

        if self.pearson_num_features < 1:
            self.pearson_num_features = int(self.pearson_num_features)
        elif self.pearson_num_features > ncols:
            self.pearson_num_features = int(ncols - 1)
        else:
            self.pearson_num_features = int(self.pearson_num_features)

        if self.kendall_num_features < 1:
            self.kendall_num_features = int(self.kendall_num_features)
        elif self.kendall_num_features > ncols:
            self.kendall_num_features = int(ncols - 1)
        else:
            self.kendall_num_features = int(self.kendall_num_features)

        if self.spearman_num_features < 1:
            self.spearman_num_features = int(self.spearman_num_features)
        elif self.spearman_num_features > ncols:
            self.spearman_num_features = int(ncols - 1)
        else:
            self.spearman_num_features = int(self.spearman_num_features)

        return self


class ANOVAFvalueSelection(base.BaseSelector):
    """

    """

    SEED = 0
    NAME = 'ANOVAFvalueSelection'

    def __init__(
        self,
        num_features: bool=None,
        error_handling: str='all'
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

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=100, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameter(num_features)

        return config

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)

        self._check_params(X, y)
        selector = SelectKBest(f_classif, k=self.num_features)
        selector.fit(X, y)

        _support = selector.get_support(indices=True)
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


class FisherScoreSelection(base.BaseSelector):
    """

    """

    SEED = 0
    NAME = 'FisherScoreSelection'

    def __init__(
        self,
        num_features: bool=None,
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

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=100, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameter(num_features)

        return config

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)

        def _fisher_score(X, y):
            # Wrapping skfeature Fisher score.
            scores = fisher_score(X, y)
            return np.argsort(scores, 0)[::-1]

        self._check_params(X, y)
        _support = _fisher_score(X, y)[:self.num_features]
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

    SEED = 0
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

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=100, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameter(num_features)

        return config

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around the sklearn formatter function.

        return check_X_y(X, y)

    def fit(self, X, y=None, **kwargs):

        X, y = self._check_X_y(X, y)

        p_values = self.wilcoxon_rank_sum(X, y)

        _support = np.argsort(p_values)[:self.num_features]
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

    SEED = 0
    NAME = 'MRMRSelection'

    def __init__(
        self,
        num_neighbors: int=None,
        num_features: int=None,
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

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=10, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=100, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

    def fit(self, X, y=None, **kwargs):
        """

        """
        # Ensures all elements of X > 0.
        X, y = self._check_X_y(X, y)

        self._check_params(X, y)
        try:
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
        except:
            warnings.warn('Failed support with {}.'.format(self.__name__))
            _support = []
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
        # Assumes X is already Z-score transformed rendering all features on
        # comparable scales such that a shift of all individ. feature values
        # renders all features > 0, whilst on comparable scales.
        X_shited = X + np.abs(np.min(X)) + 1

        return X_shifted, y


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
        [4]: M. Robnik-Sikonja, Kononenko, I.: 1994, 2003, ‘Theoretical and
             Empirical Analysis of ReliefF and RReliefF‘. In: Machine Learning
             Journal 53, p23-69.

    """

    SEED = 0
    NAME = 'ReliefFSelection'

    def __init__(
        self,
        num_neighbors: int=None,
        num_features: int=None,
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

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=10, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=100, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

    def _check_X_y(self, X, y):
        # A wrapper around sklearn formatter.

        X, y = check_X_y(X, y)

        # Scaling to [0, 1] range as recommended for algorithm.
        if self.scaler is None:
            self.scaler = MinMaxScaler()

        X = self.scaler.fit_transform(X)

        return X, y

    def fit(self, X, y=None, **kwargs):

        # NOTE: Includes scaling features to [0, 1] range.
        X, y = self._check_X_y(X, y)

        self._check_params(X, y)

        selector = ReliefF(
            n_neighbors=self.num_neighbors,
            n_features_to_keep=self.num_features
        )
        selector.fit(X, y)

        _support = selector.top_features[:self.num_features]
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

    SEED = 0
    NAME = 'MutualInformationSelection'

    def __init__(
        self,
        num_neighbors: int=None,
        num_features: int=None,
        random_state: int=None,
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

        num_neighbors = UniformIntegerHyperparameter(
            'num_neighbors', lower=10, upper=100, default_value=20
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=100, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters((num_neighbors, num_features))

        return config

    def fit(self, X, y, **kwargs):
        """

        """
        X, y = self._check_X_y(X, y)

        def _mutual_info_classif(X, y):
            # Wrapping sklearn mutual info clf enabling parameter config.
            return mutual_info_classif(
                X, y,
                discrete_features=False,
                n_neighbors=self.num_neighbors,
                random_state=self.random_state
            )

        self._check_params(X, y)

        selector = SelectKBest(_mutual_info_classif, k=self.num_features)
        selector.fit(X, y)

        _support = selector.get_support(indices=True)
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


class ChiSquareSelection(base.BaseSelector):

    SEED = 0
    NAME = 'ChiSquareSelection'

    def __init__(
        self,
        num_features: int=None,
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

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=100, default_value=20
        )
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y, **kwargs):
        """

        """
        # Ensures all elements of X > 0 for Chi2 test.
        X, y = self._check_X_y(X, y)

        self._check_params(X, y)

        selector = SelectKBest(chi2, k=self.num_features)
        selector.fit(X, y)

        _support = selector.get_support(indices=True)
        self.support = self.check_support(_support, X)

        return self

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

        X, y = check_X_y(X, y)
        # Assumes X is already Z-score transformed rendering all features on
        # comparable scales such that a shift of all individ. feature values
        # renders all features > 0, whilst on comparable scales.
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


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    _X, y = iris.data, iris.target

    X = np.zeros((_X.shape[0], _X.shape[1] + 2))
    X[:, 0] = np.ones(_X.shape[0])#np.random.random(_X.shape[0])
    X[:, -1] = np.ones(_X.shape[0])#np.random.random(_X.shape[0])
    X[:, 1:-1] = _X

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    gfs = GeneralizedFisherScore(num_classes=2, feature_pairs=2, gamma=0.5)
    print(X_std[:4, :])
    gfs.fit(X_std, y)
    X_sub = gfs.transform(X_std)
    print(X_sub[:4, :])
