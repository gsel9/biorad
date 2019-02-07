# -*- coding: utf-8 -*-
#
# classifiers.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

from . import base

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter


class RFEstimator(base.BaseEstimator):

    NAME = 'RFEstimator'

    def __init__(
        self,
        mode='classification',
        model=RandomForestClassifier(
            n_jobs=-1,
            verbose=False,
            oob_score=False,
            min_samples_split=2,
            class_weight='balanced',
        )
    ):

        super().__init__(model=model, mode=mode)

    @property
    def hparam_space(self):
        """Returns the PLS Regression hyperparameter space."""

        # NOTE: This algorithm is stochastic and its performance varies
        # across random states.
        hparam_space = (
            UniformIntegerHyperparameter(
                '{}__random_state'.format(self.NAME), lower=0, upper=1000,
            ),
            UniformIntegerHyperparameter(
                '{}__n_estimators'.format(self.NAME),
                lower=10,
                upper=3000,
                default_value=100
            ),
            CategoricalHyperparameter(
                '{}__criterion'.format(self.NAME),
                ['gini', 'entropy'], default_value='gini'
            ),
            CategoricalHyperparameter(
                '{}__max_depth'.format(self.NAME),
                [3, 5, None], default_value=None
            ),
            CategoricalHyperparameter(
                '{}__max_features'.format(self.NAME),
                ['auto', 'sqrt', 'log2', None], default_value=None
            ),
            CategoricalHyperparameter(
                '{}__bootstrap'.format(self.NAME),
                [True, False], default_value=True
            ),
            UniformFloatHyperparameter(
                '{}__min_samples_leaf'.format(self.NAME),
                lower=1.5,
                upper=50.5,
                default_value=1.0
            ),
        )
        return hparam_space


class PLSREstimator(base.BaseEstimator):

    NAME = 'PLSREstimator'

    def __init__(
        self,
        mode='classification',
        model=PLSRegression(scale=False, copy=True, max_iter=-1)
    ):

        super().__init__(model=model, mode=mode)

    @property
    def hparam_space(self):
        """Returns the PLS Regression hyperparameter space."""

        # NOTE: This algorithm is not stochastic and its performance does not
        # varying depending on a random number generator.
        hparam_space = (
            UniformFloatHyperparameter(
                '{}__tol'.format(self.NAME),
                lower=1e-9,
                upper=1e-3,
                default_value=1e-7
            ),
            UniformIntegerHyperparameter(
                '{}__n_components'.format(self.NAME),
                lower=1,
                upper=40,
                default_value=27
            ),
        )
        return hparam_space


# * The n_jobs > 1 does not have any effect when solver
#   is set to 'liblinear'.
# * Should explicitly specify penalty since this param depends
#   on the selected solver. Using L1 enables dim reduction in high
#   dim classification problems.
# * The `liblinear` solver allows for binary classification with L1 and L2
#   regularization, is robust to unscaled data sets, penalize the intercept
#   term, but is not faster for larger data sets.
# * Dual formulation is only implemented for l2 penalty with liblinear
#   solver. Thus, enabling to select from l1 or l2 requires Dual=False.
# * Set multi class to `ovr` for binary problems.
# * The max_iter is not usefull with `liblinear` solver.
class LogRegEstimator(base.BaseEstimator):

    NAME = 'LogRegEstimator'

    def __init__(
        self,
        mode='classification',
        model=LogisticRegression(
            solver='liblinear',
            max_iter=1000,
            verbose=0,
            n_jobs=1,
            dual=False,
            multi_class='ovr',
            warm_start=False,
            class_weight='balanced',
        )
    ):

        super().__init__(model=model, mode=mode)

    @property
    def hparam_space(self):
        """Returns the LR hyperparameter space."""

        # NOTE: This algorithm is stochastic and its performance varies
        # across random states.
        hparam_space = (
            UniformIntegerHyperparameter(
                '{}__random_state'.format(self.NAME), lower=0, upper=1000,
            ),
            UniformFloatHyperparameter(
                '{}__C'.format(self.NAME),
                lower=0.001, upper=1000.0, default_value=1.0
            ),
            CategoricalHyperparameter(
                '{}__penalty'.format(self.NAME),
                ['l1', 'l2'], default_value='l2'
            ),
        )
        return hparam_space


class SVCEstimator(base.BaseEstimator):

    NAME = 'SVCEstimator'

    def __init__(
        self,
        mode='classification',
        model=SVC(
            class_weight='balanced',
            verbose=False,
            cache_size=500,
            max_iter=-1,
            decision_function_shape='ovr',
        )
    ):

        super().__init__(model=model, mode=mode)

    @property
    def hparam_space(self):
        """Returns the SVC hyperparameter space."""

        # NOTE: This algorithm is stochastic and its performance varies
        # across random states.
        hparam_space = (
            # Hyperparameters shared by all kernels
            UniformIntegerHyperparameter(
                '{}__random_state'.format(self.NAME), lower=0, upper=1000,
            ),
            UniformFloatHyperparameter(
                '{}__C'.format(self.NAME),
                lower=0.001, upper=1000.0, default_value=1.0
            ),
            CategoricalHyperparameter(
                '{}__shrinking'.format(self.NAME),
                [True, False], default_value=True
                #['true', 'false'], default_value='true'
            ),
            # Hyperparameters specific to kernels.
            CategoricalHyperparameter(
                '{}__kernel'.format(self.NAME),
                ['linear', 'rbf', 'poly', 'sigmoid'],
            ),
            # - Poly kernel only:
            UniformIntegerHyperparameter(
                '{}__degree'.format(self.NAME), 1, 5, default_value=3
            ),
            # - Poly and sigmoid kernels:
            UniformFloatHyperparameter
                ('{}__coef0'.format(self.NAME), 0.0, 10.0, default_value=0.0
            ),
            # - RBF, poly and sigmoid kernels.
            CategoricalHyperparameter(
                '{}__gamma'.format(self.NAME),
                ['auto', 'value'], default_value='auto'
            ),
            UniformFloatHyperparameter(
                '{}__gamma_value'.format(self.NAME), 0.0001, 8, default_value=1
            ),
            # Activate hyperparameters according to choice of kernel.
            InCondition(child=degree, parent=kernel, values=['poly']),
            InCondition(child=gamma_value, parent=gamma, values=['value']),
            InCondition(
                child=gamma, parent=kernel, values=['rbf', 'poly', 'sigmoid']
            ),
            InCondition(
                child=coef0, parent=kernel, values=['poly', 'sigmoid']
            ),
        )
        return hparam_space


# NOTE: This algorithm does not associate hyperparameters.
class GNBEstimator(base.BaseEstimator):

    NAME = 'GNBEstimator'

    def __init__(
        self,
        mode='classification',
        model=GaussianNB()
    ):

        super().__init__(model=model, mode=mode)


class KNNEstimator(base.BaseEstimator):

    NAME = 'KNNEstimator'

    def __init__(
        self,
        mode='classification',
        model=KNeighborsClassifier(algorithm='auto')
    ):

        super().__init__(model=model, mode=mode)

    hparam_space = (
        UniformIntegerHyperparameter(
            '{}__n_neighbors'.format(self.NAME), 3, 100, default_value=5
        ),
        UniformIntegerHyperparameter(
            '{}__leaf_size'.format(self.NAME), 10, 100, default_value=30
        ),
        CategoricalHyperparameter(
            '{}__metric'.format(self.NAME),
            ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            default_value='minkowski'
        ),
        UniformIntegerHyperparameter(
            '{}__p'.format(self.NAME), 1, 5, default_value=2
        ),
        # Activate hyperparameters according to choice of metric.
        InCondition(child=p, parent=metric, values=['minkowski']),
    )
    return hparam_space


if __name__ == '__main__':
    pass
