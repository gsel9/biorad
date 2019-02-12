# -*- coding: utf-8 -*-
#
# meta_estimators.py
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


SEED = 0


class RFMetaEstimator(base.MetaClassifier):

    NAME = 'RFMetaClassifier'

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
    def config_space(self):
        """Returns the RF Regression hyperparameter space."""

        random_states = UniformIntegerHyperparameter(
            'random_state', lower=0, upper=1000,
        )
        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=10, upper=3000, default_value=100
        )
        criterion = CategoricalHyperparameter(
            'criterion', ['gini', 'entropy'], default_value='gini'
        )
        max_depth = CategoricalHyperparameter(
            'max_depth', [3, 5, None], default_value=None
        )
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', None], default_value=None
        )
        bootstrap = CategoricalHyperparameter(
            'bootstrap', [True, False], default_value=True
        )
        min_samples_leaf = UniformFloatHyperparameter(
            'min_samples_leaf', lower=1e-6, upper=0.5,
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameters(
            (
                random_states,
                n_estimators,
                criterion,
                max_depth,
                max_features,
                bootstrap,
                min_samples_leaf,
                num_features
            )
        )
        return config


class PLSRMetaEstimator(base.MetaClassifier):

    NAME = 'PLSRMetaClassifier'

    def __init__(
        self,
        mode='classification',
        model=PLSRegression(scale=False, copy=True, max_iter=-1)
    ):

        super().__init__(model=model, mode=mode)

    @property
    def config_space(self):
        """Returns the PLS regression hyperparameter configuration space."""

        global SEED

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        tol = UniformFloatHyperparameter(
            'tol', lower=1e-9, upper=1e-3, default_value=1e-7
        )
        n_components = UniformIntegerHyperparameter(
            'n_components', lower=1, upper=40, default_value=27
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameters((tol, n_components, num_features))

        return config


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
class LogRegMetaEstimator(base.MetaClassifier):

    NAME = 'LogRegMetaClassifier'

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
    def config_space(self):
        """Returns the LR hyperparameter configuration space."""

        global SEED

        random_states = UniformIntegerHyperparameter(
            'random_state', lower=0, upper=1000,
        )
        C = UniformFloatHyperparameter(
            'C', lower=0.001, upper=1000.0, default_value=1.0
        )
        penalty = CategoricalHyperparameter(
            'penalty', ['l1', 'l2'], default_value='l2'
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameters((random_states, C, penalty, num_features))

        return config


class SVCMetaEstimator(base.MetaClassifier):

    NAME = 'SVCMetaClassifier'

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
    def config_space(self):
        """Returns the SVC hyperparameter configuration space."""

        global SEED

        random_states = UniformIntegerHyperparameter(
            'random_state', lower=0, upper=1000,
        )
        C = UniformFloatHyperparameter(
            'C', lower=0.001, upper=1000.0, default_value=1.0
        )
        shrinking = CategoricalHyperparameter(
            'shrinking', [True, False], default_value=True
            #['true', 'false'], default_value='true'
        )
        kernel = CategoricalHyperparameter(
            'kernel', ['linear', 'rbf', 'poly', 'sigmoid'],
        )
        gamma = CategoricalHyperparameter(
            'gamma', ['auto', 'value'], default_value='auto'
        )
        gamma_value = UniformFloatHyperparameter(
            'gamma_value', lower=0.0001, upper=8, default_value=1
        )
        degree = UniformIntegerHyperparameter(
            'degree', lower=1, upper=5, default_value=3
        )
        coef0 = UniformFloatHyperparameter(
            'coef0', lower=0.0, upper=10.0, default_value=0.0
        )
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameters(
            (
                random_states,
                C,
                shrinking,
                kernel,
                degree,
                coef0,
                gamma,
                gamma_value,
                num_features
            )
        )
        # Conditionals on hyperparameters specific to kernels.
        config.add_conditions(
            (
                InCondition(child=degree, parent=kernel, values=['poly']),
                InCondition(child=gamma_value, parent=gamma, values=['value']),
                InCondition(
                    child=coef0, parent=kernel,
                    values=['poly', 'sigmoid']
                ),
                InCondition(
                    child=gamma, parent=kernel,
                    values=['rbf', 'poly', 'sigmoid']
                )
            )
        )
        return config


class GNBMetaEstimator(base.MetaClassifier):

    NAME = 'GNBMetaClassifier'

    def __init__(
        self,
        mode='classification',
        model=GaussianNB()
    ):

        super().__init__(model=model, mode=mode)

    @property
    def config_space(self):
        """Returns the KNN hyperparameter configuration space."""

        global SEED

        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )
        # Add hyperparameter to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameter(num_features)

        return config


class KNNMetaEstimator(base.MetaClassifier):

    NAME = 'KNNMetaClassifier'

    def __init__(
        self,
        mode='classification',
        model=KNeighborsClassifier(algorithm='auto')
    ):

        super().__init__(model=model, mode=mode)

    @property
    def config_space(self):
        """Returns the KNN hyperparameter configuration space."""

        global SEED

        n_neighbors = UniformIntegerHyperparameter(
            'n_neighbors', 3, 100, default_value=5
        )
        leaf_size = UniformIntegerHyperparameter(
            'leaf_size', 10, 100, default_value=30
        )
        metric = CategoricalHyperparameter(
            'metric',
            ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            default_value='minkowski'
        )
        p = UniformIntegerHyperparameter('p', 1, 5, default_value=2)
        num_features = UniformIntegerHyperparameter(
            'num_features', lower=2, upper=50, default_value=20
        )

        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(SEED)
        config.add_hyperparameters(
            (n_neighbors, leaf_size, metric, p, num_features)
        )
        # Conditionals on hyperparameters specific to kernels.
        config.add_condition(
            InCondition(child=p, parent=metric, values=['minkowski'])
        )
        return config


if __name__ == '__main__':
    pass
