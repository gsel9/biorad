# -*- coding: utf-8 -*-
#
# classification.py
#

"""
Wrappers for classification algorithms ensuring unified API for model
comparison experiments.
"""

__author__ = 'Severin Langberg'
__contact__ = 'langberg91@gmail.com'


import numpy as np

#from pyglmnet import GLM

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from . import base


class GroupLASSOEstimator(base.BaseClassifier):
    """A logistic regression wrapper."""

    NAME = 'GroupLASSOEstimator'

    def __init__(
        self,
        group_idx=None,
        model=None,#GLM(max_iter=800),
        random_state=0
    ):

        model.group = group_idx
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Logistic regression hyperparameter space."""

        distr = CategoricalHyperparameter(
            'distr', ['gaussian', 'binomial', 'poisson', 'softplus'],
            default_value='poisson'
        )
        alpha = UniformFloatHyperparameter(
            'alpha', lower=0, upper=1, default_value=0.5
        )
        reg_lambda = UniformFloatHyperparameter(
            'reg_lambda', lower=1e-10, upper=10-1e-10, default_value=1e-3
        )
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', lower=1e-10, upper=10-1e-10, default_value=0.01
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (distr, alpha, reg_lambda, learning_rate)
        )
        return config


class ElasticNetEstimator(base.BaseClassifier):
    """A logistic regression wrapper."""

    NAME = 'ElasticNetEstimator'

    def __init__(
        self,
        model=ElasticNet(
            alpha=1.0,
            fit_intercept=True,
            normalize=False,
            precompute=False,
            max_iter=int(1e4),
            positive=False,
            selection='cyclic'
        ),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Logistic regression hyperparameter space."""

        tol = UniformFloatHyperparameter(
            'tol', lower=1e-9, upper=1e-4, default_value=1e-7
        )
        l1_ratio = UniformFloatHyperparameter(
            'l1_ratio', lower=0, upper=1, default_value=0.5
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters((tol, l1_ratio))

        return config


class LightGBM(base.BaseClassifier):
    """A Light Gradient Boosting wrapper"""

    NAME = 'LightGBM'

    def __init__(
        self,
        model=LGBMClassifier(
            boosting_type='gbdt',
            class_weight='balanced',
            objective='binary',
            n_jobs=-1
        ),
        random_state=0
    ):

        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """LightGBM hyperparameter space."""

        # The mumber of Decision Trees.
        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=10, upper=200, default_value=100
        )
        # The maximum depth of each decision tree. Generally, boosting
        # algorithms are configured with weak learners = shallow decision trees.
        max_depth = UniformIntegerHyperparameter(
            'max_depth', lower=5, upper=500, default_value=100
        )
        # L1 regularization term on weights.
        reg_alpha = UniformFloatHyperparameter(
            'reg_alpha', lower=1e-8, upper=100, default_value=1e-3
        )
        # L2 regularization term on weights.
        reg_lambda = UniformFloatHyperparameter(
            'reg_lambda', lower=1e-8, upper=100, default_value=1e-3
        )
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', lower=1e-8, upper=50, default_value=0.01
        )
        min_data_in_leaf = UniformIntegerHyperparameter(
            'min_data_in_leaf', lower=2, upper=10, default_value=5
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                n_estimators,
                max_depth,
                reg_alpha,
                reg_lambda,
                learning_rate,
                min_data_in_leaf
            )
        )
        return config


class XGBoosting(base.BaseClassifier):
    """An eXtreme Gradient Boosting wrapper"""

    NAME = 'XGBoost'

    def __init__(
        self,
        model=XGBClassifier(
            missing=None,
            booster='gbtree',
            objective='binary:logistic',
            eval_metric='auc',
            n_jobs=-1
        ),
        random_state=0
    ):

        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """XGBoost hyperparameter space."""

        # The mumber of Decision Trees.
        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=10, upper=200, default_value=100
        )
        # The maximum depth of each decision tree. Generally, boosting
        # algorithms are configured with weak learners = shallow decision trees.
        max_depth = UniformIntegerHyperparameter(
            'max_depth', lower=5, upper=500, default_value=100
        )
        # L1 regularization term on weights.
        reg_alpha = UniformFloatHyperparameter(
            'reg_alpha', lower=1e-8, upper=100, default_value=1e-3
        )
        # L2 regularization term on weights.
        reg_lambda = UniformFloatHyperparameter(
            'reg_lambda', lower=1e-8, upper=100, default_value=1e-3
        )
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', lower=1e-8, upper=50, default_value=0.01
        )
        min_data_in_leaf = UniformIntegerHyperparameter(
            'min_data_in_leaf', lower=2, upper=10, default_value=5
        )
        # The minimum loss reduction required to make a split.
        min_split_loss = UniformFloatHyperparameter(
            'min_split_loss', lower=1e-10, upper=1e-4, default_value=1e-9
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                n_estimators,
                max_depth,
                reg_alpha,
                reg_lambda,
                learning_rate,
                min_data_in_leaf,
                min_split_loss
            )
        )
        return config


class ExtraTreeEstimator(base.BaseClassifier):
    """Extra-trees differ from classic decision trees in the way they are built.
    """

    NAME = 'ExtraTreeEstimator'

    def __init__(
        self,
        model=ExtraTreeClassifier(class_weight='balanced'),
        random_state=0
    ):

        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Decision tree hyperparameter space."""

        criterion = CategoricalHyperparameter(
            'criterion', ['gini', 'entropy'], default_value='gini'
        )
        max_depth = CategoricalHyperparameter(
            'max_depth', [5, 10, 20, None], default_value=None
        )
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', None], default_value=None
        )
        min_samples_leaf = UniformFloatHyperparameter(
            'min_samples_leaf', lower=1e-10, upper=0.3,
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                criterion,
                max_depth,
                max_features,
                min_samples_leaf
            )
        )
        return config


class DTreeEstimator(base.BaseClassifier):
    """A Decision tree classifier wrapper"""

    NAME = 'DTreeEstimator'

    def __init__(
        self,
        model=DecisionTreeClassifier(class_weight='balanced'),
        random_state=0
    ):

        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Decision tree hyperparameter space."""

        criterion = CategoricalHyperparameter(
            'criterion', ['gini', 'entropy'], default_value='gini'
        )
        max_depth = CategoricalHyperparameter(
            'max_depth', [5, 10, 20, None], default_value=None
        )
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', None], default_value=None
        )
        min_samples_leaf = UniformFloatHyperparameter(
            'min_samples_leaf', lower=1e-10, upper=0.25,
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                criterion,
                max_depth,
                max_features,
                min_samples_leaf
            )
        )
        return config


class RFEstimator(base.BaseClassifier):
    """A random rorest classifier wrapper."""

    NAME = 'RFEstimator'

    def __init__(
        self,
        model=RandomForestClassifier(
            n_jobs=-1,
            verbose=False,
            oob_score=False,
            class_weight='balanced',
        ),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Random forest classifier hyperparameter space."""

        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=10, upper=1000, default_value=100
        )
        criterion = CategoricalHyperparameter(
            'criterion', ['gini', 'entropy'], default_value='gini'
        )
        max_depth = CategoricalHyperparameter(
            'max_depth', [5, 10, 20, None], default_value=None
        )
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', None], default_value=None
        )
        bootstrap = CategoricalHyperparameter(
            'bootstrap', [True, False], default_value=True
        )
        min_samples_leaf = UniformFloatHyperparameter(
            'min_samples_leaf', lower=1e-10, upper=0.25,
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                n_estimators,
                criterion,
                max_depth,
                max_features,
                bootstrap,
                min_samples_leaf
            )
        )
        return config


class PLSREstimator(base.BaseClassifier):
    """A PLSR wrapper."""

    NAME = 'PLSREstimator'

    def __init__(
        self,
        model=PLSRegression(
            scale=False,
            copy=True,
            max_iter=int(1e4)
        ),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """PLSR hyperparameter space."""

        tol = UniformFloatHyperparameter(
            'tol', lower=1e-9, upper=1e-4, default_value=1e-7
        )
        n_components = UniformIntegerHyperparameter(
            'n_components', lower=1, upper=30, default_value=10
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters((tol, n_components))

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
class LogRegEstimator(base.BaseClassifier):
    """A logistic regression wrapper."""

    NAME = 'LogRegEstimator'

    def __init__(
        self,
        model=LogisticRegression(
            solver='liblinear',
            max_iter=int(1e4),
            verbose=0,
            n_jobs=1,
            dual=False,
            multi_class='ovr',
            warm_start=False,
            class_weight='balanced',
        ),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Logistic regression hyperparameter space."""

        C_param = UniformFloatHyperparameter(
            'C', lower=1e-8, upper=1000.0, default_value=1.0
        )
        penalty = CategoricalHyperparameter(
            'penalty', ['l1', 'l2'], default_value='l1'
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters((C_param, penalty))

        return config


class SVCEstimator(base.BaseClassifier):
    """A support vector classifier wrapper."""

    NAME = 'SVCEstimator'

    def __init__(
        self,
        model=SVC(
            class_weight='balanced',
            verbose=False,
            cache_size=1500,
            max_iter=int(3e4),
            decision_function_shape='ovr',
        ),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """SVC hyperparameter space."""

        C_param = UniformFloatHyperparameter(
            'C', lower=1e-8, upper=1000.0, default_value=1.0
        )
        shrinking = CategoricalHyperparameter(
            'shrinking', [True, False], default_value=True
        )
        kernel = CategoricalHyperparameter(
            'kernel', ['linear', 'rbf', 'poly', 'sigmoid'],
        )
        gamma = CategoricalHyperparameter(
            'gamma', ['auto', 'value'], default_value='auto'
        )
        gamma_value = UniformFloatHyperparameter(
            'gamma_value', lower=1e-8, upper=10, default_value=1
        )
        degree = UniformIntegerHyperparameter(
            'degree', lower=1, upper=5, default_value=2
        )
        coef0 = UniformFloatHyperparameter(
            'coef0', lower=0.0, upper=10.0, default_value=0.0
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                C_param,
                shrinking,
                kernel,
                degree,
                coef0,
                gamma,
                gamma_value
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


class KNNEstimator(base.BaseClassifier):

    NAME = 'KNNEstimator'

    def __init__(
        self,
        model=KNeighborsClassifier(algorithm='auto'),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """KNN hyperparameter space."""

        n_neighbors = UniformIntegerHyperparameter(
            'n_neighbors', 1, 100, default_value=5
        )
        leaf_size = UniformIntegerHyperparameter(
            'leaf_size', 1, 100, default_value=20
        )
        metric = CategoricalHyperparameter(
            'metric',
            ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            default_value='euclidean'
        )
        p_param = UniformIntegerHyperparameter('p', 1, 5, default_value=2)
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters((n_neighbors, leaf_size, metric, p_param))
        # Conditionals on hyperparameters specific to kernels.
        config.add_condition(
            InCondition(child=p_param, parent=metric, values=['minkowski'])
        )
        return config


if __name__ == '__main__':
    pass
