# -*- coding: utf-8 -*-
#
# classification.py
#

"""
Wrappers for classification algorithms ensuring unified API for model
comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from . import base


class RidgeClassifierEstimator(base.BaseClassifier):

    NAME = 'RidgeClassifier'

    def __init__(
        self,
        model=RidgeClassifier(class_weight='balanced'),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Logistic regression hyperparameter space."""

        alpha = UniformFloatHyperparameter(
            'alpha', lower=1e-8, upper=100, default_value=1.0
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameter(alpha)

        return config


class QuadraticDiscriminantEstimator(base.BaseClassifier):
    """A logistic regression wrapper."""

    NAME = 'QuadraticDiscriminantEstimator'

    def __init__(
        self,
        model=QuadraticDiscriminantAnalysis(),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Logistic regression hyperparameter space."""

        reg_param = UniformFloatHyperparameter(
            'reg_param', lower=1e-9, upper=1-1e-9, default_value=1e-3
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameter(reg_param)

        return config


class LinearDiscriminantEstimator(base.BaseClassifier):
    """Linear Discriminant Analysis.

    Notes:
        * Using the SVD solver does not allow for shrinkage because it does
          not require computing the co-varaiance matrix rendering the SVD
          solver recommended for high-dimensional data.

    """

    NAME = 'LinearDiscriminantEstimator'

    def __init__(
        self,
        model=LinearDiscriminantAnalysis(),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Logistic regression hyperparameter space."""

        n_components = UniformIntegerHyperparameter(
            'n_components', lower=2, upper=50, default_value=10
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(n_components)

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
            'min_data_in_leaf', lower=2, upper=5, default_value=100
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                min_data_in_leaf,
                n_estimators,
                max_depth,
                reg_alpha,
                reg_lambda,
                learning_rate
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
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                n_estimators,
                max_depth,
                reg_alpha,
                reg_lambda,
                learning_rate
            )
        )
        return config


class ExtraTreesEstimator(base.BaseClassifier):
    """Extra-trees differ from classic decision trees in the way they are built.
    """

    NAME = 'ExtraTreesEstimator'

    def __init__(
        self,
        model=ExtraTreesClassifier(class_weight='balanced'),
        random_state=0
    ):

        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Decision tree hyperparameter space."""

        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=10, upper=200, default_value=100
        )
        criterion = CategoricalHyperparameter(
            'criterion', ['gini', 'entropy'], default_value='gini'
        )
        max_depth = CategoricalHyperparameter(
            'max_depth', [5, 10, 20, 'none'], default_value='none'
        )
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', 'none'], default_value='none'
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                n_estimators,
                criterion,
                max_depth,
                max_features
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
        # NOTE: Default value = 'none' is translated to None in base class.
        # ConfigSpace does not allow for None values as default.
        max_depth = CategoricalHyperparameter(
            'max_depth', [5, 10, 20, 'none'], default_value='none'
        )
        # NOTE: Default value = 'none' is translated to None in base class.
        # ConfigSpace does not allow for None values as default.
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', 'none'],
            default_value='none'
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                criterion,
                max_depth,
                max_features
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
            max_features=None,
            class_weight='balanced',
        ),
        random_state=0
    ):
        super().__init__(model=model, random_state=random_state)

        self.random_state = random_state

    @property
    def config_space(self):
        """Decision tree hyperparameter space."""

        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=10, upper=800, default_value=100
        )
        criterion = CategoricalHyperparameter(
            'criterion', ['gini', 'entropy'], default_value='gini'
        )
        # NOTE: Default value = 'none' is translated to None in base class.
        # ConfigSpace does not allow for None values as default.
        max_depth = CategoricalHyperparameter(
            'max_depth', [5, 10, 20, 'none'], default_value='none'
        )
        # NOTE: Default value = 'none' is translated to None in base class.
        # ConfigSpace does not allow for None values as default.
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', 'none'],
            default_value='none'
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters(
            (
                n_estimators,
                criterion,
                max_depth,
                max_features
            )
        )
        return config


# * The n_jobs > 1 does not have any effect when solver
#   is set to 'liblinear'.
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
            gamma='auto',
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
            'C', lower=1e-8, upper=100.0, default_value=1.0
        )
        shrinking = CategoricalHyperparameter(
            'shrinking', [True, False], default_value=True
        )
        kernel = CategoricalHyperparameter(
            'kernel', ['linear', 'rbf', 'poly', 'sigmoid'],
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
            )
        )
        # Conditionals on hyperparameters specific to kernels.
        config.add_conditions(
            (
                InCondition(child=degree, parent=kernel, values=['poly']),
                InCondition(
                    child=coef0, parent=kernel, values=['poly', 'sigmoid']
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
            'n_neighbors', lower=1, upper=100, default_value=5
        )
        leaf_size = UniformIntegerHyperparameter(
            'leaf_size', lower=1, upper=100, default_value=20
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
