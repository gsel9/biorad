# -*- coding: utf-8 -*-
#
# classifiers.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression

from tsetlinmachine import TsetlinLayer

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

import base
import pyximport

import numpy as np

# To import Cython modules.
pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True
)


class TsetlinMachine(base.BaseClassifier):

    SEED = 0
    NAME = 'TsetlinMachine'

    def __init__(
        self,
        model=TsetlinLayer(None, None, None, None, None)
    ):

        self.model = model

    @property
    def config_space(self):
        """Returns the RF Regression hyperparameter space."""

        threshold = UniformFloatHyperparameter(
            'threshold', lower=, upper=,
        )
        precision = UniformFloatHyperparameter(
            'precision', lower=, upper=,
        )
        number_of_clauses = UniformIntegerHyperparameter(
            'number_of_clauses', lower=, upper=,
        )
        states = UniformIntegerHyperparameter(
            'states', lower=, upper=,
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters(
            (
                criterion,
                max_depth,
                max_features,
                min_samples_leaf
            )
        )
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config


    def set_params(self, **params):
        """Update estimator hyperparamter configuration.

        Kwargs:
            params (dict): Hyperparameter settings.

        """
        params = self._check_config(params)
        self.model.set_params(**params)

        return self

    def get_params(self, deep=True):
        """Returns hyperparameter configurations.

        """
        return self.model.get_params(deep=deep)

    def fit(self, X, y=None, **kwargs):
        """Train classifier with optional sequential feature selection to
        reduce in the input feature space.

        """
        self._check_params(X, y)
        if self.with_selection:
            model = deepcopy(self.model)
            selector = sffs.SequentialFeatureSelector(
                estimator=model,
                k_features=self.num_features,
                forward=self.forward,
                floating=self.floating,
                scoring=self.scoring,
                cv=self.cv
            )
            selector.fit(X, y)
            self.support = np.array(selector.k_feature_idx_, dtype=int)
            # Check hyperparameter setup with the reduced feature set.
            self._check_params(X[:, self.support], y)
            self.model.fit(X[:, self.support], y, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)

        return self

    def predict(self, X):
        """Generate model prediction.

        """
        if self.with_selection:
            y_pred = np.squeeze(self.model.predict(X[:, self.support]))
        else:
            y_pred = np.squeeze(self.model.predict(X))

        return np.array(y_pred, dtype=int)


class DTreeEstimator(base.BaseClassifier):

    SEED = 0
    NAME = 'DTreeEstimator'

    def __init__(
        self,
        model=DecisionTreeClassifier(
            min_samples_split=2,
            class_weight='balanced',
        ),
        with_selection: bool=False,
        scoring='roc_auc',
        cv: int=0,
        forward: bool=True,
        floating: bool=False,
    ):

        super().__init__(
            model=model,
            with_selection=with_selection,
            scoring=scoring,
            cv=cv,
            forward=forward,
            floating=floating
        )
        self.with_selection = with_selection

    @property
    def config_space(self):
        """Returns the RF Regression hyperparameter space."""

        criterion = CategoricalHyperparameter(
            'criterion', ['gini', 'entropy'], default_value='gini'
        )
        max_depth = CategoricalHyperparameter(
            'max_depth', [3, 5, 10, 20, None], default_value=None
        )
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', None], default_value=None
        )
        min_samples_leaf = UniformFloatHyperparameter(
            'min_samples_leaf', lower=1e-6, upper=0.5 - 1e-10,
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters(
            (
                criterion,
                max_depth,
                max_features,
                min_samples_leaf
            )
        )
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config


class RFEstimator(base.BaseClassifier):

    SEED = 0
    NAME = 'RFEstimator'

    def __init__(
        self,
        model=RandomForestClassifier(
            n_jobs=-1,
            verbose=False,
            oob_score=False,
            min_samples_split=2,
            class_weight='balanced',
        ),
        with_selection: bool=False,
        scoring='roc_auc',
        cv: int=0,
        forward: bool=True,
        floating: bool=False,
    ):
        super().__init__(
            model=model,
            with_selection=with_selection,
            scoring=scoring,
            cv=cv,
            forward=forward,
            floating=floating
        )
        self.with_selection = with_selection

    @property
    def config_space(self):
        """Returns the RF Regression hyperparameter space."""

        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=2, upper=3000, default_value=100
        )
        criterion = CategoricalHyperparameter(
            'criterion', ['gini', 'entropy'], default_value='gini'
        )
        max_depth = CategoricalHyperparameter(
            'max_depth', [3, 5, 10, 20, None], default_value=None
        )
        max_features = CategoricalHyperparameter(
            'max_features', ['auto', 'sqrt', 'log2', None], default_value=None
        )
        bootstrap = CategoricalHyperparameter(
            'bootstrap', [True, False], default_value=True
        )
        min_samples_leaf = UniformFloatHyperparameter(
            'min_samples_leaf', lower=1e-6, upper=0.5 - 1e-10,
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
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
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config


class PLSREstimator(base.BaseClassifier):

    SEED = 0
    NAME = 'PLSREstimator'

    def __init__(
        self,
        model=PLSRegression(
            scale=False,
            copy=True,
            max_iter=int(1e4)
        ),
        with_selection: bool=False,
        scoring='roc_auc',
        cv: int=0,
        forward: bool=True,
        floating: bool=False,
    ):
        super().__init__(
            model=model,
            with_selection=with_selection,
            scoring=scoring,
            cv=cv,
            forward=forward,
            floating=floating
        )
        self.with_selection = with_selection

    @property
    def config_space(self):
        """Returns the PLS regression hyperparameter configuration space."""

        tol = UniformFloatHyperparameter(
            'tol', lower=1e-9, upper=1e-3, default_value=1e-7
        )
        n_components = UniformIntegerHyperparameter(
            'n_components', lower=1, upper=40, default_value=27
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters((tol, n_components))
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

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

    SEED = 0
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
        with_selection: bool=False,
        scoring='roc_auc',
        cv: int=0,
        forward: bool=True,
        floating: bool=False,
    ):
        super().__init__(
            model=model,
            with_selection=with_selection,
            scoring=scoring,
            cv=cv,
            forward=forward,
            floating=floating
        )
        self.with_selection = with_selection

    @property
    def config_space(self):
        """Returns the LR hyperparameter configuration space."""

        C_param = UniformFloatHyperparameter(
            'C', lower=0.001, upper=1000.0, default_value=1.0
        )
        penalty = CategoricalHyperparameter(
            'penalty', ['l1', 'l2'], default_value='l2'
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters((C_param, penalty))
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config


class SVCEstimator(base.BaseClassifier):

    SEED = 0
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
        with_selection: bool=False,
        scoring='roc_auc',
        cv: int=0,
        forward: bool=True,
        floating: bool=False,
    ):
        super().__init__(
            model=model,
            with_selection=with_selection,
            scoring=scoring,
            cv=cv,
            forward=forward,
            floating=floating
        )
        self.with_selection = with_selection

    @property
    def config_space(self):
        """Returns the SVC hyperparameter configuration space."""

        C_param = UniformFloatHyperparameter(
            'C', lower=0.001, upper=1000.0, default_value=1.0
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
            'gamma_value', lower=0.0001, upper=8, default_value=1
        )
        degree = UniformIntegerHyperparameter(
            'degree', lower=1, upper=5, default_value=3
        )
        coef0 = UniformFloatHyperparameter(
            'coef0', lower=0.0, upper=10.0, default_value=0.0
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
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
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)
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


# NOTE: This algorithm does not associate hyperparameters.
class GNBEstimator(base.BaseClassifier):

    SEED = 0
    NAME = 'GNBEstimator'

    def __init__(
        self,
        model=GaussianNB(),
        with_selection=False,
        scoring='roc_auc',
        cv: int=0,
        forward: bool=True,
        floating: bool=False,
    ):

        super().__init__(
            model=model,
            with_selection=with_selection,
            scoring=scoring,
            cv=cv,
            forward=forward,
            floating=floating
        )
        self.with_selection = with_selection

    @property
    def config_space(self):
        """Returns the KNN hyperparameter configuration space."""

        config = ConfigurationSpace()
        config.seed(self.SEED)
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config


class KNNEstimator(base.BaseClassifier):

    SEED = 0
    NAME = 'KNNEstimator'

    def __init__(
        self,
        model=KNeighborsClassifier(algorithm='auto'),
        with_selection: bool=False,
        scoring='roc_auc',
        cv: int=0,
        forward: bool=True,
        floating: bool=False,
    ):
        super().__init__(
            model=model,
            with_selection=with_selection,
            scoring=scoring,
            cv=cv,
            forward=forward,
            floating=floating
        )
        self.with_selection = with_selection

    @property
    def config_space(self):
        """Returns the KNN hyperparameter configuration space."""

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
        p_param = UniformIntegerHyperparameter('p', 1, 5, default_value=2)
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters((n_neighbors, leaf_size, metric, p_param))
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)
        # Conditionals on hyperparameters specific to kernels.
        config.add_condition(
            InCondition(child=p_param, parent=metric, values=['minkowski'])
        )
        return config


if __name__ == '__main__':
    pass
