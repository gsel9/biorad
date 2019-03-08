# -*- coding: utf-8 -*-
#
# classifiers.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

# Shallow learners.
from pyglmnet import GLM
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression

# Tsetlin machine.
import numpy as np
import pyximport

import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)
#import TsetlinMachine


# Gradient boosting trees.
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

# Hyperparameter configs. Installation successfull by conda instal gcc and swig.
from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

try:
    from . import base
except:
    import base


class GroupLASSO(base.BaseClassifier):

    SEED = 0
    NAME = 'GroupLASSO'

    def __init__(
        self,
        group_idx,
        model=None,
        with_selection: bool=False,
        scoring='roc_auc',
        cv: int=0,
        forward: bool=True,
        floating: bool=False,
    ):

        # Hack:
        self.model = GLM(
            distr='binomial',
            tol=None,
            group=group_idx,
            alpha=None
        )
        super().__init__(
            model=self.model,
            with_selection=with_selection,
            scoring=scoring,
            cv=cv,
            forward=forward,
            floating=floating
        )
        self.with_selection = with_selection

    def get_params(self, deep=True):

        return {
            'alpha': self.model.alpha,
            'tol': self.model.tol
        }

    def set_params(self, **params):
        """Update estimator hyperparamter configuration.

        Kwargs:
            params (dict): Hyperparameter settings.

        """
        params = self._check_config(params)

        for key, value in params.items():
            if key == 'alpha':
                self.model.alpha = value
            elif key == 'tol':
                self.model.tol = value

        return self

    @property
    def config_space(self):
        """
        TODO
        """
        alpha = UniformFloatHyperparameter(
            'alpha', lower=0.0, upper=10.0, default_value=1.0
        )
        tol = UniformFloatHyperparameter(
            'tol', lower=1e-9, upper=1e-3, default_value=1e-7
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters(
            (
                alpha,
                tol
            )
        )
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config


class LightGBM(base.BaseClassifier):

    SEED = 0
    NAME = 'LightGBM'

    def __init__(
        self,
        model=None,#LGBMClassifier(
        #    boosting_type='gbdt',
        #    class_weight='balanced',
        #    objective='binary',
        #    n_jobs=-1
        #),
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
        """
        TODO
        """

        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=2, upper=3000, default_value=100
        )
        # The maximum depth of a tree.
        num_leaves = UniformIntegerHyperparameter(
            'num_leaves', lower=2, upper=3000, default_value=50
        )
        min_data_in_leaf = UniformIntegerHyperparameter(
            'min_data_in_leaf', lower=2, upper=100, default_value=50
        )
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', lower=1e-10, upper=5 - 1e-10, default_value=0.1
        )
        # L1 regularization term on weights.
        reg_alpha = UniformFloatHyperparameter(
            'reg_alpha', lower=1e-10, upper=5 - 1e-10, default_value=1e-3
        )
        # L2 regularization term on weights.
        reg_lambda = UniformFloatHyperparameter(
            'reg_lambda', lower=1e-10, upper=5 - 1e-10, default_value=1e-3
        )
        # The maximum depth of a tree.
        max_depth = UniformIntegerHyperparameter(
            'max_depth', lower=2, upper=1000, default_value=100
        )
        min_child_weight = UniformIntegerHyperparameter(
            'min_child_weight', lower=1, upper=10, default_value=1
        )
        min_child_samples = UniformIntegerHyperparameter(
            'min_child_samples', lower=2, upper=50, default_value=20
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters(
            (
                n_estimators,
                num_leaves,
                #min_data_in_leaf,
                min_child_weight,
                min_child_samples,
                learning_rate,
                reg_alpha,
                reg_lambda,
                max_depth,
            )
        )
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config


class XGBoosting(base.BaseClassifier):
    """eXtreme Gradient Boosting"""

    SEED = 0
    NAME = 'XGBoost'

    def __init__(
        self,
        model=None,#XGBClassifier(
            #missing=None,
            #booster='gbtree',
            #objective='binary:logistic',
            #eval_metric='auc',
            #n_jobs=-1
        #),
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
        """
        TODO
        """
        n_estimators = UniformIntegerHyperparameter(
            'n_estimators', lower=2, upper=3000, default_value=100
        )
        # The minimum sum of weights of all observations required in a child.
        min_child_weight = UniformIntegerHyperparameter(
            'min_child_weight', lower=1, upper=10, default_value=1
        )
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', lower=1e-10, upper=5 - 1e-10, default_value=0.1
        )
        # L1 regularization term on weights.
        reg_alpha = UniformFloatHyperparameter(
            'reg_alpha', lower=1e-10, upper=5 - 1e-10, default_value=1e-3
        )
        # L2 regularization term on weights.
        reg_lambda = UniformFloatHyperparameter(
            'reg_lambda', lower=1e-10, upper=5 - 1e-10, default_value=1e-3
        )
        # The minimum loss reduction required to make a split.
        gamma = UniformFloatHyperparameter(
            'gamma', lower=1e-10, upper=1e-3, default_value=0
        )
        # The minimum loss reduction required to make a split.
        max_delta_step = UniformIntegerHyperparameter(
            'max_delta_step', lower=0, upper=100, default_value=0
        )
        # The maximum depth of a tree.
        max_depth = UniformIntegerHyperparameter(
            'max_depth', lower=2, upper=1000, default_value=100
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters(
            (
                n_estimators,
                min_child_weight,
                learning_rate,
                reg_alpha,
                reg_lambda,
                gamma,
                max_delta_step,
                max_depth,
            )
        )
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config


class TsetlinMachineWrapper(base.BaseClassifier):

    SEED = 0
    NAME = 'TsetlinMachine'

    def __init__(
        self,
        model=None,#TsetlinMachine.TsetlinMachine(),
        with_selection: bool=False,
        scoring='roc_auc',
        cv: int=0,
        num_epochs: int=500,
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
        self.num_epochs = num_epochs
        self.with_selection = with_selection

    @property
    def config_space(self):
        """Returns the RF Regression hyperparameter space."""

        threshold = UniformFloatHyperparameter(
            'threshold', lower=2, upper=100, default_value=10
        )
        precision = UniformFloatHyperparameter(
            'precision', lower=1, upper=10, default_value=3
        )
        number_of_clauses = UniformIntegerHyperparameter(
            'number_of_clauses', lower=10, upper=int(1e5), default_value=300
        )
        number_of_features = UniformIntegerHyperparameter(
            'number_of_features', lower=10, upper=int(1e5),
        )
        states = UniformIntegerHyperparameter(
            'states', lower=10, upper=1000, default_value=100
        )
        epochs = UniformIntegerHyperparameter(
            'epochs', lower=100, upper=1000, default_value=500
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameters(
            (
                threshold,
                precision,
                number_of_clauses,
                number_of_features,
                states,
                epochs
            )
        )
        # Add additional hyperparameter for a number of feature to select.
        if self.with_selection:
            num_features = UniformIntegerHyperparameter(
                'num_features', lower=2, upper=50, default_value=20
            )
            config.add_hyperparameter(num_features)

        return config

    def fit(self, X, y, **kwargs):

        self.model.fit(X, y, np.int32(y.size), np.int32(self.num_epochs))

        return self

    def predict(self, X):

        y_preds = []
        for x in X:
            y_pred = self.model.predict(X)
            y_preds.append(y_pred)

        return np.array(y_preds, dtype=int)

    def get_params(self, deep=True):
        return {
            'number_of_clauses': self.model.get_number_of_clauses(),
            'number_of_features': self.model.get_number_of_features(),
            'number_of_states': self.model.get_number_of_states(),
            's': self.model.get_s(),
            'threshold': self.model.get_threshold(),
        }

    def set_params(self, **params):
        """Update estimator hyperparamter configuration.

        Kwargs:
            params (dict): Hyperparameter settings.

        """
        params = self._check_config(params)

        for key, value in params.items():
            if key == 'number_of_clauses':
                self.model.set_number_of_clauses(value)
            elif key == 'number_of_features':
                self.model.set_number_of_features(value)
            elif key == 'number_of_states':
                self.model.set_number_of_states(value)
            elif key == 's':
                self.model.set_s(value)
            elif key == 'threshold':
                self.model.set_threshold(value)

        return self


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

    from sklearn.datasets import make_classification

    _X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        shuffle=True,
        random_state=0
    )
    #y = y.astype(np.int32)
    #X = np.zeros_like(_X, dtype=np.int32)
    #X[X > 0.5] = 1
    #m = TsetlinMachineWrapper()
    #m.fit(X, y)
    #y_pred = m.predict(y)
    #print(y_pred)
    #print(sum(y_pred == y) / np.size(y))
    m = GLM(group=[1,1,2])
    print(m.group)
