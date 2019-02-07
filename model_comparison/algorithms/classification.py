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
from sklearn.cross_decomposition import PLSRegression

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter


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
            )
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

        # NOTE: This algorithm is not stochastic and its performance does not
        # varying depending on a random number generator.
        hparam_space = (
            # Hyperparameters shared by all kernels
            UniformFloatHyperparameter(
                'svc__C', 0.001, 1000.0, default_value=1.0
            ),
            CategoricalHyperparameter(
                'svc__shrinking', ['true', 'false'], default_value='true'
            ),
            # Hyperparameters specific to kernels.
            CategoricalHyperparameter(
                'svc__kernel', ['linear', 'rbf', 'poly', 'sigmoid'],
            ),
            # - Poly kernel only:
            UniformIntegerHyperparameter('svc__degree', 1, 5, default_value=3),
            # - Poly and sigmoid kernels:
            UniformFloatHyperparameter
                ('svc__coef0', 0.0, 10.0, default_value=0.0
            ),
            # - RBF, poly and sigmoid kernels.
            CategoricalHyperparameter(
                'gamma', ['auto', 'value'], default_value='auto'
            ),
            UniformFloatHyperparameter(
                'gamma_value', 0.0001, 8, default_value=1
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


if __name__ == '__main__':
    pass

    """
    # Gaussian Naive Bayes:
    GaussianNB.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(GaussianNB()))
        ],
        'params': hyperparams.gnb_param_space(
            estimator_name_func, priors=None, var_smoothing=None
        )
    },

    AdaBoostClassifier.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                AdaBoostClassifier(base_estimator=LogisticRegression())
            ))
        ],
        'params': hyperparams.adaboost_param_space(
            estimator_name_func,
            n_estimators=None,
            learning_rate=None,
            random_state=None
        )
    },

    # Logistic Regression:
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
    # REF: sklearn docs.
    LogisticRegression.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                LogisticRegression(
                    solver='liblinear',
                    max_iter=1000,
                    verbose=0,
                    n_jobs=1,
                    dual=False,
                    multi_class='ovr',
                    warm_start=False,
                    class_weight='balanced',
                )
            ))
        ],
        'params': hyperparams.logreg_hparam_space(
            estimator_name_func,
            penalty=None,
            C=None,
            tol=None,
            random_state=None,
            fit_intercept=True,
            intercept_scaling=None,
        )
    },
    KNeighborsClassifier.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                KNeighborsClassifier(algorithm='auto')
            ))
        ],
        'params': hyperparams.knn_param_space(
            estimator_name_func,
            n_neighbors=None,
            weights=None,
            leaf_size=None,
            metric=None,
            p=None,
        )
    },
    DecisionTreeClassifier.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                DecisionTreeClassifier(class_weight='balanced',)
            ))
        ],
        'params': hyperparams.decision_tree_param_space(
            estimator_name_func,
            criterion=None,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
        )
    },
    # Linear Support Vector Machines:
    # * Must select kernel a priori because the hyperparamter space
    #   generator function is not evaluated for each suggested
    #   configuration. Thus, settings depending on the specified kernel
    #   will not be updated according to the sampled kernel function.
    LinearSVC.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                LinearSVC(
                    class_weight='balanced',
                    verbose=False,
                    max_iter=-1,
                    multi_class='ovr'
                )
            ))
        ],
        'params': hyperparams.linear_svc_param_space(
            estimator_name_func,
            penalty=None,
            loss=None,
            dual=None,
            tol=None,
            C=None,
            fit_intercept=True,
            intercept_scaling=None,
            random_state=None,
        ),
    },
    # Random Forest Classifier:
    RandomForestClassifier.__name__: {
        'estimator': [
            (CLF_LABEL, PipeEstimator(
                RandomForestClassifier(
                    n_jobs=-1,
                    verbose=False,
                    oob_score=False,
                    class_weight='balanced',
                )
            ))
        ],
        'params': hyperparams.trees_param_space(
            estimator_name_func,
            n_estimators=None,
            max_features=None,
            criterion=None,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
            bootstrap=None,
            random_state=None,
        )
    },
    """
