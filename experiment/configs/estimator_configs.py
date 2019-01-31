# -*- coding: utf-8 -*-
#
# estimator_configs.py
#

"""
Classification algorithm setup including hyperparameter configurations.

NB: Make sure to update the number of original features in the data set.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import sys
sys.path.append('./..')

import numpy as np

from hyperopt.pyll import scope

from backend import hyperparams
from backend.formatting import PipeEstimator

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier


# Globals
CLF_LABEL = 'clf'
NUM_ORIG_FEATURES = 1200


@scope.define
def estimator_name_func(param_name):

    global CLF_LABEL

    return '{}__{}'.format(CLF_LABEL, param_name)


classifiers = {
    KNeighborsClassifier.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                KNeighborsClassifier(algorithm='auto')
            ))
        ],
        'params': hyperparams.knn_param_space(
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
    AdaBoostClassifier.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                AdaBoostClassifier(
                    base=LogisticRegression(), n_estimators=1000
                )
            ))
        ],
        'params': hyperparams.adaboost_param_space(
            estimator_name_func,
            n_estimators=None,
            learning_rate=None,
            random_state=None
        )
    },
    # Support Vector Machines:
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
                    decision_function_shape='ovr'
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
    },
    SVC.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                SVC(
                    class_weight='balanced',
                    verbose=False,
                    cache_size=500,
                    max_iter=-1,
                    decision_function_shape='ovr',
                )
            ))
        ],
        'params': hyperparams.svc_param_space(
            estimator_name_func,
            kernel='rbf'
            gamma=None,
            degree=None,
            tol=None,
            C=None,
            shrinking=None,
            coef0=None,
            random_state=None,
            n_features=NUM_ORIG_FEATURES,
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
    # Partial Least Squares Regression:
    # * Recommended to use copy = True.
    # REF: sklearn docs
    PLSRegression.__name__: {
        'estimator': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PipeEstimator(
                PLSRegression(scale=True, copy=True, max_iter=-1))
            )
        ],
        'params': hyperparams.plsr_hparam_space(
            estimator_name_func,
            n_components=None,
            tol=None,
        )
    }
}
