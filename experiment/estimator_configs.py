# -*- coding: utf-8 -*-
#
# classification.py
#

"""
Classification algorithm setup including hyperparameter configurations.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


from backend import hyperparams

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression


# Globals
CLF_LABEL = 'clf'
NAME_FUNC = lambda param_name: '{}__{}'.format(CLF_LABEL, param_name),


classifiers = {
    # Support Vector Machines
    SVC.__name__: {
        'pipe': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, SVC())
        ],
        'params': hyperparams.svc_param_space(
            NAME_FUNC,
            gamma=None,
            degree=None,
            tol=None,
            C=None,
            shrinking=None,
            coef0=None,
            kernel=None,
            n_features=1,
            class_weight='balanced',
            max_iter=-1,
            verbose=False,
            cache_size=512
        ),
    },
    # Random Forest Classifier
    RandomForestClassifier.__name__: {
        'pipe': [
            (CLF_LABEL, RandomForestClassifier())
        ],
        'params': hyperparams.trees_param_space(
            NAME_FUNC,
            n_estimators=None,
            max_features=None,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
            bootstrap=None,
            oob_score=False,
            n_jobs=-1,
            verbose=False,
        )
    },
    # Gaussian Naive Bayes.
    GaussianNB.__name__: {
        'pipe': [
            ('{]_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, GaussianNB())
        ],
        'params': hyperparams.gnb_param_space(
            name_func, priors=None, var_smoothing=None
        )
    },
    # Logistic Regression
    LogisticRegression.__name__: {
        'pipe': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, LogisticRegression())
        ],
        'params': hyperparams.logreg_hparam_space(
            NAME_FUNC,
            penalty=None,
            C=None,
            tol=None,
            dual=False,
            solver='liblinear',
            fit_intercept=True,
            intercept_scaling=1,
            class_weight='balanced',
            multi_class='ovr',
            warm_start=False,
            max_iter=-1,
            verbose=0,
            n_jobs=-1
        )
    },
    # Partial Least Squares Regression
    PLSRegression.__name__: {
        'pipe': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PLSRegression())
        ],
        'params': hyperparams.plsr_hparam_space(
            NAME_FUNC,
            n_components=None,
            tol=None,
            n_features=1,
            max_iter=-1,
            scale=True,
            copy=True
        )
    }
}
