# -*- coding: utf-8 -*-
#
# test_model_selection.py
#

"""
Model selection test module.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import sys
sys.path.append('../')

import pytest

import numpy as np
import feature_selection as selection

from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@pytest.fixture
def clf():

    return RandomForestClassifier(random_state=0)


@pytest.fixture
def data():

    X, y = make_classification(
        n_samples=100,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
        shuffle=True
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Demo run:
    # * 97.80 % accuracy seems to be a fairly good score.
    #
    # TODO: Need seed + clf + selector name for unique ID to prelim results files.

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # TEMP:
    from sklearn.feature_selection.univariate_selection import SelectPercentile, chi2
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    pipe = Pipeline([
        ('kbest', SelectPercentile(chi2)),
        ('clf_scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=0))
    ])

    # Can specify hparam distr in config files that acn direclty be read into
    # Python dict with hyperopt distirbutions?

    # Parameter search space
    space = {}
    # Random number between 50 and 100
    space['kbest__percentile'] = hp.uniform('kbest__percentile', 50, 100)
    # Random number between 0 and 1
    #space['clf__l1_ratio'] = hp.uniform('clf__l1_ratio', 0.0, 1.0)
    # Log-uniform between 1e-9 and 1e-4
    #space['clf__alpha'] = hp.loguniform('clf__alpha', -9*np.log(10), -4*np.log(10))
    # Random integer in 20:5:80
    #space['clf__n_iter'] = 20 + 5 * hp.randint('clf__n_iter', 12)
    # Random number between 50 and 100
    space['clf__class_weight'] = hp.choice('clf__class_weight', [None,]) #'balanced']),
    # Discrete uniform distribution
    space['clf__max_leaf_nodes'] = scope.int(hp.quniform('clf__max_leaf_nodes', 30, 150, 1))
    # Discrete uniform distribution
    space['clf__min_samples_leaf'] = scope.int(hp.quniform('clf__min_samples_leaf', 20, 500, 5))

    results = bbc_cv_selection(
        X_train, y_train,
        algo=tpe.suggest,
        model_id='',
        model=pipe,
        param_space=space,
        score_func=roc_auc_score,
        path_tmp_results='./',
        cv=5,
        oob=5,
        max_evals=7,
        shuffle=True,
        verbose=1,
        random_state=0,
        alpha=0.05,
        balancing=True,
        write_prelim=False,
        error_score=np.nan
    )
