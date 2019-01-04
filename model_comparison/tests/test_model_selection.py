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

    from sklearn.datasets import load_breast_cancer

    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    from sklearn.feature_selection.univariate_selection import chi2
    from sklearn.feature_selection.univariate_selection import SelectPercentile

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

    # Parameter search space
    space = {}
    space['clf__max_leaf_nodes'] = scope.int(
        hp.quniform('clf__max_leaf_nodes', 30, 150, 1)
    )
    space['clf__min_samples_leaf'] = scope.int(
        hp.quniform('clf__min_samples_leaf', 20, 500, 5)
    )
    space['clf__class_weight'] = hp.choice('clf__class_weight', [None,])
    space['kbest__percentile'] = hp.uniform('kbest__percentile', 50, 100)

    results = model_selection(
        X_train, y_train,
        algo=tpe.suggest,
        model=pipe,
        space=space,
        score_func=roc_auc_score,
        path_tmp_results='',
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
