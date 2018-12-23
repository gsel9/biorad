# -*- coding: utf-8 -*-
#
# test_feature_selection.py
#

"""
Feature selection test module.
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


def test_permutation_importance(data, clf):

    X_train, X_test, y_train, y_test = data

    X_train_std, X_test_std, support = selection.permutation_selection(
        X_train, X_test, y_train, y_test,
        roc_auc_score, clf,
        num_rounds=5, random_state=0
    )
    assert len(support) < 3


def test_wilcoxon(data):

    X_train, X_test, y_train, y_test = data

    X_train_std, X_test_std, support = selection.wilcoxon_selection(
        X_train, X_test, y_train, y_test, thresh=1
    )
    assert len(support) == 3

    X_train_std, X_test_std, support = selection.wilcoxon_selection(
        X_train, X_test, y_train, y_test, thresh=0.01
    )
    assert len(support) <= 3


def test_relieff(data):

    X_train, X_test, y_train, y_test = data

    X_train_std, X_test_std, support = selection.relieff(
        X_train, X_test, y_train, y_test, num_neighbors=3, num_features=1
    )
    assert len(support) == 1

    X_train_std, X_test_std, support = selection.relieff(
        X_train, X_test, y_train, y_test, num_neighbors=3, num_features=2
    )
    assert len(support) == 2


def test_mrmr(data):

    X_train, X_test, y_train, y_test = data

    X_train_std, X_test_std, support = selection.mrmr_selection(
        X_train, X_test, y_train, y_test, num_features=1
    )
    assert len(support) == 1

    X_train_std, X_test_std, support = selection.mrmr_selection(
        X_train, X_test, y_train, y_test, num_features='auto'
    )
    assert len(support) <= 3


def test_selector(data):

    def dummy(X_train, X_test, y_train, y_test, param1, param2):

        support = np.arange(X_train.shape[1], dtype=int)

        return X_train, X_test, support

    X_train, X_test, y_train, y_test = data

    selector = selection.Selector('dummy', dummy, {'param1': 1, 'param2': 3})
    X_train, X_test, support = selector(X_train, X_test, y_train, y_test)

    assert len(support) == 3
