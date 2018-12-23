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


def test_wilcoxon(data, clf):

    X_train, X_test, y_train, y_test = data

    X_train_std, X_test_std, support = selection.wilcoxon_selection(
        X_train, X_test, y_train, y_test, thresh=1
    )
    assert len(support) == 3

    X_train_std, X_test_std, support = selection.wilcoxon_selection(
        X_train, X_test, y_train, y_test, thresh=0.01
    )
    assert len(support) <= 3


def test_relieff(data, clf):

    X_train, X_test, y_train, y_test = data
