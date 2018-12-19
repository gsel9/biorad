# -*- coding: utf-8 -*-
#
# model_comparison_setup.py
#

"""
Setup model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np
import pandas as pd


def target(path_to_target, index_col=0):

    var = pd.read_csv(path_to_target, index_col=index_col)
    return np.squeeze(var.values)


def feature_set(path_to_data, index_col=0):

    data = pd.read_csv(path_to_data, index_col=index_col)
    return np.array(data.values, dtype=float)


if __name__ == '__main__':
    import sys
    # Add backend directory to PATH variable.
    sys.append('./../backend')

    import os
    import feature_selection

    from datetime import datetime
    from model_selection import nested_632plus
    from model_comparison import model_comparison

    from sklearn.metrics import precision_recall_fscore_support

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression

    """SETUP:

    METRIC:
    * Relapse/not survival are positively labeled classes.
    * Main focus is the ability to detect correctly positive samples (cancer situations).
    * Use precision and recall to focus on minority positive class.

    COMPARISON SCHEME:
    * Vallieres et al. uses .632+
    * Léger et al. uses .632

    FEATURE SELECTION:
    * Pass current estimator to permutaiton importance wrapper.

    QUESTIONS:
    - Wilcoxon test for feature selection?
        - Remove features with indicated identical distribution to target?
        - Perform Z-score transformation?
        - Bonferroni correction?

    """

    # Comparing on precision.
    LOSS = precision_recall_fscore_support

    # Mumber of OOB splits.
    NUM_REPS = 100

    # Number of repeated experiments (40 reps also used in a paper).
    NUM_ROUNDS = 40

    # Classifiers and feature selectors reported by:
    # - Zhang et al.
    # - Wu et al.
    # - Griegthuysen et al.
    # - Limkin et al.
    # - Parmar et al.
    estimators = {
        'rf': RandomForestClassifier,
        'svc': SVC,
        'logreg': LogisticRegression,
        'nb': GaussianNB,
        'plsr': PLSRegression
    }
    selectors = {
        'permutation': feature_selection.permutation_importance,
        'wlcx': feature_selection.wilcoxon_selection,
        'relieff_k10': feature_selection.relieff,
        'relieff_k30': feature_selection.relieff,
    }

    hparams = {
        'rf': {
            'n_estimators': [100, 300, 600, 1000],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced'],
            'max_depth': [None],
            'n_jobs': [-1]
        },
        'svc': {
            # class_weight: balanced by default.
            'tol': [0.0001, 0.01, 0.1, 1],
            'C': [0.001, 0.01, 1, 10],
            'kernel': ['linear', 'rbf', 'ploy'],
            'degree': [2, 3],
            'max_iter': [2000]
        },
        'logreg_l1': {
            'tol': [0.0001, 0.01, 0.1, 1],
            'C': [0.001, 0.01, 1, 10],
            'class_weight': ['balanced'],
            # Use L1 reg. to reduce dim. feature space.
            'penalty': ['l1'],
            'solver': ['liblinear'],
            'max_iter': [2000],
            'n_jobs': [-1]
        },
        'gnb': {
            'priors': [[0.677, 0.323]]
        },
        'pls': {
            'tol': [0.0001, 0.01, 0.1, 1],
            'n_components': [None],
            'max_iter': [2000]
        },
    }
    selector_params = {
        'permutation_imp': {'model': None, 'num_rounds': 1},
        'wlcx': {'thresh': 0.05},
        'relieff_k10': {'k': 5, 'n_neighbors': 20},
        'relieff_k30': {'k': 30, 'n_neighbors': 20},
    }

    # Feature data.
    X = feature_set('./data/data_to_analysis.csv')

    # Disease-Free Survival.
    y = target('./data/target_dfs.csv')

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=NUM_ROUNDS)

    # Comparison scheme.
    comparison_scheme = nested_point632plus

    _ = model_comparison(
        comparison_scheme,
        X, y,
        estimators, hparams,
        selectors, selector_params,
        random_states, NUM_REPS,
        path_to_results='./results_dfs.csv',
        score_func=LOSS
    )
