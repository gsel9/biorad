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

    ALGORITHMS:
    * Zhang et al.
    * Wu et al.
    * Griegthuysen et al.
    * Limkin et al.
    * Parmar et al.

    FEATURE SELECTION:
    * Pass current estimator to permutaiton importance wrapper.

    QUESTIONS:
    - Wilcoxon test for feature selection?
        - Remove features with indicated identical distribution to target?
        - Perform Z-score transformation?
        - Bonferroni correction?
    - Random sampling of hparams?
    - Feature selection across parameter grid and select most stable features?

    """

    # Comparing on precision.
    LOSS = precision_recall_fscore_support

    # Mumber of OOB splits per level.
    NUM_REPS = 100

    # Number of repeated experiments (40 reps also used in a paper).
    NUM_ROUNDS = 40

    # Classifiers and feature selectors:
    estimators = {
        'logreg': LogisticRegression,
        'rf': RandomForestClassifier,
        'plsr': PLSRegression,
        'nb': GaussianNB,
        'svc': SVC,
    }
    selectors = {
        'permutation': feature_selection.permutation_importance,
        'wlcx': feature_selection.wilcoxon_selection,
        'relieff_k10_n5': feature_selection.relieff,
        'relieff_k30_n5': feature_selection.relieff,
        'relieff_k10_n20': feature_selection.relieff,
        'relieff_k30_n20': feature_selection.relieff,
        'mutual_info_k10_n5': feature_selection.mutual_info,
        'mutual_info_k30_n5': feature_selection.mutual_info,
        'mutual_info_k10_n20': feature_selection.mutual_info,
        'mutual_info_k30_n20': feature_selection.mutual_info
    }
    estimator_params = {
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
        'relieff_k10': {'num_features': 10, 'num_neighbors': 5},
        'relieff_k30': {'num_features': 30, 'num_neighbors': 5},
        'relieff_k10': {'num_features': 10, 'num_neighbors': 20},
        'relieff_k30': {'num_features': 30, 'num_neighbors': 20},
        'mutual_info_k10': {'num_features': 10, 'num_neighbors': 20},
        'mutual_info_k30': {'num_features': 30, 'num_neighbors': 20},
        'mutual_info_k10': {'num_features': 10, 'num_neighbors': 5},
        'mutual_info_k30': {'num_features': 30, 'num_neighbors': 5},
    }

    # Feature data.
    X = feature_set('./../../../data/to_analysis/clinical_params.csv')

    # Disease-Free Survival.
    y = target('./../../../data/to_analysis/target_dfs.csv')

    # Comparison scheme.
    comparison_scheme = nested_point632plus

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=NUM_ROUNDS)

    _ = model_comparison(
        comparison_scheme,
        X, y,
        estimators, estimator_params,
        selectors, selector_params,
        random_states, NUM_REPS,
        path_to_results='./results/dfs.csv',
        score_func=LOSS
    )
