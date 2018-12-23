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
    sys.path.append('./../backend')

    import os
    import feature_selection

    from datetime import datetime
    import nested_632plus
    from model_comparison import model_comparison

    from sklearn.metrics import precision_recall_fscore_support

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression

    """SETUP:

    METRIC:
    * Relapse/not survival are positively labeled classes.
    * Main focus is the ability to detect correctly positive samples
      (cancer situations).
    * Use precision (and recall) to focus on minority positive class.
    * Compute weighted (by support) average score across all samples.

    COMPARISON SCHEME:
    * Vallieres et al. uses .632+
    * Léger et al. uses .632

    ALGORITHMS:
    * Zhang et al.
    * Wu et al.: Naive Baye’s + ReliefF
    * Griegthuysen et al.
    * Limkin et al.
    * Parmar et al.: Random Forest + Wilcoxon
    * Avonzo et al. describes studies.
    """

    # ToDos:
    # * Sequential test runs with each model to check is working.
    # * Run achieving complete overfitting of all models (may be recommended by
    #   Francois to check correctness of procedure).
    # * Iniate clinical, image and clinical + image experiments.
    # *

    # Comparing on precision.
    LOSS = precision_recall_fscore_support

    # Evaluating model general performance.
    EVAL = np.median

    # Mumber of OOB splits per level.
    NUM_SPLITS = 100

    # Number of repeated experiments (40 reps also used in a paper).
    NUM_ROUNDS = 40

    # Classifiers and feature selectors:
    estimators = {
        #'logreg': LogisticRegression,
        #'rf': RandomForestClassifier,
        #'plsr': PLSRegression,
        'gnb': GaussianNB,
        #'svc': SVC,
    }
    selectors = {
        #'permutation': feature_selection.permutation_selection,
        #'wlcx': feature_selection.wilcoxon_selection,
        'relieff_5': feature_selection.relieff_selection,
        #'relieff_20': feature_selection.relieff_selection,
        #'mrmr': feature_selection.mrmr_selection
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
        'logreg': {
            'tol': [0.0001, 0.01, 0.1, 1],
            'C': [0.001, 0.01, 1, 10],
            'class_weight': ['balanced'],
            'penalty': ['l1', 'l2'],
            'solver': ['newton-cg'],
            'max_iter': [2000],
            'n_jobs': [-1]
        },
        'gnb': {
            'priors': [[0.677, 0.323]]
        },
        'pls': {
            'tol': [0.0001, 0.01, 0.1, 1],
            'n_components': [10, 20, 30, 40],
            'max_iter': [2000]
        },
    }
    selector_params = {
        'permutation': {'num_rounds': 100},
        'wlcx': {'thresh': 0.05},
        'relieff_5': {'num_neighbors': 10, 'num_features': 5},
        'relieff_20': {'num_neighbors': 10, 'num_features': 20},
        'mrmr': {'num_features': 'auto', 'k': 5}
    }
    # Feature data.
    X = feature_set('./../../../data/to_analysis/clinical_params.csv')

    # Disease-Free Survival.
    y = target('./../../../data/to_analysis/target_dfs.csv')

    # Comparison scheme.
    comparison_scheme = nested_632plus.nested_point632plus

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=NUM_ROUNDS)

    _ = model_comparison(
        comparison_scheme,
        X, y,
        NUM_SPLITS,
        random_states,
        estimators,
        estimator_params,
        selectors,
        selector_params,
        score_func=LOSS,
        score_eval=EVAL,
        verbose=0,
        n_jobs=None,
        path_to_results='./results/original_dfs.csv',
    )
