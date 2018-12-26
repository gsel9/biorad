# -*- coding: utf-8 -*-
#
# complete_decorr.py
#

"""
Model comparison experiments of decorrelated complete feature set.
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
    sys.path.append('./../model_comparison')

    import os
    import feature_selection

    from datetime import datetime
    from model_comparison import model_comparison
    from model_selection import nested_point632plus

    from sklearn.metrics import roc_auc_score

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression

    """SETUP:

    METRIC:

    Old:
    * Relapse/not survival are positively labeled classes.
    * Main focus is the ability to detect correctly positive samples
      (cancer situations).
    * Use precision (and recall) to focus on minority positive class.
    * Compute weighted (by support) average score across all samples.
    New:
    * Upsample according to Vallieres scheme and score with AUC metric.

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
    # FEATURE SET:
    X = feature_set('./../../data_source/to_analysis/complete_decorr.csv')

    # TARGET:
    #y = target('./../../data_source/to_analysis/target_lrr.csv')
    y = target('./../../data_source/to_analysis/target_dfs.csv')

    # RESULTS LOCATION:
    path_to_results = './../../data_source/experiments/no_filtering_dfs.csv'

    # SETUP:
    NUM_ROUNDS = 2#40
    NUM_SPLITS = 4#100
    EVAL = np.median
    LOSS = roc_auc_score

    estimators = {
        #'logreg': LogisticRegression,
        #'rf': RandomForestClassifier,
        #'plsr': PLSRegression,
        'gnb': GaussianNB,
        #'svc': SVC,
    }

    selectors = {
        # QUESTION: Swap with RF from paper? Drop?
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
            'max_iter': [2000],
        },
        'gnb': {
            # DFS:
            'priors': [[0.677, 0.323]]
            # LRR:
            # 'priors'. [[0.75, ]]
        },
        'plsr': {
            'tol': [0.0001, 0.01, 0.1, 1],
            'n_components': [10, 20, 30, 40],
            'max_iter': [2000]
        },
    }

    selector_params = {
        'permutation': {'num_rounds': 100},
        'wlcx': {'thresh': 0.05},
        'relieff_5': {'num_neighbors': 7, 'num_features': 5},
        'relieff_20': {'num_neighbors': 7, 'num_features': 20},
        'mrmr': {'num_features': 'auto', 'k': 5}
    }

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=NUM_ROUNDS)

    model_comparison(
        nested_point632plus,
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
        path_to_results=path_to_results
    )
