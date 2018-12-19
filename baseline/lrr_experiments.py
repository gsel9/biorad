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
    N_REPS = 100

    # Number of repeated experiments (40 reps also used in a paper).
    n_experiments = 40

    # Classifiers and feature selectors reported by:
    # - Zhang et al.
    # - Wu et al.
    # - Griegthuysen et al.
    # - Limkin et al.
    # - Parmar et al.
    estimators = {
        'rf': RandomForestClassifier,
        'svc': SVC,
        'lr': LogisticRegression,
        'nb': GaussianNB,
        'plsr': PLSRegression
    }
    selectors = {
        'permutation': feature_selection.permutation_importance,
        'wlcx': feature_selection.wilcoxon_selection,
        'relieff_k10': feature_selection.relieff,
        'relieff_k30': feature_selection.relieff,
    }

    # Shared hyperparameters:
    MAX_ITER = [1500]
    PENALTY = ['l2']
    SOLVER = ['lsqr']
    CLASS_WEIGHT = ['balanced']
    C = [0.001, 0.01, 0.1, 1, 10, 100, 300]
    TOL = [0.001, 0.01, 0.1, 0.3, 0.7, 1]
    N_ESTIMATORS = [20, 50, 100, 500, 1000]
    LEARNINGR_RATE = [0.001, 0.05, 0.2, 0.6, 1]

    # Priors for both target variables.
    PFS_PRIORS = [0.677, 0.323]
    LRC_PRIORS = [0.753, 0.247]

    selector_params = {
        'permutation_imp': {'model': None, 'num_rounds': 1},
        'wlcx': {'thresh': 0.05},
        'relieff_k5': {'k': 5, 'n_neighbors': 20},
        'relieff_k25': {'k': 25, 'n_neighbors': 20},
    }




    pfs_hparams = {
        'adaboost': {
            'n_estimators': N_ESTIMATORS, 'learning_rate': LEARNINGR_RATE
        },
        'lda': {
            # NOTE: n_components determined in model selection work function.
            'n_components': [None], 'tol': TOL, 'priors': [PFS_PRIORS],
            'solver': SOLVER
        },
        'qda': {
            'priors': [PFS_PRIORS], 'tol': TOL
        },
        'pls': {
            'n_components': [None], 'tol': TOL,
        },
        'gnb': {
            'priors': [PFS_PRIORS]
        },
        'logreg': {
            'C': C, 'solver': ['sag'], 'penalty': PENALTY,
            'class_weight': CLASS_WEIGHT, 'max_iter': MAX_ITER
        },
    }


    path_to_results = './results/results_pfs.csv'

    # Feature data.
    X = feature_set('./data/data_to_analysis.csv')

    # Disease-Free Survival.
    y = target('./data/target_dfs.csv')

    
    results_pfs = model_comparison(
        comparison_scheme, X, y_pfs, estimators, pfs_hparams, selectors,
        selector_params, random_states, N_REPS, path_to_pfsresults,
        score_func=SCORE
    )

    lrc_hparams = {
        'adaboost': {
            'n_estimators': N_ESTIMATORS, 'learning_rate': LEARNINGR_RATE
        },
        'lda': {
            # NOTE: n_components determined in model selection work function.
            'n_components': [None], 'tol': TOL, 'priors': [LRC_PRIORS],
            'solver': SOLVER
        },
        'qda': {
            'priors': [LRC_PRIORS], 'tol': TOL
        },
        'pls': {
            'n_components': [None], 'tol': TOL,
        },
        'gnb': {
            'priors': [LRC_PRIORS]
        },
        'logreg': {
            'C': C, 'solver': ['sag'], 'penalty': PENALTY,
            'class_weight': CLASS_WEIGHT, 'max_iter': MAX_ITER
        },
    }

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=n_experiments)
    # Comparison scheme.
    comparison_scheme = nested_point632plus

    results_lrc = model_comparison(
        comparison_scheme, X, y_lrc, estimators, lrc_hparams, selectors,
        selector_params, random_states, N_REPS, path_to_lrcresults,
        score_func=SCORE
    )
