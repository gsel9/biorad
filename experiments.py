# -*- coding: utf-8 -*-
#
# model_comparison_setup.py
#
# TODO:
# * Include Wilcoxon FS
# * Add n_jobs=-1 to models
# * Pass current model to permutation importance wrapper

"""
Setup model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np
import pandas as pd


def logreg(penalty='l1', seed=None):

    return LogisticRegression(
        penalty=penalty, class_weight='balanced', random_state=seed,
        solver='liblinear'
    )


def forest(seed=None):

    return RandomForestClassifier(
        n_estimators=30, class_weight='balanced', random_state=seed
    )


def target(path_to_target, index_col=0):

    var = pd.read_csv(path_to_target, index_col=index_col)
    return np.squeeze(var.values)


def feature_set(path_to_data, index_col=0):

    data = pd.read_csv(path_to_data, index_col=index_col)
    return np.array(data.values, dtype=float)


if __name__ == '__main__':

    import os
    import feature_selection

    from datetime import datetime
    from model_comparison import model_comparison
    from model_selection import nested_point632plus

    from sklearn.metrics import roc_auc_score

    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # Setup: number of target features, random seed, number of OOB splits.
    K, SEED, N_REPS = 15, 0, 10

    # Shared hyperparameters:
    MAX_ITER = [1500]
    PENALTY = ['l2']
    SOLVER = ['lsqr']
    CLASS_WEIGHT = ['balanced']
    C = [0.001, 0.01, 0.1, 1, 10, 100, 300]
    TOL = [0.001, 0.01, 0.1, 0.3, 0.7, 1]
    N_ESTIMATORS = [20, 50, 100, 500, 1000]
    LEARNINGR_RATE = [0.001, 0.05, 0.2, 0.6, 1]

    # Priors for both target variables summing to 1.0.
    PFS_PRIORS = [0.677, 0.323]
    LRC_PRIORS = [0.753, 0.247]

    # Loss function.
    SCORE = roc_auc_score

    # Repeatability and reproducibility.
    np.random.seed(SEED)

    # Number of repeated experiments.
    n_experiments = 10
    random_states = np.random.randint(1000, size=n_experiments)

    # Set comparison procedure.
    comparison_scheme = nested_point632plus

    # Feature selection algorithms.
    selectors = {
        'logregl1_permut_imp': feature_selection.permutation_importance,
        'logregl2_permut_imp': feature_selection.permutation_importance,
        'rf_permut_imp': feature_selection.permutation_importance,
        'var_thresh': feature_selection.variance_threshold,
        'relieff': feature_selection.relieff,
        'mutual_info': feature_selection.mutual_info
    }
    # Feature selection parameters.
    selector_params = {
        'logregl1_permut_imp': {
            'model': logreg(penalty='l1', seed=SEED), 'thresh': 0.0,
            'nreps': 1
        },
        'logregl2_permut_imp': {
            'model': logreg(penalty='l2', seed=SEED), 'thresh': 0.0,
            'nreps': 1
        },
        'rf_permut_imp': {
            'model': forest(seed=SEED), 'thresh': 0.0, 'nreps': 1
        },
        'mutual_info': {'n_neighbors': 20, 'thresh': 0.05},
        'relieff': {'k': K, 'n_neighbors': 20},
        'var_thresh': {'alpha': 0.05},
    }
    # Classification algorithms.
    estimators = {
        'adaboost': AdaBoostClassifier,
        'lda': LinearDiscriminantAnalysis,
        'logreg': LogisticRegression,
        'gnb': GaussianNB,
        'pls': PLSRegression,
        'qda': QuadraticDiscriminantAnalysis,
    }

    # Feature data.
    X = feature_set('./data/data_to_analysis.csv')

    # PFS target.
    y_pfs = target('./data/target_dfs.csv')
    path_to_pfsresults = './results/results_pfs.csv'

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
    results_pfs = model_comparison(
        comparison_scheme, X, y_pfs, estimators, pfs_hparams, selectors,
        selector_params, random_states, N_REPS, path_to_pfsresults,
        score_func=SCORE
    )

    # LRC target.
    y_lrc = target('./data/target_lrr.csv')
    path_to_lrcresults = './results/results_lrr.csv'

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
    results_lrc = model_comparison(
        comparison_scheme, X, y_lrc, estimators, lrc_hparams, selectors,
        selector_params, random_states, N_REPS, path_to_lrcresults,
        score_func=SCORE
    )
