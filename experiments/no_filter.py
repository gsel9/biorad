# -*- coding: utf-8 -*-

import sys
import time

import numpy as np
import pandas as pd


# NOTE: To utils
def target(path_to_target, index_col=0):

    var = pd.read_csv(path_to_target, index_col=index_col)
    return np.squeeze(var.values).astype(np.float32)


# NOTE: To utils
def feature_set(path_to_data, index_col=0):

    data = pd.read_csv(path_to_data, index_col=index_col)
    return np.array(data.values, dtype=np.float32)


if __name__ == '__main__':
    import sys
    sys.path.append('./../model_comparison')

    import os
    import feature_selection

    from scipy import stats
    from datetime import datetime
    from model_comparison import model_comparison
    from model_selection import nested_point632plus

    from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support

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
    CV = 10
    N_ITER = 100

    NUM_REPS = 100
    NUM_SPLITS = 10
    NUM_RESAMPLINGS = 500
    EVAL = np.median
    SCORING = roc_auc_score

    estimators = {
        'logreg': LogisticRegression,
        'rf': RandomForestClassifier,
        'plsr': PLSRegression,
        'gnb': GaussianNB,
        'svc': SVC,
    }
    selectors = {
        #'permutation': feature_selection.permutation_selection,
        'wlcx': feature_selection.wilcoxon_selection,
        'relieff_5': feature_selection.relieff_selection,
        'relieff_20': feature_selection.relieff_selection,
        'mrmr': feature_selection.mrmr_selection
    }

    # HYPERPARAMETER DISTRIBUTIONS
    # - We want about an equal chance of ending up with a number of any order
    # of magnitude within our range of interest.
    n_estimators = stats.expon(scale=100)
    max_depth = stats.randint(1, 40)

    estimator_hparams = {
        # Classification hyperparameters.
        'rf': {
            # From scikit-learn example (typically, many estimators = good).
            'n_estimators': stats.expon(scale=100),
            # (typically, deep = good)
            'max_depth': stats.randint(1, 40),
            # Has somethin gto do with feature selection? Should thus be skipped.
            'max_features': stats.randint(1, 11),
            'min_samples_split': stats.randint(2, 11),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced'],
        },
        'svc': {
            # class_weight: balanced by default.
            # Limiting the number of unique values (size=100) to ensure a
            # certain degree of diversity in the hparam values.
            # Small tolerance typically good, but want to checkout a few
            # alternatives.
            'tol': stats.reciprocal(size=10),
            # From scikit-learn docs.
            'C': stats.expon(scale=100),
            # From scikit-learn docs.
            'gamma': stats.expon(scale=.1),
            'kernel': ['linear', 'rbf', 'ploy'],
            'degree': [2, 3],
            'max_iter': [2000]
        },
        'logreg': {
            'tol': stats.reciprocal(size=10),
            'C': stats.expon(scale=100),
            'class_weight': ['balanced'],
            'penalty': ['l1', 'l2'],
            'max_iter': [2000],
            'solver': ['liblinear'],
        },
        'gnb': {
            # DFS:
            'priors': [[0.677, 0.323]]
            # LRR:
            # 'priors'. [[0.75, ]]
        },
        'plsr': {
            'tol': stats.reciprocal(size=10),
            'n_components': stats.expon(scale=100),
            'max_iter': [2000]
        },
    selector_hparams = {
        # Feature selection hyperparameters.
        'permutation': {
            'num_rounds': [100]
        },
        'wlcx': {
            'thresh': [0.05]
        },
        'relieff': {
            'num_neighbors': stats.expon(scale=100),
            'num_features': stats.expon(scale=100)
        },
        'mrmr': {
            'num_features': ['auto'],
            # See paper
            'k': [3, 5, 7]
        }
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
        verbose=1,
        n_jobs=None,
        path_to_results=path_to_results
    )
