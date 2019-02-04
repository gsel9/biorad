# -*- coding: utf-8 -*-
#
# main.py
#

"""
Execute radiomic model comparison experiments.

NOTE:
* Required to specify the number of original features included in the data set
  in the config file.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import sys
sys.path.append('./../')

import time
import backend

import numpy as np
import pandas as pd


# TODO: To utils?
def load_target(path_to_target, index_col=0, classify=True):

    var = pd.read_csv(path_to_target, index_col=index_col)
    if classify:
        return np.squeeze(var.values).astype(np.int32)
    else:
        return np.squeeze(var.values).astype(np.float32)


# TODO: To utils?
def load_predictors(path_to_data, index_col=0, regex=None):

    data = pd.read_csv(path_to_data, index_col=index_col)
    if regex is None:
        return np.array(data.values, dtype=np.float32)
    else:
        target_features = data.filter(regex=regex)
        return np.array(data.loc[:, target_features].values, dtype=np.float32)


if __name__ == '__main__':
    """To Dos:

    * Design hparam spaces around Alises configs.
    * Run experiment for comparison with Alise's results.

    """
    import sys
    sys.path.append('./../../model_comparison')

    import os
    import backend
    import comparison
    import model_selection

    from selector_configs import selectors
    from estimator_configs import classifiers

    import hyperopt
    from functools import partial

    from sklearn.metrics import roc_auc_score

    # TEMP:
    from sklearn.preprocessing import StandardScaler
    from dgufs.dgufs import DGUFS
    from scipy import linalg
    from mlxtend.preprocessing import minmax_scaling
    from sklearn.decomposition import PCA

    # FEATURE SET:
    y_name = 'Toklasser'
    xls = pd.ExcelFile('X_endelig_squareroot.xlsx')
    data_raw_df = pd.read_excel(xls, sheet_name='tilbakefall', index_col=0)
    n_features = 2
    y = data_raw_df[y_name].values
    X = data_raw_df.drop(y_name, 1).values

    #load_predictors('./../../data_source/to_analysis/alise_setup.csv')
    #X = load_predictors('./../../data_source/to_analysis/sqroot_concat.csv')

    # TARGET:
    #y = load_target('./../../data_source/to_analysis/target_dfs.csv')
    #y = load_target('./../../data_source/to_analysis/target_lrr.csv')

    # RESULTS LOCATION:
    # Categorical/continous
    #path_to_results = './chi2_mi.csv' (slow)
    #path_to_results = './chi2_anova.csv' #(fast)
    #path_to_results = './mi_mi.csv' (slow)
    #path_to_results = './mi_anovatest.csv' #(fast) + winner!
    #path_to_results = './mi_mrmr.csv' #(fast) + winner!
    #path_to_results = './zca_corr_mi_anovatest.csv.csv'
    #path_to_results = './pca_mi_anovatest.csv.csv'
    #path_to_results = './mi_dgufs.csv' (slow)
    path_to_results = './test.csv'

    # mRMR feature screening is very slow, but indicates the best results.

    #path_to_results = './../data/experiments/no_filter_concat_dfs.csv'
    #path_to_results = './../data/experiments/complete_decorr_lrr.csv'

    # EXPERIMENTAL SETUP:
    CV = 4
    MAX_EVALS = 2
    NUM_EXP_REPS = 10
    SCORING = roc_auc_score

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=NUM_EXP_REPS)

    # Generate pipelines from config elements.
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    # Parameters to tune the TPE algorithm.
    tpe = partial(
        hyperopt.tpe.suggest,
        # Sample 1000 candidates and select the candidate with highest
        # Expected Improvement (EI).
        n_EI_candidates=5,
        # Use 20 % of best observations to estimate next set of parameters.
        gamma=0.2,
        # First 20 trials are going to be random (include probability theory
        # for 90 % CI with this setup).
        n_startup_jobs=2,
    )
    comparison.model_comparison(
        model_selection.nested_kfold_selection,
        X, y,
        tpe,
        {'ReliefFSelection_PLSRegression': pipes_and_params['ReliefFSelection_PLSRegression']},
        SCORING,
        CV,
        MAX_EVALS,
        shuffle=True,
        verbose=0,
        random_states=random_states,
        balancing=False,
        write_prelim=True,
        error_score='all',
        n_jobs=1,
        path_final_results=path_to_results
    )
    res = pd.read_csv(path_to_results, index_col=0)
    print(np.mean(res['test_score']))
