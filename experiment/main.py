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
    # TODO: run for multiple configurations and select the features with ICC greater than thresh.


    # EXPERIMENT:
    # Baseline experiment including all features. Results compared to sessions
    # including dim reduction a priori.
    #
    import sys
    sys.path.append('./../model_comparison')

    import os
    import backend
    import configs
    import comparison
    import model_selection

    from configs.selector_configs import selectors
    from configs.estimator_configs import classifiers

    import hyperopt
    from functools import partial

    from dgufs.dgufs import DGUFS

    from sklearn.metrics import roc_auc_score

    # FEATURE SET:
    # * NB: Do not scale prior to model comparison due to min variance
    #   filtering. If Z-score transform, var(X) = 1. Use scaler in pipeline
    #   instead.
    X = load_predictors('./../../data_source/to_analysis/no_filter_concat.csv')
    data = pd.read_csv('./../../data_source/to_analysis/no_filter_concat.csv', index_col=0)

    # TARGET:
    #y = load_target('./../../data_source/to_analysis/target_dfs.csv')
    y = load_target('./../../data_source/to_analysis/target_lrr.csv')

    # RESULTS LOCATION:
    path_to_results = '.test.csv'
    #path_to_results = './../data/experiments/no_filter_concat_dfs.csv'
    #path_to_results = './../data/experiments/complete_decorr_lrr.csv'

    # EXPERIMENTAL SETUP:
    CV = 4
    MAX_EVALS = 200
    NUM_EXP_REPS = 30
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
        n_EI_candidates=500,
        # Use 20 % of best observations to estimate next set of parameters.
        gamma=0.20,
        # First 20 trials are going to be random (include probability theory
        # for 90 % CI with this setup).
        n_startup_jobs=25,
    )
    comparison.model_comparison(
        model_selection.nested_kfold_selection,
        X, y,
        tpe,
        {'PermutationSelection_PLSRegression': pipes_and_params['PermutationSelection_PLSRegression']},
        SCORING,
        CV,
        MAX_EVALS,
        shuffle=True,
        verbose=1,
        random_states=random_states,
        balancing=False,
        write_prelim=True,
        error_score='all',
        n_jobs=None,
        path_final_results=path_to_results
    )
