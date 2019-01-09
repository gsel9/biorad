# -*- coding: utf-8 -*-
#
# run_experiment.py
#

"""
Experimental setup.

Notes
* Specify the number of original features in the data set in config.
* Maintain config setup for each experiment as a reminder on the experimental
  details.

Question:
 * How to easily config and execute experiments?

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
    # EXPERIMENT: Timing the execution of the assumed most demanding experiment
    # to estimate an upper bound on experimental time complexity.
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

    from hyperopt import tpe

    from sklearn.metrics import roc_auc_score

    # FEATURE SET:
    X = load_predictors('./../../data_source/to_analysis/complete_decorr.csv')

    # TARGET:
    #y = load_target('./../../data_source/to_analysis/target_lrr.csv')
    y = load_target('./../../data_source/to_analysis/target_dfs.csv')

    # RESULTS LOCATION:
    path_to_results = './../data/experiments/bigo_timing_dfs.csv'
    #path_to_results = './../data/experiments/complete_decorr_dfs.csv'
    #path_to_results = './../data/experiments/complete_decorr_lrr.csv'

    # EXPERIMENTAL SETUP:
    CV = 2#10
    OOB = 3#00
    MAX_EVALS = 3#100
    NUM_EXP_REPS = 1 #30
    SCORING = roc_auc_score

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=NUM_EXP_REPS)

    # Generate pipelines from config elements.
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    # TMP: The assumed most demanding experiment
    name = 'PermutationSelection_SVC'
    _pipes_and_params = {name: pipes_and_params[name]}
    comparison.model_comparison(
        model_selection.bbc_cv_selection,
        X, y,
        tpe.suggest,
        _pipes_and_params,
        SCORING,
        CV,
        OOB,
        MAX_EVALS,
        shuffle=True,
        verbose=1,
        random_states=random_states,
        alpha=0.05,
        balancing=True,
        write_prelim=True,
        error_score=np.nan,
        n_jobs=None,
        path_final_results=path_to_results
    )
