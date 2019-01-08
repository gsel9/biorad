# -*- coding: utf-8 -*-
#
# run_experiment.py
#

"""
Experimental setup.

Notes
* Specify the number of original features in the data set in config.

"""

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
    #from sklearn.metrics import matthews_corrcoef
    #from sklearn.metrics import precision_recall_fscore_support

    # FEATURE SET:
    X = load_predictors('./../../data_source/to_analysis/complete_decorr.csv')

    # TARGET:
    #y = load_target('./../../data_source/to_analysis/target_lrr.csv')
    y = load_target('./../../data_source/to_analysis/target_dfs.csv')

    # RESULTS LOCATION:
    path_to_results = './../data/experiments/dfs_init.csv'

    # SETUP:
    CV = 3
    OOB = 5
    MAX_EVALS = 4
    SCORING = roc_auc_score

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.arange(2) #np.random.randint(1000, size=40)

    # Generate pipelines from config elements.
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    # QUESTION: How to easily config and execute experiments?
    # * Maintain config setup for each experiment as a reminder on the setup.
    # *

    comparison.model_comparison(
        model_selection.bbc_cv_selection,
        X, y,
        tpe.suggest,
        pipes_and_params,
        SCORING,
        CV,
        OOB,
        MAX_EVALS,
        shuffle=True,
        verbose=0,
        random_states=random_states,
        alpha=0.05,
        balancing=True,
        write_prelim=True,
        error_score=np.nan,
        n_jobs=-1,
        path_final_results=path_to_results
    )
