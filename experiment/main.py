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

    from hyperopt import tpe

    from sklearn.metrics import roc_auc_score

    # FEATURE SET:
    X = load_predictors('./../../data_source/to_analysis/complete_decorr.csv')

    # TARGET:
    #y = load_target('./../../data_source/to_analysis/target_dfs.csv')
    y = load_target('./../../data_source/to_analysis/target_lrr.csv')

    # RESULTS LOCATION:
    path_to_results = './../data/experiments/complete_decorr_dfs.csv'
    #path_to_results = './../data/experiments/complete_decorr_lrr.csv'

    # EXPERIMENTAL SETUP:
    CV = 10
    OOB = 300
    MAX_EVALS = 25
    NUM_EXP_REPS = 30
    SCORING = roc_auc_score

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=NUM_EXP_REPS)

    # Generate pipelines from config elements.
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    comparison.model_comparison(
        model_selection.bbc_cv_selection,
        X, y,
        tpe.suggest,
        pipes_and_params,
        SCORING,
        CV,
        OOB,
        # TODO: Potentially increase max evals if feature screening sign. reduces
        # size feature space which in turn speeds up computations. Validate the
        # number of max evals by inspecting the loss collected from eval of the
        # objective function.
        MAX_EVALS,
        shuffle=True,
        verbose=1,
        random_states=random_states,
        alpha=0.05,
        balancing=True,
        write_prelim=True,
        # NB: To ensure same number of features are selected in each fold for
        # making y_pred comparable across each fold.
        error_score='random_subset',
        n_jobs=None,
        path_final_results=path_to_results
    )
