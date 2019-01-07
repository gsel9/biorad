# -*- coding: utf-8 -*-
#
# baseline.py
#

"""
Configurations for radiomics baseline experiments.

Goal: Specify classifiers, FS algorithms and experimental setup (run configs,
e.g. random states) and execute.

To Dos:
* Create module (YAML file) with classifiers and FS algorithms that can be can
  be sewn together in sklearn pipelines passed as a dict with unique labels.
* Include a flag in each FS class whether or not should standardize data in
  advance.

Notes:
* Need something that cant raverse a module and collect specified objects.

# Instantiate selector representation for pipeline compatibility.
#selector = feature_selection.Selector(selector_id, procedure)
# Combine parameter grids.
#param_grid = [
#    _format_hparams(selector_params[selector_id]),
#    _format_hparams(estimator_params[estimator_id])
#]
# Create estiamtor pipeline.
#pipeline = Pipeline([
#    (SELECTOR_ID, selector(random_state=random_state)),
#    (ESTIMATOR_ID, estimator(random_state=random_state))
#])


def _format_hparams(params, kind='estimator'):
    # Format parameter names to scikit Pipeline compatibility.

    global ESTIMATOR_ID, SELECTOR_ID

    if kind == 'estimator':
        ext = ESTIMATOR_ID
    elif kind == 'selector':
        ext = SELECTOR_ID
    else:
        raise ValueError('Invalid kind {} not in (`estimator`, `selector`)'
                         ''.format(kind))
    # Update keys.
    for set_key,  param_set in hparams.items():
        hparams[set_key] = {
            key.replace(key, ('__').join((ext, key)): params
            for key, params in param_set.items()
        }
    return hparams

"""


import sys
import time
import backend

import numpy as np
import pandas as pd


# TODO: To utils?
def load_target(path_to_target, index_col=0):

    var = pd.read_csv(path_to_target, index_col=index_col)
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
    # TODO: Move to backend?
    import sys
    sys.path.append('./../model_comparison')

    import os
    import backend
    import model_selection
    import comparison_frame

    from selector_configs import selectors
    from estimator_configs import classifiers

    from hyperopt import tpe

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import precision_recall_fscore_support

    # TEMP:
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)

    # FEATURE SET:
    #X = load_predictors('./../../data_source/to_analysis/complete_decorr.csv')

    # TARGET:
    #y = load_target('./../../data_source/to_analysis/target_lrr.csv')
    #y = load_target('./../../data_source/to_analysis/target_dfs.csv')

    # RESULTS LOCATION:
    #path_to_results = './../../data_source/experiments/no_filtering_dfs.csv'

    # SETUP:
    CV = 10
    OOB = 500
    MAX_EVALS = 100
    SCORING = roc_auc_score

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    #random_states = np.random.randint(1000, size=40)
    random_states = np.arange(2)
    #
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    comparison_frame.model_comparison(
        model_selection.bbc_cv_selection,
        X, y,
        tpe.suggest,
        pipes_and_params,
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
        n_jobs=-1,
        path_final_results='./test_results'
    )
    pipe, params = pipes_and_params['PermutationSelectionRF_PLSRegression']
    #print(params)
