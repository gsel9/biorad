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
# TODO: Add option to filter features based on reexes (e.g. only clinical and PET)
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

    from hyperopt import tpe

    from model_selection import bbc_cv_selection
    from model_comparison import model_comparison

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import precision_recall_fscore_support

    # TEMP:
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import Pipeline
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
    random_states = np.random.randint(1000, size=40)


    from selector_configs import selectors
    from estimator_configs import classifiers


    # TODO:
    # * Combine all pipes and corresponding params in a dict so that in
    #   model comparison loop a pipeline with (key, model) is unpacked together
    #   with a hparam space of key__param_name.
    # * Use pipe label + random state in prelim file name.

    from collections import OrderedDict

    pipes_and_params = OrderedDict()
    for classifier_name, clf_setup in classifiers.items():
        for selector_name, sel_setup in selectors.items():
            pipe_label = '{}_{}'.format(selector_name, classifier_name)
            # Joining two lists of selector and estimator pipe elements.
            pipe_elem = [*sel_setup['selector'], *clf_setup['estimator']]
             # Joining two dicts of selector and estimator parameters.
            pipe_param_space = {**sel_setup['params'], **clf_setup['params']}
            # Format for model comparison experiments.
            pipes_and_params['pipe_label'] = {
                'pipe': pipe_elem, 'params': pipe_param_space
            }

    print(pipes_and_params)



    """
    # Parameter search space
    space = {}
    # Random number between 50 and 100
    space['kbest__percentile'] = hp.uniform('kbest__percentile', 50, 100)
    # Random number between 0 and 1
    #space['clf__l1_ratio'] = hp.uniform('clf__l1_ratio', 0.0, 1.0)
    # Log-uniform between 1e-9 and 1e-4
    #space['clf__alpha'] = hp.loguniform('clf__alpha', -9*np.log(10), -4*np.log(10))
    # Random integer in 20:5:80
    #space['clf__n_iter'] = 20 + 5 * hp.randint('clf__n_iter', 12)
    # Random number between 50 and 100
    space['clf__class_weight'] = hp.choice('clf__class_weight', [None,]) #'balanced']),
    space['clf__n_estimators'] = scope.int(hp.quniform('clf__clf__n_estimators', 20, 500, 5))
    # Discrete uniform distribution
    space['clf__max_leaf_nodes'] = scope.int(hp.quniform('clf__max_leaf_nodes', 30, 150, 1))
    # Discrete uniform distribution
    space['clf__min_samples_leaf'] = scope.int(hp.quniform('clf__min_samples_leaf', 20, 500, 5))
    """



    """
    model_comparison(
        model_selection.bbc_cv_selection,
        X_train, y_train,
        tpe.suggest,
        pipes,
        space,
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
        path_to_results='test_results'
    )
    """
