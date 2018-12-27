# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Framework for performing model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import feature_selection

import numpy as np
import pandas as pd

from utils import ioutil

from multiprocessing import cpu_count
from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

# Name of directory to store temporary results.
TMP_RESULTS_DIR = 'tmp_model_comparison'
# Extensions to estimator and selector hyperparameters. 
ESTIMATOR_ID = 'estimator'
SELECTOR_ID = 'dim_red'


def _cleanup(results, path_to_results):

    ioutil.write_final_results(path_to_results, results)

    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(TMP_RESULTS_DIR)

    return results


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


def model_comparison(
        comparison_scheme,
        X, y,
        n_splits,
        random_states,
        estimators,
        estimator_params,
        selectors=None,
        selector_params=None,
        score_func=None,
        score_eval=None,
        verbose=0,
        n_jobs=None,
        path_to_results=None
    ):
    """Compare model performances with optional feature selection.

    Args:
        comparison_scheme (function):
        X (array-like):
        y (array-like):
        n_splits (int):
        random_states (array-like):
        estimators (dict):
        estimator_params (dict):
        selectors (dict):
        selector_params (dict):
        score_func ():
        score_eval ():
        verbose (int):
        n_jobs (int):
        path_to_results (str):

    Returns:
        None: Writes results to disk an removes tmp directory.

    """
    global TMP_RESULTS_DIR, ESTIMATOR_ID, SELECTOR_ID

    # Setup temporary directory.
    path_tmp_results = ioutil.setup_tempdir(TMP_RESULTS_DIR, root='.')

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for estimator_name, estimator in estimators.items():
        # Setup hyperparameter grid.
        #hparam_grid = ParameterGrid(estimator_params[estimator_name])
        for selector_name, procedure in selectors.items():
            # TODO: Enable scikit API (fit, transform)
            selector = feature_selection.Selector(
                selector_name, procedure, selector_params[selector_name]
            )
            # TODO: Do a check estimator for random states etc.
            pipe = Pipeline([
                (SELECTOR_ID, selector(random_state=random_state)),
                (ESTIMATOR_ID, estimator(random_state=random_state))
            ])
            results.extend(
                joblib.Parallel(
                    n_jobs=n_jobs, verbose=verbose
                )(
                    joblib.delayed(comparison_scheme)(
                        X, y,
                        pipe,

                        ##

                        n_splits,
                        random_state,
                        path_tmp_results,
                        estimator,
                        hparam_grid,
                        selector=selector,
                        n_jobs=n_jobs, verbose=verbose,
                        score_func=score_func, score_eval=score_eval
                    )
                    for random_state in random_states
                )
            )
        # Tear down temporary dirs after succesfully written results to disk.
        _cleanup(results, path_to_results)

    return None
