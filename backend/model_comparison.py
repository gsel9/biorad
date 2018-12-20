# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Framework for performing model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import utils
import ioutil
import shutil
import logging
import model_selection
import feature_selection

import numpy as np
import pandas as pd

from datetime import datetime
from multiprocessing import cpu_count
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

# Name of directory to store temporary results.
TMP_RESULTS_DIR = 'tmp_model_comparison'


def _cleanup(results, path_to_results):

    ioutil.write_final_results(path_to_results, results)

    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(TMP_RESULTS_DIR)

    return results


def model_comparison(
        comparison_scheme,
        X, y,
        n_splits,
        random_states,
        path_to_results,
        estimators, estimator_params,
        selectors=None, selector_params=None,
        verbose=1, score_func=None, score_metric=None, n_jobs=None
    ):
    """Compare model performances with optional feature selection.

    Args:
        comparison_scheme ():
        X (array-like):
        y (array-like):
        n_splits (int):
        random_states (array-like):
        estimators (dict):
        estimator_params (dict):
        selectors (dict):
        selector_params (dict):
        verbose (int):
        score_func ():
        n_jobs (int):

    Returns:
        None: Writes results to disk an removes tmp directory.

    """

    global TMP_RESULTS_DIR

    # Setup temporary directory.
    path_tmp_results = ioutil.setup_tempdir(TMP_RESULTS_DIR, root='.')

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for estimator_name, estimator in estimators.items():
        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(estimator_params[estimator_name])
        # Skip feature selection.
        if selectors is None:
            results.extend(
                joblib.Parallel(
                    n_jobs=n_jobs, verbose=verbose
                )(
                    joblib.delayed(comparison_scheme)(
                        X, y,
                        n_splits,
                        random_state,
                        path_tmp_results,
                        estimator, hparam_grid,
                        selector=None,
                        n_jobs=n_jobs, verbose=verbose,
                        score_func=score_func, score_metric=score_metric
                    )
                    for random_state in random_states
                )
            )
        # Including feature selection.
        else:
            for selector_name, procedure in selectors.items():
                selector = {
                    'name': selector_name,
                    'func': fprocedure,
                    'params': selector_params[selector_name]
                }
                results.extend(
                    joblib.Parallel(
                        n_jobs=n_jobs, verbose=verbose
                    )(
                        joblib.delayed(comparison_scheme)(
                            X, y,
                            n_splits,
                            random_state,
                            path_tmp_results,
                            estimator, hparam_grid,
                            selector=selector,
                            n_jobs=n_jobs, verbose=verbose,
                            score_func=score_func, score_metric=score_metric
                        )
                        for random_state in random_states
                    )
                )
        # Tear down temporary dirs after succesfully written results to disk.
        _cleanup(results, path_to_results)

    return None


if __name__ == '__main__':

    # TODO:
    # * Include Wilcoxon FS
    # * Add n_jobs = -1 to models
    # * Pass current estimator to permutation importance
