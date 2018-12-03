# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
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


TMP_RESULTS_DIR = 'tmp_model_comparison'


def _cleanup(results, path_to_results):

    ioutil.write_final_results(path_to_results, results)

    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(TMP_RESULTS_DIR)

    return results


def model_comparison(*args, verbose=1, score_func=None, n_jobs=None, **kwargs):
    """Collecting repeated average performance measures of selected models.

    """
    (
        comparison_scheme, X, y, estimators, estimator_params, selectors,
        fs_params, random_states, n_splits, path_to_results
    ) = args

    global TMP_RESULTS_DIR

    # Setup temporary directory.
    path_tempdir = ioutil.setup_tempdir(TMP_RESULTS_DIR, root='.')

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for estimator_name, estimator in estimators.items():

        print('Running estimator: {}\n{}'.format(estimator.__name__, '-' * 30))

        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(estimator_params[estimator_name])

        for fs_name, fs_func in selectors.items():

            print('Running selector: {}\n{}'.format(fs_name, '-' * 30))

            selector = {
                'name': fs_name, 'func': fs_func, 'params': fs_params[fs_name]
            }
            # Repeating experiments.
            results.extend(
                joblib.Parallel(
                    n_jobs=n_jobs, verbose=verbose
                )(
                    joblib.delayed(comparison_scheme)(
                        X, y, estimator, hparam_grid, selector, n_splits,
                        random_state, path_tempdir, verbose=verbose,
                        score_func=score_func, n_jobs=n_jobs
                    ) for random_state in random_states
                )
            )
    results = _cleanup(results, path_to_results)

    return results
