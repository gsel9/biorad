# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Framework for performing model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

import numpy as np
import pandas as pd

from utils import ioutil

from sklearn.externals import joblib
from multiprocessing import cpu_count

from joblib import Memory
from shutil import rmtree
from tempfile import mkdtemp


# Name of directory to store temporary results.
TMP_RESULTS_DIR = 'tmp_model_comparison'


def model_comparison(
    comparison_scheme,
    X, y,
    algo,
    pipes_and_params,
    score_func,
    cv,
    max_evals,
    shuffle=True,
    verbose=0,
    random_states=None,
    balancing=True,
    write_prelim=True,
    error_score='nan',
    n_jobs=1,
    path_final_results=None
):
    """
    Compare model performances with optional feature selection.

    Args:
        comparison_scheme (function):
        X (array-like):
        y (array-like):
        algo ():
        cv (int):
        max_evals (int):
        shuffle ():
        verbose ():
        random_states (array-like):
        balancing ():
        write_prelim (bool):
        error_score (float):
        n_jobs (int):
        path_final_results (str):

    """
    global TMP_RESULTS_DIR, ESTIMATOR_ID, SELECTOR_ID

    if write_prelim:
        # Setup temporary directory to store preliminary results.
        path_tmp_results = ioutil.setup_tempdir(TMP_RESULTS_DIR, root='.')

    # Set the default number of available working CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for label, (pipe, params) in pipes_and_params.items():
        # Create a temporary folder to store the state of the pipeline
        # transformers for quick access.
        #_cachedir = mkdtemp()
        #pipe.memory = Memory(cachedir=_cachedir, verbose=10)
        #
        results.extend(
            joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(comparison_scheme)(
                    X=X, y=y,
                    algo=algo,
                    model_id=label,
                    model=pipe,
                    param_space=params,
                    score_func=score_func,
                    cv=cv,
                    max_evals=max_evals,
                    shuffle=shuffle,
                    verbose=verbose,
                    random_state=random_state,
                    balancing=balancing,
                    path_tmp_results=path_tmp_results,
                    error_score=error_score
                )
                for random_state in random_states
            )
        )
        # Delete the temporary pipeline cache.
        #rmtree(_cachedir)
    if write_prelim:
        # Tear down temporary dirs after saving final results to disk.
        _save_and_cleanup(path_final_results, path_tmp_results, results)

    return None


def _save_and_cleanup(path_final_results, path_tmp_results, results):

    # Write final results to disk.
    ioutil.write_final_results(path_final_results, results)
    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(path_tmp_results)

    return None
