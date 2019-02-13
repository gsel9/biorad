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
from collections import OrderedDict

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from joblib import Memory
from shutil import rmtree
from tempfile import mkdtemp
from multiprocessing import cpu_count

from smac.configspace import ConfigurationSpace


# Name of directory to store temporary results.
TMP_RESULTS_DIR = 'tmp_model_comparison'


def model_comparison(
    comparison_scheme,
    X, y,
    experiments,
    score_func,
    cv,
    max_evals,
    shuffle=True,
    verbose=0,
    random_states=None,
    output_dir=None,
    write_prelim=False,
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

    # Setup temporary directory to store preliminary results.
    if write_prelim:
        path_tmp_results = ioutil.setup_tempdir(TMP_RESULTS_DIR, root='.')
    else:
        path_tmp_results = None

    # Set the default number of available working CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for experiment_id, setup in experiments.items():
        workflow = config_experiment(experiment_id, setup, 0)
        print(workflow.items())
    """
        results.extend(
            joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(comparison_scheme)(
                    X=X, y=y,
                    experiment_id=experiment_id,
                    hparam_space=hparam_space,
                    workflow=workflow,
                    cv=cv,
                    output_dir=output_dir,
                    score_func=score_func,
                    max_evals=max_evals,
                    verbose=verbose,
                    random_state=random_state,
                    path_tmp_results=path_tmp_results,
                    error_score=error_score
                )
                for random_state in random_states
            )
        )

    # Tear down temporary dirs after saving final results to disk.
    if write_prelim:
        _cleanup_prelim(path_tmp_results)

    _write_results(path_final_results, results)

    return None
    """


def config_experiment(experiment_id, setup, random_state):
    """

    """

    config_space = ConfigurationSpace()
    config_space.seed(random_state)

    workflow = OrderedDict()
    for name, algorithm in setup:
        # Join hyperparameter spaces.
        if hasattr(algorithm, 'config_space'):
            config_space.add_configuration_space(
                prefix=name,
                configuration_space=algorithm.config_space,
                delimiter='__'
            )
        # Set seed for random generator of stochastic algorithms.
        if hasattr(algorithm, 'random_state'):
            algorithm.random_state = random_state

    workflow[experiment_id] = (Pipeline(setup), config_space)

    return workflow


def _write_results(path_final_results, results):

    # Write final results to disk.
    ioutil.write_final_results(path_final_results, results)

    return None


def _cleanup_prelim(path_tmp_results):

    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(path_tmp_results)

    return None
