# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Framework for performing model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

from multiprocessing import cpu_count
from smac.configspace import ConfigurationSpace
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from utils import ioutil


def model_comparison(
    comparison_scheme,
    X, y,
    experiments,
    score_func,
    cv: int,
    max_evals: int,
    shuffle: bool=True,
    verbose: int=0,
    random_states=None,
    output_dir: str=None,
    write_prelim: bool=False,
    n_jobs: int=1,
    path_final_results: str=None
):
    """
    Compare model performances with optional feature selection.

    Args:

    """
    # Setup temporary directory to store preliminary results.
    if write_prelim:
        path_tmp_results = ioutil.setup_tempdir(
            'tmp_model_comparison', root='.'
        )
    else:
        path_tmp_results = None
    # Set the default number of available working CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1

    results = []
    for experiment_id, setup in experiments.items():
        results.extend(
            joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(comparison_scheme)(
                    X=X, y=y,
                    experiment_id=experiment_id,
                    workflow=config_experiment(setup, random_state),
                    cv=cv,
                    output_dir=output_dir,
                    score_func=score_func,
                    max_evals=max_evals,
                    verbose=verbose,
                    random_state=random_state,
                    path_tmp_results=path_tmp_results,
                )
                for random_state in random_states
            )
        )
    # Remove temporary dir if succesfully writing final results to disk.
    if write_prelim:
        _cleanup_prelim(path_tmp_results)

    _write_results(path_final_results, results)

    return None


def config_experiment(setup, random_state):
    """Setup experimental configurations:
    - Joins hyperparameter spaces.
    - Assigns random state to stichastic algorithms.
    - Formats workflow as a scikit-learn Pipeline object.

    Returns:
        (tuple): Pipeline and hyperparameter space.

    """
    config_space = ConfigurationSpace()
    config_space.seed(random_state)
    for name, algorithm in setup:
        # Join hyperparameter spaces.
        if hasattr(algorithm, 'config_space'):
            config_space.add_configuration_space(
                prefix=name,
                configuration_space=algorithm.config_space,
                delimiter='__'
            )
        # Set random generator seed for stochastic algorithms.
        if hasattr(algorithm, 'random_state'):
            algorithm.random_state = random_state

    return (Pipeline(setup), config_space)


def _write_results(path_final_results, results):
    # Write final results to disk.
    ioutil.write_final_results(path_final_results, results)

    return None


def _cleanup_prelim(path_tmp_results):
    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(path_tmp_results)

    return None
