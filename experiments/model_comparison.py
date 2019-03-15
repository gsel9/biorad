# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Perform model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

from smac.configspace import ConfigurationSpace
from sklearn.pipeline import Pipeline

# Local imports.
from comparison_schemes import nested_cross_validation_smac
from utils import ioutil


def model_comparison_fn(
    X,
    y,
    experiments,
    score_func,
    cv: int,
    max_evals: int,
    shuffle: bool=True,
    verbose: int=0,
    random_states=None,
    output_dir: str=None,
    write_prelim: bool=False,
    n_jobs: int=None,
    path_final_results: str=None
):
    """
    Compare model performances with optional feature selection.

    Args:

    """
    # Setup temporary directory to store preliminary results.
    if write_prelim:
        path_tmp_results = ioutil.setup_tempdir('tmp_comparison', root='.')
    else:
        path_tmp_results = None

    # Assign the default number of available working CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1
    # Run experiments.
    results = []
    for experiment_id, setup in experiments.items():
        results.extend(
            Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(nested_cross_validation_smac)(
                    X=X, y=y,
                    experiment_id=experiment_id,
                    workflow=config_smac_experiment(setup, random_state),
                    cv=cv,
                    output_dir=output_dir,
                    score_func=score_func,
                    max_evals=max_evals,
                    verbose=verbose,
                    shuffle=shuffle,
                    random_state=random_state,
                    path_tmp_results=path_tmp_results,
                )
                for random_state in random_states
            )
        )
    # Remove temporary directory if process completed succesfully.
    if write_prelim:
        ioutil.teardown_tempdir(path_tmp_results)
    # Write final results to disk.
    ioutil.write_final_results(path_final_results, results)


def config_smac_experiment(procedure, random_state):
    """Setup experimental configurations. Joins hyperparameter spaces, assigns
    random state to stochastic algorithms and formats workflow as a
    scikit-learn Pipeline object.

    Returns:
        (tuple): Scikit-learn Pipeline and SMAC ConfigurationSpace
            hyperparameter space.

    """
    config_space = ConfigurationSpace()
    config_space.seed(random_state)
    for name, algorithm in procedure:
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

    return (Pipeline(procedure), config_space)
