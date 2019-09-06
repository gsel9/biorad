# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Work function for model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


from typing import Callable, List, Dict

import numpy as np

from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

from smac.configspace import ConfigurationSpace
from sklearn.pipeline import Pipeline

# Local imports.
from utils import ioutil
from comparison_schemes import nested_cross_validation_smac


def model_comparison_fn(X: np.ndarray,
                        y: np.ndarray,
                        experiments: Dict,
                        score_func: Callable,
                        cv: int,
                        max_evals: int,
                        verbose: int = 0,
                        random_states: List = None,
                        output_dir: str = None,
                        n_jobs: int = None,
                        path_final_results: str = None):
    """
    Compare model performances with optional feature selection.

    Args:
        X: Feature matrix (n samples x n features).
        y: Ground truth vector (n samples).
        experiments: Models where model name is key and a list of tuples 
            (Estimator.__name__, Estimator()).
        score_func: Optimisation objective.
        cv: The number of cross-validation folds.
        max_evals (int): The number of hyperparameter configurations to
            evalaute.
        verbose: Verbosity level.
        random_states: A list of seed values for pseudo-random number
            generator.
        output_dir: Directory to store SMAC output.
        n_jobs: The number of CPUs to use to do the computation.
        path_final_results: Reference to file with experimental results.

    """
    
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1
    
    # Setup temporary directory to store preliminary results.
    path_tmp_results = ioutil.setup_tempdir('tmp_comparison', root='.')

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
                    random_state=random_state,
                    path_tmp_results=path_tmp_results,
                )
                for random_state in random_states
            )
        )
    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(path_tmp_results)
    
    # Write final results to disk.
    ioutil.write_final_results(path_final_results, results)


def config_smac_experiment(procedure, random_state):
    """
    * Join hyperparameter spaces
    * Set random_state attribute to stochastic algorithms.
    * Wrap workflow in a scikit-learn Pipeline.

    Args:
        procedure ():
        random_state (int):

    Returns:
        (tuple): Scikit-learn Pipeline and SMAC ConfigurationSpace
            hyperparameter space.

    """
    
    config_space = ConfigurationSpace()
    config_space.seed(random_state)
    
    for name, algorithm in procedure:
        
        if hasattr(algorithm, 'random_state'):
            algorithm.random_state = random_state
            
        if hasattr(algorithm, 'config_space'):
            config_space.add_configuration_space(
                prefix=name,
                delimiter='__',
                configuration_space=algorithm.config_space
            )
            
    return (Pipeline(procedure), config_space)
