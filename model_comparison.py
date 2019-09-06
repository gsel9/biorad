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

from tqdm import tqdm
import numpy as np

from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

# Local imports.
from utils import ioutil
from comparison_schemes import nested_cross_validation


def model_comparison_experiment(X: np.ndarray,
                                y: np.ndarray,
                                models: Dict,
                                hparams: Dict,
                                score_func: Callable,
                                cv: int,
                                max_evals: int,
                                random_states: List = None,
                                n_jobs: int = None,
                                path_final_results: str = None):
    """
    Compare model performances with optional feature selection.

    Args:
        X: Feature matrix (n samples x n features).
        y: Ground truth vector (n samples).
        models: Key-value pairs with model name and a scikit-learn Pipeline.
        hparams: Optimisation objective.
    """

    # Set the number of workers to use for parallelisation.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1

    # Setup temporary directory to store preliminary results.
    path_tmp_results = ioutil.setup_tempdir('tmp_comparison', root='.')

    # Iterate through models and run experiments in parallel.
    results = []
    for model_name, model in tqdm(models.items()):

        # Get hyperparameters for this model.
        model_hparams = hparams[model_name]

        results.extend(
            Parallel(n_jobs=n_jobs)(
                delayed(nested_cross_validation)(
                    X=X, y=y,
                    model=model,
                    experiment_id=model_name,
                    hparams=model_hparams,
                    cv=cv,
                    score_func=score_func,
                    max_evals=max_evals,
                    random_state=random_state,
                    path_tmp_results=path_tmp_results,
                )
                for random_state in random_states
            )
        )

    # Remove temporary directory.
    ioutil.teardown_tempdir(path_tmp_results)

    # Write final results to disk.
    ioutil.write_final_results(path_final_results, results)
