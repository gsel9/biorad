# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Work function for model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


from multiprocessing import cpu_count
from typing import Callable, Dict, List

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pandas import DataFrame
from comparison_schemes import nested_cross_validation
from utils import ioutil

def model_comparison_experiment(X: np.ndarray,
                                y: np.ndarray,
                                models: Dict,
                                hparams: Dict,
                                score_func: Callable,
                                cv: int,
                                max_evals: int,
                                df: DataFrame,
                                selector: str,
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

        # Get hyper-parameters for this model.
        model_hparams = hparams[model_name]
        for random_state in random_states:
            result, df = (
                (nested_cross_validation)(
                        X=X, y=y,
                        model=model,
                        experiment_id=model_name,
                        hparams=model_hparams,
                        cv=cv,
                        score_func=score_func,
                        max_evals=max_evals,
                        random_state=random_state,
                        path_tmp_results=path_tmp_results,
                        df=df,
                        selector=selector
                    )
                )
            results.append(result)
    #  todo extend the list of results for multiple random states
    print(df)
    # Remove temporary directory.
    ioutil.teardown_tempdir(path_tmp_results)
    print(results)
    # Write final results to disk.
    ioutil.write_final_results(path_final_results, results)
    return df
