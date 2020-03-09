# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Schemes for model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
from collections import OrderedDict
from datetime import datetime
from typing import Callable, Tuple
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Local imports.
from utils import ioutil


def nested_cross_validation(X: np.ndarray,
                            y: np.ndarray,
                            experiment_id: str,
                            model: str,
                            hparams: dict,
                            score_func: Callable,
                            df: DataFrame,
                            selector: str,
                            cv: int = 10,
                            output_dir = None,
                            max_evals: int = 100,
                            verbose: int = 1,
                            random_state = None,
                            path_tmp_results: str = None):
    """
    Nested cross-validtion model comparison.

    Args:
        X: Feature matrix (n samples x n features).
        y: Ground truth vector (n samples).
        experiment_id: A unique name for the experiment.
        workflow: A scikit-learn pipeline or model and the associated
            hyperparameter space.
        score_func: Optimisation objective.
        cv (int): The number of cross-validation folds.
        random_states: A list of seed values for pseudo-random number
            generator.
        output_dir: Directory to store SMAC output.
        path_tmp_results: Reference to preliminary experimental results.

    Returns:
        (dict):

    """

    # Name of file with preliminary results.
    if path_tmp_results is None:
        path_case_file = ''
    else:
        path_case_file = os.path.join(
            path_tmp_results, f'experiment_{random_state}_{experiment_id}'
        )

    # Check if prelimnary results aleady exists. If so, load results and
    # proceed to next experiment.
    if os.path.isfile(path_case_file):
        output = ioutil.read_prelim_result(path_case_file)
        print(f'Reloading results from: {path_case_file}')

    # Run a new cross-validation experiment.
    else:
        # Theoutput written to file.
        output = {'random_state': random_state, 'model_name': experiment_id}

        # Time the execution.
        if verbose > 0:
            start_time = datetime.now()
            print(f'Running experiment {random_state} with {experiment_id}')

        # Set random state for the model.
        model.random_state = random_state

        # Record model training and validation performance.
        test_scores, train_scores = [], []

        # Run outer K-folds.
        kfolds = StratifiedKFold(cv, shuffle=True, random_state=random_state)
        for (train_idx, test_idx) in kfolds.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Find optimal hyper-parameters and run inner K-folds.
            optimizer = RandomizedSearchCV(
                estimator=model,
                param_distributions=hparams,
                n_iter=max_evals,
                scoring='roc_auc',
                cv=cv,
                random_state=random_state
            )
            optimizer.fit(X_train, y_train)

            # Include the optimal hyper-parameters in the output.
            output.update(**optimizer.best_params_)
            best_model = optimizer.best_estimator_
            best_model.fit(X_train, y_train)

            # Record training and validation performance of the selected model.
            test_scores.append(
                score_func(y_test, np.squeeze(best_model.predict(X_test)))
            )
            train_scores.append(
                score_func(y_train, np.squeeze(best_model.predict(X_train)))
            )
        # The model performance included in the output.
        output.update(
            OrderedDict(
                [
                    ('test_score', np.mean(test_scores)),
                    ('train_score', np.mean(train_scores)),
                    ('test_score_variance', np.var(test_scores)),
                    ('train_score_variance', np.var(train_scores)),
                ]
            )
        )
        df.at[experiment_id, selector] = np.mean(test_scores)
        if path_tmp_results is not None:
            ioutil.write_prelim_results(path_case_file, output)

            if verbose > 0:
                duration = datetime.now() - start_time
                print(f'Experiment {random_state} completed in {duration}')
                output['exp_duration'] = duration
    return output, df
