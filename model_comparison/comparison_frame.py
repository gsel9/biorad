# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Framework for performing model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import utils

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from multiprocessing import cpu_count


# Name of directory to store temporary results.
TMP_RESULTS_DIR = 'tmp_model_comparison'


def model_comparison(
    comparison_scheme,
    X, y,
    algo,
    pipes_and_params,
    score_func,
    cv,
    oob,
    max_evals,
    shuffle=True,
    verbose=0,
    random_states=None,
    alpha=0.05,
    balancing=True,
    write_prelim=True,
    error_score=np.nan,
    n_jobs=1,
    path_final_results=None
):
    """
    Compare model performances with optional feature selection.

    Args:
        comparison_scheme (function):
        X (array-like):
        y (array-like):
        cv (int):
        obb (int):
        max_evals (int):
        random_states (array-like):
        estimators (dict):
        verbose (int):
        n_jobs (int):
        path_to_results (str):

    """
    global TMP_RESULTS_DIR, ESTIMATOR_ID, SELECTOR_ID

    # Setup temporary directory to store preliminary results.
    path_tmp_results = utils.ioutil.setup_tempdir(TMP_RESULTS_DIR, root='.')

    # Set the default number of available working CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for label, (pipe, params) in pipes_and_params.items():
        results.extend(
            joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(comparison_scheme)(
                    X, y,
                    algo,
                    label,
                    pipe,
                    params,
                    score_func,
                    path_tmp_results,
                    cv, oob, max_evals,
                    shuffle,
                    verbose,
                    random_state,
                    alpha,
                    balancing,
                    write_prelim,
                    error_score
                )
                for random_state in random_states
            )
        )
    # Tear down temporary dirs after saving final results to disk.
    _save_and_cleanup(path_final_results, path_tmp_results, results)

    return None


def _save_and_cleanup(path_final_results, path_tmp_results, results):

    # Write final results to disk.
    utils.ioutil.write_final_results(path_final_results, results)
    # Remove temporary directory if process completed succesfully.
    utils.ioutil.teardown_tempdir(path_tmp_results)

    return None


if __name__ == '__main__':
    # Demo run:
    # * 97.80 % accuracy seems to be a fairly good score.
    #
    # TODO: Need seed + clf + selector name for unique ID to prelim results files.
    import model_selection

    from hyperopt import hp
    from hyperopt import tpe
    from hyperopt.pyll.base import scope

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # TEMP:
    from sklearn.feature_selection.univariate_selection import SelectPercentile, chi2
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    pipe = Pipeline([
        ('kbest', SelectPercentile(chi2)),
        ('clf_scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=0))
    ])

    pipes = {
        'test': pipe
    }

    # Can specify hparam distr in config files that acn direclty be read into
    # Python dict with hyperopt distirbutions?

    # Parameter search space
    space = {}
    # Random number between 50 and 100
    space['kbest__percentile'] = hp.uniform('kbest__percentile', 50, 100)
    # Random number between 0 and 1
    #space['clf__l1_ratio'] = hp.uniform('clf__l1_ratio', 0.0, 1.0)
    # Log-uniform between 1e-9 and 1e-4
    #space['clf__alpha'] = hp.loguniform('clf__alpha', -9*np.log(10), -4*np.log(10))
    # Random integer in 20:5:80
    #space['clf__n_iter'] = 20 + 5 * hp.randint('clf__n_iter', 12)
    # Random number between 50 and 100
    space['clf__class_weight'] = hp.choice('clf__class_weight', [None,]) #'balanced']),
    space['clf__n_estimators'] = scope.int(hp.quniform('clf__clf__n_estimators', 20, 500, 5))
    # Discrete uniform distribution
    space['clf__max_leaf_nodes'] = scope.int(hp.quniform('clf__max_leaf_nodes', 30, 150, 1))
    # Discrete uniform distribution
    space['clf__min_samples_leaf'] = scope.int(hp.quniform('clf__min_samples_leaf', 20, 500, 5))

    model_comparison(
        model_selection.bbc_cv_selection,
        X_train, y_train,
        tpe.suggest,
        pipes,
        space,
        roc_auc_score,
        cv=5,
        oob=5,
        max_evals=5,
        shuffle=True,
        verbose=1,
        random_states=np.arange(2),
        alpha=0.05,
        balancing=True,
        write_prelim=True,
        error_score=np.nan,
        n_jobs=-1,
        path_to_results='test_results'
    )
