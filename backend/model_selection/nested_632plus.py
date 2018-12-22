# -*- coding: utf-8 -*-
#
# nested_632plus.py
#

"""
The nested .632+ Out-of-Bag procedure for model selection. The .632+ bootstrap
method was proposed by Efron and Tibhirani.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import utils
import ioutil
import model_selection
import feature_selection

import numpy as np
import pandas as pd

from datetime import datetime
from collections import OrderedDict
from sklearn.externals import joblib


def nested_point632plus(
        X, y,
        n_splits,
        random_state,
        path_tmp_results,
        estimator, hparam_grid,
        selector,
        n_jobs=1, verbose=0, score_func=None, score_eval=None
    ):
    """Model performance evaluation according to the .632+ bootstrap method.
    Results are written to disk.

    Args:
        X (array-like):
        y (array-like):
        n_splits (int):
        random_state (int):
        path_tmp_results (str):
        estimator ():
        hparam_grid ():
        selector ():
        eval_method (str, {'mean', 'median'}):
        n_jobs (int):
        verbose (int):
        score_func ():
        score_eval (str):

    """
    # Setup:
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        print('Reloading results from: {}'.format(path_case_file))
    else:
        print('Initiating experiment: {}'.format(random_state))
        start_time = datetime.now()
        results = _nested_point632plus(
            *args, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        duration = datetime.now() - start_time()
        print('Experiment {} completed in {}'.format(random_state, duration))

    return None


def _nested_point632plus(
        X, y,
        n_splits,
        random_state,
        path_tmp_results,
        estimator, hparam_grid,
        selector, support_method,
        n_jobs, verbose, score_func, score_metric
    ):

    # Bookeeping results and feature votes.
    results = {'experiment_id': random_state}
    features = np.zeros(X.shape[1], dtype=int)

    # Outer OOB resampler.
    sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    train_scores, test_scores, opt_hparams = [], [], []
    for num, (train_idx, test_idx) in enumerate(sampler.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Perform exhaustive hyperparameter search.
        best_model, best_support = oob_exhaustive_search(
            X_train, y_train
            n_splits,
            random_state,
            estimator, hparam_grid, selector,
            n_jobs=n_jobs, verbose=verbose,
            score_func=score_func, score_metric=score_metric
        )
        # Error mechanism for logging results.
        if best_model is None and best_support is None:
            train_scores.append(None), test_scores.append(None)
            opt_hparams.append(None)
        else:
            # NB: Need error handling.
            best_model = _check_estimator(
                np.size(best_support), best_model.get_params(), estimator,
                random_state=random_state
            )
            # NB: Need error handling.
            train_score, test_score = utils.scale_fit_predict632(
                best_model, X_train[:, best_support], X_test[:, best_support],
                y_train, y_test, score_func=score_func, score_metric
            )
            train_scores.append(train_score), test_scores.append(test_score)
            opt_hparams.append(best_model.get_params())
            features[best_support] += 1
        # Apply mode to all opt hparam settings.
        best_model_hparams = utils.select_hparams(opt_hparams)
        # Retain features with max activations.
        best_support, num_votes = _filter_support(features, method='max')
    # Callback handling of preliminary results.
    end_results = _update_prelim_results(
        results,
        path_tempdir,
        random_state,
        estimator, best_model_hparams,
        selector, best_support, metric
        score_metric(test_scores), score_metric(train_scores)
    )
    return end_results


def oob_exhaustive_search(
        X_train, y_train
        n_splits,
        random_state,
        estimator, hparam_grid,
        selector,
        n_jobs=n_jobs,
        verbose=verbose,
        score_func=score_func, score_metric=score_metric
    ):
    # Inner OOB resampler.
    sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    # Exhaustive hyperparameter search.
    best_test_score = 0
    best_model, best_support = [], []
    for combo_num, hparams in enumerate(hparam_grid):

        # Bookeeping of feature votes.
        features = np.zeros(X.shape[1], dtype=int)

        train_scores, test_scores = [], []
        for split_num, (train_idx, test_idx) in enumerate(sampler.split(X, y)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # NOTE: Z-score transformation included in selection procedure.

            # Pass model, score func, data, and random state always
            X_train_sub, X_test_sub, support = selector(
                X_train, X_test, y_train, y_test,
                score_func,
                model,
                num_rounds,
                random_state
            )



            # NB: Need error handling.
            model = _check_estimator(
                np.size(support), hparams, estimator, random_state=random_state
            )
            train_score, test_score = utils.scale_fit_predict632(
                model,
                X_train_sub, X_test_sub, y_train, y_test,
                score_func
            )
            features[support] += 1
            train_scores.append(train_score), test_scores.append(test_score)
        # Comparing general model performances.
        if score_eval(test_scores) > best_test_score:
            best_test_score = current_score
            # Retain features of max activation.
            best_support, num_votes = utils.select_support(features)
            # Re-instantiate a new untrained model.
            best_model = _check_estimator(
                np.size(support), hparams, estimator, random_state=random_state
            )
        return best_model, best_support
