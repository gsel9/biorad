# -*- coding: utf-8 -*-
#
# nested_632plus.py
#
# In experimental setup:
# * Specify the number of components (include a PCA on the feature set to infere
#   rasonable range.)
# *


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
            estimator, hparam_grid,
            selector,
            n_jobs=n_jobs,
            verbose=verbose,
            score_func=score_func,
            score_metric=score_metric
        )
        # NOTE: Z-score transformation and error handlng included in
        # function.
        train_score, test_score = utils.scale_fit_predict632(
            X_train[:, best_support], X_test[:, best_support],
            y_train, y_test,
            score_func,
            best_model
        )
        # NOTE: Error handling mechanism.
        if train_score is None and test_score is None:
            pass
        else:
            # TODO:

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
            # Exectue modeling procedure for performance evaluation.
            train_score, test_score = _eval_candidate_procedure(
                X[train_idx], X[test_idx],
                y[train_idx], y[test_idx],
                estimator, hparams,
                selector,
                score_func,
                random_state,
            )
            # NOTE: Error handling mechanism.
            if train_score is None and test_score is None:
                pass
            else:
                features[support] += 1
                train_scores.append(train_score)
                test_scores.append(test_score)
        if score_eval(test_scores) > best_test_score:
            best_test_score = current_score
            # Retain features of max activation.
            best_support, num_votes = utils.select_support(features)
            # Re-instantiate a new untrained model.
            best_model = check_estimator(
                len(support),
                hparams,
                estimator,
                random_state=random_state
            )
        return best_model, best_support


def _eval_candidate_procedure(*args):
    # Evaluate the performance of modeling procedure.
    (
        X_train, X_test, y_train, y_test,
        hparams, estimator,
        selector,
        score_func,
        random_state,

    ) = args
    # Reconstruct a model prior to feature selection.
    model = check_estimator(
        hparams,
        estimator,
        support=None,
        random_state=random_state
    )
    # NOTE: Z-score transformation and error handlng included in
    # selector.
    X_train_sub, X_test_sub, support = selector(
        X_train, X_test, y_train, y_test,
        random_state=random_state,
        score_func=score_func,
        model=model
    )
    # Reconstruct a model prior to predictions.
    model = check_estimator(
        hparams,
        estimator,
        support=support,
        random_state=random_state
    )
    # NOTE: Z-score transformation and error handlng included in
    # function.
    train_score, test_score = utils.scale_fit_predict632(
        X_train_sub, X_test_sub, y_train, y_test,
        score_func,
        model
    )
    return train_score, test_score
