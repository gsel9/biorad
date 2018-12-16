# -*- coding: utf-8 -*-
#
# model_selection.py
#
# TODO:
# * Optional feature selection
# * Select from median (not mean)
# * Default mechanism to return None if error occurs.
# * Separate directoris with model copmarison schemes. One module per scheme.

"""
Frameworks for performing model selection.
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
        selector=None,
        n_jobs=1, verbose=0, score_func=None, comparison='mean'
    ):
    """

    """
    if comparison == 'mean':
        gen_score = np.mean
    elif comparison == 'median':
        gen_score = np.median
    else:
        raise ValueError('Invalid metric {}. Should be `mean` or `median`.'
                         ''.format(comparison))
    # Setup:
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        print('')
    else:
        print('')
        start_time = datetime.now()
        results = _nested_point632plus(
            *args, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        duration = datetime.now() - start_time()
        print('')

    return results


def _nested_point632plus(
        X, y,
        n_splits,
        random_state,
        path_tmp_results,
        estimator, hparam_grid,
        selector,
        n_jobs, verbose, score_func, gen_performance
    ):
    # Worker function for the nested .632+ Out-of-Bag model comparison scheme.

    results = {'experiment_id': random_state}
    features = np.zeros(X.shape[1], dtype=int)

    sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    # Outer loop for best models average performance.
    train_scores, test_scores, opt_hparams = [], [], []
    for num, (train_idx, test_idx) in enumerate(sampler.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_model, best_support = oob_exhaustive_search(
            estimator, hparam_grid, selector, X_train, y_train, n_splits,
            random_state, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        best_model = _check_estimator(
            np.size(best_support), best_model.get_params(), estimator,
            random_state=random_state
        )
        train_score, test_score = utils.scale_fit_predict632(
            best_model, X_train[:, best_support], X_test[:, best_support],
            y_train, y_test, score_func=score_func, gen_performance
        )
        train_scores.append(train_score), test_scores.append(test_score)
        # Bookeeping of best feature subset and hparams in each fold.
        features[best_support] += 1
        opt_hparams.append(best_model.get_params())

    # NOTE: Selecting mode of hparams as opt hparam settings.
    try:
        best_model_hparams = max(opt_hparams, key=opt_hparams.count)
    # NOTE: In case all hparams have equal votes.
    except:
        best_model_hparams = opt_hparams

    # Retain features with max activations.
    best_support = np.squeeze(np.where(features == np.max(features)))

    end_results = _update_prelim_results(
        results, path_tempdir, random_state, estimator, selector,
        best_model_hparams, np.mean(test_scores), np.mean(train_scores),
        best_support
    )
    return end_results


def oob_exhaustive_search(
        *args, verbose=1, score_func=None, n_jobs=1, gen_performance
    ):

    (
        estimator, hparam_grid, selector, X, y, n_splits, random_state
    ) = args

    oob_sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    # Inner loop for model selection.
    best_test_score = -np.inf
    best_model, best_support = [], []
    for combo_num, hparams in enumerate(hparam_grid):

        # Setup:
        features = np.zeros(X.shape[1], dtype=int)

        train_scores, test_scores = [], []
        for num, (train_idx, test_idx) in enumerate(oob_sampler.split(X, y)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # NOTE: Standardizing in feature sel function.
            X_train_sub, X_test_sub, support = selector['func'](
                (X_train, X_test, y_train, y_test), **selector['params']
            )
            model = _check_estimator(
                np.size(support), hparams, estimator, random_state=random_state
            )
            train_score, test_score = utils.scale_fit_predict632(
                model, X_train_sub, X_test_sub, y_train, y_test,
                score_func=score_func
            )
            # Bookkeeping of features selected in each fold.
            features[support] += 1
            train_scores.append(train_score), test_scores.append(test_score)

        # Median or mean of scores.
        if gen_performance(test_scores) > best_test_score:
            best_test_score = gen_performance(test_scores)
            # Retain features of max activations.
            best_support = np.squeeze(np.where(features == np.max(features)))
            #Re-instantiate a new untrained model.
            best_model = _check_estimator(
                np.size(support), hparams, estimator, random_state=random_state
            )
        return best_model, best_support
