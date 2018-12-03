# -*- coding: utf-8 -*-
#
# model_selection.py
#

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


THRESH = 1


def _check_estimator(nfeatures, hparams, estimator, random_state):

    # Using all available features after feature selection.
    if 'n_components' in hparams:
        if nfeatures - 1 < 1:
            hparams['n_components'] = 1
        else:
            hparams['n_components'] = nfeatures - 1

    # If stochastic algorithms.
    try:
        model = estimator(**hparams, random_state=random_state)
    except:
        model = estimator(**hparams)

    try:
        model.n_jobs = -1
    except:
        pass

    return model


def _update_prelim_results(results, path_tempdir, random_state, *args):
    # Update results <dict> container and write preliminary results to disk.
    (
        estimator, selector, best_params, avg_test_scores, avg_train_scores,
        best_features
    ) = args

    results.update(
        {
            'model': estimator.__name__,
            'selector': selector['name'],
            'best_params': best_params,
            'avg_test_score': avg_test_scores,
            'avg_train_score': avg_train_scores,
            'best_features': best_features,
            'num_features': np.size(best_features)
        }
    )
    # Write preliminary results to disk.
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    ioutil.write_prelim_results(path_case_file, results)

    return results


def nested_point632plus(*args, verbose=1, n_jobs=1, score_func=None):
    """

    """
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    # Setup:
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        if verbose > 0:
            print('Reloading previous results')

    else:
        if verbose > 0:
            start_time = datetime.now()
            print('Entering nested procedure with ID: {}'.format(random_state))
        results = _nested_point632plus(
            *args, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        if verbose > 0:
            delta_time = datetime.now() - start_time
            print('Collected results in: {}'.format(delta_time))

    return results


def _nested_point632plus(*args, n_jobs=None, score_func=None, verbose=1):
    # The worker function for the nested .632+ Out-of-Bag model comparison
    # scheme.
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    global THRESH

    # Setup:
    results = {'experiment_id': random_state}
    features = np.zeros(X.shape[1], dtype=int)

    oob_sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    # Outer loop for best models average performance.
    train_scores, test_scores, opt_hparams = [], [], []
    for num, (train_idx, test_idx) in enumerate(oob_sampler.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_model, best_support = grid_search_oob(
            estimator, hparam_grid, selector, X_train, y_train, n_splits,
            random_state, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        best_model = _check_estimator(
            np.size(best_support), best_model.get_params(), estimator,
            random_state=random_state
        )
        train_score, test_score = utils.scale_fit_predict632(
            best_model, X_train[:, best_support], X_test[:, best_support],
            y_train, y_test, score_func=score_func
        )
        train_scores.append(train_score), test_scores.append(test_score)
        # Bookeeping of best feature subset and hparams in each fold.
        features[best_support] += 1
        opt_hparams.append(best_model.get_params())

    try:
        # NOTE: Selecting mode of hparams as opt hparam settings.
        best_model_hparams = max(opt_hparams, key=opt_hparams.count)
    except:
        # In case all optimal hparams are the same max() equals none.
        best_model_hparams = opt_hparams

    end_results = _update_prelim_results(
        results, path_tempdir, random_state, estimator, selector,
        best_model_hparams, np.mean(test_scores), np.mean(train_scores),
        np.squeeze(np.where(features >= THRESH))
    )
    return end_results


def grid_search_oob(*args, verbose=1, score_func=None, n_jobs=1):

    (
        estimator, hparam_grid, selector, X, y, n_splits, random_state
    ) = args

    global THRESH

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

        if np.mean(test_scores) > best_test_score:
            best_test_score = np.mean(test_scores)
            best_model = _check_estimator(
                np.size(support), hparams, estimator, random_state=random_state
            )
            best_support = np.squeeze(np.where(features >= THRESH))

        return best_model, best_support
