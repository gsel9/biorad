# -*- coding: utf-8 -*-
#
# model_selection.py
#

"""
Model selection according to the nested .632+ nested bootstrap method developed
by Efron and Tibhirani.
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
        estimator, hparam_grid, selector,
        n_jobs=1, verbose=0, score_func=None, score_metric=None
    ):
    """Model performance evaluation according to the .632+ bootstrap method.

    """
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
        estimator, hparam_grid, selector,
        n_jobs, verbose, score_func, score_metric
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
        # Perform exhaustive hyperparameter search.
        best_model, best_support = oob_exhaustive_search(
            X_train, y_train
            n_splits,
            random_state,
            estimator, hparam_grid, selector,
            n_jobs=n_jobs, verbose=verbose,
            score_func=score_func, score_metric=score_metric
        )
        # In case of error:
        if best_model is None and best_support is None:
            train_scores.append(None), test_scores.append(None)
            opt_hparams.append(None)
        else:
            best_model = _check_estimator(
                np.size(best_support), best_model.get_params(), estimator,
                random_state=random_state
            )
            train_score, test_score = utils.scale_fit_predict632(
                best_model, X_train[:, best_support], X_test[:, best_support],
                y_train, y_test, score_func=score_func, score_metric
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
        results,
        path_tempdir,
        random_state,
        estimator, best_model_hparams,
        selector, best_support,
        score_metric(test_scores), score_metric(train_scores)
    )
    return end_results


def oob_exhaustive_search(
        X_train, y_train
        n_splits,
        random_state,
        estimator, hparam_grid,
        selector,
        n_jobs=n_jobs, verbose=verbose,
        score_func=score_func, score_metric=score_metric
    ):
    oob_sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    # Inner loop for model selection.
    best_test_score = -np.inf
    best_model, best_support = [], []
    for combo_num, hparams in enumerate(hparam_grid):
        #
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
            # NOTE: An error may occur during training.
            train_score, test_score = utils.scale_fit_predict632(
                model,
                X_train_sub, X_test_sub, y_train, y_test,
                score_func
            )
            # Bookkeeping of features selected in each fold.
            features[support] += 1
            train_scores.append(train_score), test_scores.append(test_score)

        # Generalize test scores.
        current_score = score_metric(test_scores)
        if current_score > best_test_score:
            best_test_score = current_score
            # Retain features of max activations.
            best_support = np.squeeze(np.where(features == np.max(features)))
            # Re-instantiate a new untrained model.
            best_model = _check_estimator(
                np.size(support), hparams, estimator, random_state=random_state
            )
        return best_model, best_support
