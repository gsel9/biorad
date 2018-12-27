# -*- coding: utf-8 -*-
#
# nested_632plus.py
#
# Checkout: https://github.com/tmadl/highdimensional-decision-boundary-plot
#
# REFACTORING:
# * Generate numpy objects and numpy iters in loops.
# * Generate inner and outer Z-score samples in one loop.
# * Use numpy.random.choice(a, size=None, replace=True, p=None)
# * Use a function to compute the whole series of
# * Checkout: numpy.fromfunction, frompyfunc(func, nin, nout), apply_along_axis(func1d, axis, arr, *args, …)

# ToDos:
# * Hyperopt package for hparam opt
# * sklearn Pipeline as objective function
# * Consider model selection and estimatoras one model with a common set of hparams.
# * Optimize on AUC, but also include precision, recall, f-beta and support (sklearn)


def outer_iterations():

    pass


def inner_iterations():

    # Do hparam selection
    # Evalaute performance
    pass


class LearningProcedure:

    # * Combine Z-score transformation, feature selection and estimator in a pipeline.
    # * Both FS and E have hyperparameters.
    # * Compare on errors (always >= 0, see TT paper for proof of Nonnegativity of the bias.).
    pass



def css(results, y, loss):
    """Configuration selection strategy as described by Tsamardinos & Greasidou
    (2018).

    Args:
        results (array-like): A matrix (N x C) containing out-of-sample
            predictions for N sapmles and C hyperparameter configurations. Thus,
            results[i, j] denotes the out-of-sample prediction of on the i-th
            sample of the j-th configuration.
        y (array-like): Ground truths corresponding to results.
        loss (function): Score criterion.

    Returns:
        (int): Index of the best performing configuration according to the
            loss criterion.

    """

    # Use numpy.argmin() to quickly determine optimal run.
    pass



def hyperopt_search(args, data, model, param_grid, max_evals):

    def objective(param_grid):
        # Create a pipeline object
        args.num_hidden = param_grid['num_hidden']
        args.dropout_output = param_grid['dropout_output']
        args.dropout_input = param_grid['dropout_input']
        args.clip_norm = param_grid['clip_norm']
        args.batch_size = param_grid['batch_size']
        # args.learning_rate = param_grid['learning_rate']
        print(args)
        print()
        scores = run_network(args, data, model, tuning=args.tune)
        test_score, eval_score = scores
        tf.reset_default_graph()
        eval_score = -eval_score[0]
        return {'loss': eval_score, 'params': args, 'status': STATUS_OK}

    trials = Trials()
    results = fmin(
        objective, param_grid, algo=tpe.suggest,
        trials=trials, max_evals=max_evals)

    return results, trials.results





"""
The nested .632+ bootstrap Out-of-Bag procedure for model selection.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os

import feature_selection

import numpy as np
import pandas as pd

from utils import ioutil, fwutils

from numba import jit, vectorize, float64
from datetime import datetime
from collections import OrderedDict
from sklearn.externals import joblib


def nested_point632plus(
        X, y,
        n_splits,
        random_state,
        path_tmp_results,
        estimator,
        hparam_grid,
        balance=True,
        selector=None,
        n_jobs=1, verbose=1,
        score_func=None, score_eval=None
    ):
    """Mested model performance evaluation according to the .632+ bootstrap
    method including Synthetic Minority Over-sampling.

    Args:
        X (array-like):
        y (array-like):
        n_splits (int):
        random_state (int):
        path_tmp_results (str):
        estimator ():
        hparam_grid ():
        selector ():

    Returns:
        (dict):

    """
    # Setup:
    path_case_file = os.path.join(
        path_tmp_results, '{}_{}_{}'.format(
            estimator.__name__, selector.name, random_state
        )
    )
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        print('Reloading results from: {}'.format(path_case_file))
    else:
        # Apply SMOTE to balance target class distributions.
        if balance:
            X, y = fwutils.balance_data(X, y, random_state)

        print('Initiating experiment: {}'.format(random_state))
        start_time = datetime.now()
        results = _nested_point632plus(
            X, y,
            n_splits,
            random_state,
            path_tmp_results,
            estimator,
            hparam_grid,
            selector,
            n_jobs, verbose,
            score_func, score_eval
        )
        duration = datetime.now() - start_time
        print('Experiment {} completed in {}'.format(random_state, duration))
    return results


def _nested_point632plus(
        X, y,
        n_splits,
        random_state,
        path_tmp_results,
        estimator,
        hparam_grid,
        selector,
        n_jobs, verbose,
        score_func, score_eval
    ):
    # Bookeeping results and feature votes.
    results = {'experiment_id': random_state}
    features = np.zeros(X.shape[1], dtype=int)
    # Outer OOB resampler.
    sampler = fwutils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    train_scores, test_scores, opt_hparams = [], [], []
    for split_num, (train_idx, test_idx) in enumerate(sampler.split(X, y)):

        if verbose > 0:
            print('Outer loop iter number {}'.format(split_num))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Perform exhaustive hyperparameter search.
        best_model, best_support = oob_grid_search(
            X_train, y_train,
            n_splits,
            random_state,
            estimator, hparam_grid,
            selector,
            n_jobs, verbose,
            score_func, score_eval
        )
        # NOTE: Z-score transformation and error handlng included in function.

        # NB: Training new model (can be avoided with TT method)
        train_score, test_score = scale_fit_predict632(
            X_train[:, best_support], X_test[:, best_support],
            y_train, y_test,
            score_func,
            best_model
        )
        # NOTE: Error handling mechanism.
        if train_score is None and test_score is None:
            pass
        else:
            train_scores.append(train_score), test_scores.append(test_score)
            opt_hparams.append(best_model.get_params())
            features[best_support] += 1
        # Apply mode to all opt hparam settings.
        best_model_hparams = fwutils.select_hparams(opt_hparams)
        # Retain features with max activations.
        best_support, support_votes = fwutils.select_support(features)

    # Callback handling of preliminary results.
    final_results = ioutil.update_prelim_results(
        path_tmp_results,
        score_eval(train_score),
        score_eval(test_score),
        train_scores,
        test_scores,
        estimator.__name__,
        best_model_hparams,
        selector.name,
        support_votes,
        best_support,
        random_state,
        results,
    )
    return final_results


def oob_grid_search(
        X, y,
        n_splits,
        random_state,
        estimator,
        hparam_grid,
        selector,
        n_jobs, verbose,
        score_func, score_eval
    ):
    """Perform hyperparameter optimization according to the .632+ bootstrap
    Out-of-Bag method.

    Args:

    Returns:
        (tuple): The winning model and the selected features (optional).

    """
    sampler = fwutils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    # Exhaustive hyperparameter search.
    best_test_score = -1.0 * np.inf
    best_model, best_support = None, None
    for combo_num, hparams in enumerate(hparam_grid):

        if verbose > 1:
            print('Hyperparameter combo number {}'.format(combo_num))

        # Bookeeping of feature votes.
        features = np.zeros(X.shape[1], dtype=int)

        train_scores, test_scores = [], []
        for split_num, (train_idx, test_idx) in enumerate(sampler.split(X, y)):

            if verbose > 1:
                print('Inner loop iter number {}'.format(split_num))

            # NB: Expensive (FS + model training)
            train_score, test_score, support = objective(
                X[train_idx], X[test_idx],
                y[train_idx], y[test_idx],
                estimator,
                hparams,
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
        # Update current best results.
        if score_eval(test_scores) > best_test_score:
            best_test_score = score_eval(test_scores)
            best_support, _ = fwutils.select_support(features)
            # Re-instantiate the best candidate model.
            best_model = fwutils.check_estimator(
                estimator,
                hparams,
                support_size=len(support),
                random_state=random_state
            )
        return best_model, best_support


def objective(*args):
    # Maps a modeling procedure to a score value.
    (
        X_train, X_test,
        y_train, y_test,
        estimator,
        hparams,
        selector,
        score_func,
        random_state,
    ) = args
    # Reconstruct a model prior to feature selection.
    model = fwutils.check_estimator(
        estimator,
        hparams,
        support_size=None,
        random_state=random_state
    )
    # NOTE: Z-score transformation and error handlng included in selector.
    X_train_sub, X_test_sub, support = selector(
        X_train, X_test, y_train, y_test,
        random_state=random_state,
        score_func=score_func,
        model=model
    )
    # Reconstruct a model prior to predictions.
    model = fwutils.check_estimator(
        estimator,
        hparams,
        support_size=len(support),
        random_state=random_state
    )
    # NOTE: Z-score transformation and error handlng included in function.
    train_score, test_score = scale_fit_predict632(
        X_train_sub, X_test_sub,
        y_train, y_test,
        score_func,
        model
    )
    return train_score, test_score, support


def surrogate():
    # The surrogate function, also called the response surface, is the
    # probability representation of the objective function built using previous
    # evaluations.

    pass


def selection():
    # The criteria by which the next set of hyperparameters are chosen from the
    # surrogate function. The most common choice of criteria is Expected
    # Improvement.

    # NOTE: Involves integration: fast numpy/scipy.
    pass


def scale_fit_predict632(X_train, X_test, y_train, y_test, score_func, model):
    """Assess model performance on Z-score transformed training and test sets.

    Args:
        model (object): An untrained model with `fit()` and `predict` methods.
        X_train (array-like): Training set.
        X_test (array-like): Test set.
        y_train (array-like): Training set ground truths.
        y_test (array-like): Test set ground truths.
        score_func (function): A score function for model performance
            evaluation.

    Returns:
        (float):

    """
    # Compute Z scores.
    X_train_std, X_test_std = fwutils.train_test_z_scores(X_train, X_test)
    # NOTE: Error handling mechanism.
    try:
        model.fit(X_train_std, y_train)
    except:
        return None, None
    # Aggregate model predictions.
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)
    train_score = score_func(y_train, y_train_pred)
    test_score = score_func(y_test, y_test_pred)

    # Compute .632+ scores.
    train_632_score = point632plus_score(
        y_train, y_train_pred, train_score, test_score
    )
    test_632_score = point632plus_score(
        y_test, y_test_pred, train_score, test_score
    )
    return train_632_score, test_632_score


def point632plus_score(y_true, y_pred, train_score, test_score):
    """Compute .632+ score for binary classification.

    Args:
        y_true (array-like): Ground truths.
        y_pred (array-like): Predictions.
        train_score (float): Resubstitution score.
        test_score (float): True score.

    Returns:
        (float): The .632+ score value.

    """

    @vectorize([float64(float64, float64, float64)])
    def _relative_overfit_rate(train_score, test_score, gamma):
        # Relative Overfiting Rate as described in ....
        if test_score > train_score and gamma > train_score:
            return (test_score - train_score) / (gamma - train_score)
        else:
            return 0

    @jit
    def _no_info_rate_binary(y_true, y_pred):
        # The No Information Rate as described in ...
        # NB: Only applicable to a dichotomous classification problem.
        p_one = sum(y_true) / len(y_true)
        q_one = sum(y_pred) / len(y_pred)

        return p_one * (1 - q_one) + (1 - p_one) * q_one


    @vectorize([float64(float64, float64, float64, float64)])
    def _point632plus(train_score, test_score, r_marked, test_score_marked):
        #
        point632 = 0.368 * train_score + 0.632 * test_score
        frac = (0.368 * 0.632 * r_marked) / (1 - 0.368 * r_marked)

        return point632 + (test_score_marked - train_score) * frac

    gamma = _no_info_rate_binary(y_true, y_pred)
    # Calculate adjusted parameters as described in Efron & Tibshiranir paper.
    test_score_marked = min(test_score, gamma)
    r_marked = relative_overfit_rate(train_score, test_score, gamma)

    return point632plus(train_score, test_score, r_marked, test_score_marked)
