# -*- coding: utf-8 -*-
#
# nested_632plus.py
#
# Checkout: https://github.com/tmadl/highdimensional-decision-boundary-plot
#
# REFACTORING:
# * Generate numpy objects and numpy iters in loops.
# * Generate inner and outer Z-score samples in one loop.
# * Use a function to compute the whole series of
# * Checkout: numpy.fromfunction, frompyfunc(func, nin, nout), apply_along_axis(func1d, axis, arr, *args, …)
# * OBS: Bayesian Hyperparameter Optimization recommended from stackoverflow:
#   See https://automl.github.io/SMAC3/stable/quickstart.html (with scikit-learn models).

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

from sklearn.model_selection import RandomizedSearchCV


def experiment(random_state, balancing):
    """Work function for parallell BBC-CV experiments."""

    if balancing:
        # Add SMOTE to pipeline passing random states.
        pass


class BootstrapBiasCorrectionCV:
    """The Bootstrap Bias Corrected Cross-Validation method propsoed by
    Tsamardinos & Greasidou (2018).

    Args:
        ():

    """
    # - Generate out-of-sample predictions (results) with a tuning procedure.
    # * Parallellize tuning procedure.
    # - Sample N rows from results with replacement (+ corresponding ground
    #   truths) to produce B bootstrapped matrices.
    # * Use TensorFlow as GPU-accelerated NumPy framework for parallellizing bootstrap samplings.
    # * Parameterize feature selection procedures with an option to select a
    #   priori number of features resulting in comparable dimensionality of
    #   feature spaces.
    # * Export models to JSON for production models with pipeline.

    def __init__(
        self,
        num_rounds=500,
        alpha=0.05,
        n_jobs=1,
        random_states=None,
        balancing=True,
    ):
        self.num_rounds = num_rounds
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.random_states = random_states
        self.balancing = balancing
        self.early_stopping = early_stopping

    def execute(self, X, y):

        best_model, loss, results = self.hparam_selection()

        # create results matrix by storing each prediction vector per hparam
        # configuraion. Must also store correspoding ground truths and sampled
        # hparam configurations for analysis.

        # Bootstraping of pooled out-of-sample predictions.

        # uniform re-sampling with replacement of rows of the datase
        np.random.choice(a, size=None, replace=True, p=None)

    def hparam_selection(
        self, X, y, cv, scoring, n_iter, n_jobs, verbose, random_state
    ):
        """

        """
        # TODO: Label and store each hparam configuration
        searcher = RandomizedSearchCV(
            estimator,
            param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            # NOTE: iid = False corresponds to the standard definition of
            # cross-validation.
            iid=False,
            refit=False,
            # Stratified K-fold.
            cv=cv,
            verbose=verbose,
            random_state=random_state,
            error_score=np.nan,
            return_train_score=True
        )
        searcher.fit(X, y)

        return searcher

    @staticmethod
    def css(results, y_true, loss):
        """Configuration selection strategy as described by Tsamardinos &
        Greasidou (2018).

        Args:
            results (array-like): A matrix (N x C) containing out-of-sample
                predictions for N sapmles and C hyperparameter configurations.
                Thus, results[i, j] denotes the out-of-sample prediction of on
                the i-th sample of the j-th configuration.
            y_true (array-like): Ground truths corresponding to results.
            loss (function): Score criterion.

        Returns:
            (int): Index of the best performing configuration according to the
                loss criterion.

        """

        # Compute scores for each column set of predictions.
        # NOTE: Passing ground truths as additional argument to loss function.
        scores = np.apply_along_axis(loss, axis=0, arr=results, y_true=y_true)
        # Select the configuration with the minimum average loss
        return np.argmin(scores)

    @property
    def confidence(self):
        """A (1 - alpha) confidence interval for parameter estimates as
        described by Efron and Tibshirani (1993)."""

        sorted_loss = sorted(loss)
        return sorted_loss[self.alpha / 2, (1 - self.alpha / 2)]


def scale_fit_predict(X_train, X_test, y_train, y_test, score_func, model):
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


    # NOTE: Error handling.
    try:
        model.fit(X_train_std, y_train)
    except:
        return None, None

    # Aggregate model predictions.
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)
    train_score = score_func(y_train, y_train_pred)
    test_score = score_func(y_test, y_test_pred)

    return train_score, test_score
