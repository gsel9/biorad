
# -*- coding: utf-8 -*-
#
# model_selection.py
#

"""

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

import os
import time
import utils
import warnings

import numpy as np

from copy import deepcopy
from datetime import datetime
from collections import OrderedDict

from numba import jit, vectorize, float64

from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
from hyperopt import space_eval
from hyperopt import STATUS_OK

from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict


def model_selection(
        X, y,
        experiment_id,
        pipe_and_params,
        score_func,
        cv=5,
        oob=None,
        output_dir=None,
        max_evals=None,
        verbose=1,
        shuffle=True,
        random_state=None,
        path_tmp_results=None,
        error_score='all',
    ):
    """
    Work function for parallelizable model selection experiments.

    Args:

        cv (int): The number of folds in stratified k-fold cross-validation.
        oob (int): The number of samples in out-of-bag bootstrap re-sampling.
        max_evals (int): The number of iterations in hyperparameter search.

    """
    # Unpack setup.
    model, hparam_space = pipe_and_params[experiment_id]

    # Determine if storing preliminary results.
    if path_tmp_results is not None:
        path_case_file = os.path.join(
            path_tmp_results, 'experiment_{}_{}'.format(
                random_state, experiment_id
            )
        )
    else:
        path_case_file = ''

    # Determine if results already exists.
    if os.path.isfile(path_case_file):
        output = utils.ioutil.read_prelim_result(path_case_file)
        print('Reloading results from: {}'.format(path_case_file))
    else:
        output = {'exp_id': random_state, 'model_id': experiment_id}
        if verbose > 0:
            print('Running experiment: {}'.format(random_state))
            start_time = datetime.now()

        optimizer = SMACSearchCV(
            cv=cv,
            model=model,
            max_evals=max_evals,
            score_func=score_func,
            hparam_space=hparam_space,
            random_state=random_state,
            shuffle=shuffle,
            verbose=verbose,
            output_dir=output_dir,
            store_predictions=True
        )
        optimizer.fit(X, y)

        # Estimate average model performace.
        _model = deepcopy(model)
        _model.set_params(**optimizer.best_config)
        if oob is None:
            results = cross_val_score(
                X, y, cv, shuffle, random_state, _model, score_func
            )
        else:
            # TODO:
            #results = bbc_cv_score(X, y, oob, random_state, model)
            pass

        output.update(results)
        if path_tmp_results is not None:
            print('Writing results...')
            utils.ioutil.write_prelim_results(path_case_file, output)

        if verbose > 0:
            durat = datetime.now() - start_time
            print('Experiment {} completed in {}'.format(random_state, durat))
            output['exp_duration'] = durat

    return output


def cross_val_score(X, y, cv, shuffle, random_state, model, score_func):

    test_scores, train_scores = [], []
    folds = StratifiedKFold(cv, shuffle, random_state)
    for train_idx, test_idx in folds.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        test_scores.append(
            score_func(y_test, np.squeeze(model.predict(X_test)))
        )
        train_scores.append(
            score_func(y_train, np.squeeze(model.predict(X_train)))
        )
    return OrderedDict(
        [
            ('test_score', np.mean(test_scores)),
            ('train_score', np.mean(train_scores)),
            ('test_score_variance', np.var(test_scores)),
            ('train_score_variance', np.var(train_scores)),
        ]
    )


def bbc_cv_selection(
    X, y,
    experiment_id,
    model,
    hparam_space,
    score_func,
    n_splits,
    output_dir,
    max_evals,
    verbose=1,
    shuffle=True,
    random_state=None,
    path_tmp_results=None,
    error_score='all',
):
    """
    Work function for parallelizable model selection experiments.

    Args:

        cv (int): The number of folds in stratified k-fold cross-validation.
        oob (int): The number of samples in out-of-bag bootstrap re-sampling.
        max_evals (int): The number of iterations in hyperparameter search.

    """
    # Determine if storing preliminary results.
    if path_tmp_results is not None:
        path_case_file = os.path.join(
            path_tmp_results, 'experiment_{}_{}'.format(
                random_state, experiment_id
            )
        )
    else:
        path_case_file = ''
    # Determine if results already exists.
    if os.path.isfile(path_case_file):
        output = utils.ioutil.read_prelim_result(path_case_file)
        print('Reloading results from: {}'.format(path_case_file))
    else:
        output = {'exp_id': random_state, 'model_id': experiment_id}
        if verbose > 0:
            print('Running experiment: {}'.format(random_state))
            start_time = datetime.now()

        optimizer = SMACSearchCV(
            cv=cv,
            model=model,
            max_evals=max_evals,
            score_func=score_func,
            hparam_space=hparam_space,
            random_state=random_state,
            shuffle=shuffle,
            verbose=verbose,
            output_dir=output_dir,
            store_predictions=True
        )
        optimizer.fit(X, y)

        # Evaluate model performance with BBC-CV method.
        bbc_cv = BootstrapBiasCorrectedCV(
            random_state=random_state,
            score_func=score_func,
            alpha=0.05,
            oob=oob,
        )
        # Add results from parameter search to output. Particularily usefull in
        # assessing if sufficient number of objective function evaluations.
        output.update(bbc_cv.evaluate(*optimizer.oos_pairs))
        output.update(optimizer.test_loss)
        output.update(optimizer.train_loss)
        output.update(optimizer.test_loss_var)
        output.update(optimizer.train_loss_var)
        output.update(optimizer.best_config)

        if path_tmp_results is not None:
            print('Writing results...')
            utils.ioutil.write_prelim_results(path_case_file, output)

        if verbose > 0:
            durat = datetime.now() - start_time
            print('Experiment {} completed in {}'.format(random_state, durat))
            output['exp_duration'] = durat

    return output


def nested_kfold_selection(
    X, y,
    experiment_id,
    model,
    hparam_space,
    score_func,
    n_splits,
    output_dir,
    max_evals,
    verbose=1,
    shuffle=True,
    random_state=None,
    path_tmp_results=None,
    error_score='all',
):
    """
    Work function for parallelizable model selection experiments.

    Args:

        cv (int): The number of folds in stratified k-fold cross-validation.
        oob (int): The number of samples in out-of-bag bootstrap re-sampling.
        max_evals (int): The number of iterations in hyperparameter search.

    """
    # Determine if storing preliminary results.
    if path_tmp_results is not None:
        path_case_file = os.path.join(
            path_tmp_results, 'experiment_{}_{}'.format(
                random_state, experiment_id
            )
        )
    else:
        path_case_file = ''
    # Determine if results already exists.
    if os.path.isfile(path_case_file):
        output = utils.ioutil.read_prelim_result(path_case_file)
        print('Reloading results from: {}'.format(path_case_file))
    else:
        output = {'exp_id': random_state, 'model_id': experiment_id}
        if verbose > 0:
            print('Running experiment: {}'.format(random_state))
            start_time = datetime.now()

        results = nested_kfold(
                X=X, y=y,
                experiment_id=experiment_id,
                model=model,
                hparam_space=hparam_space,
                score_func=score_func,
                n_splits=n_splits,
                output_dir=output_dir,
                max_evals=max_evals,
                verbose=verbose,
                shuffle=shuffle,
                random_state=random_state,
                path_tmp_results=path_tmp_results,
                error_score=error_score,
            )

        output.update(results)
        if path_tmp_results is not None:
            print('Writing results...')
            utils.ioutil.write_prelim_results(path_case_file, output)

        if verbose > 0:
            durat = datetime.now() - start_time
            print('Experiment {} completed in {}'.format(random_state, durat))
            output['exp_duration'] = durat

    return output


class BootstrapBiasCorrectedCV:
    """

    Args:
        score_func (function):
        n_iter (int):
        random_state (int):

    """

    def __init__(
        self,
        random_state,
        score_func,
        error_score=np.nan,
        oob=200,
        alpha=0.05,
    ):
        self.score_func = score_func
        self.oob = oob
        self.alpha = alpha
        self.random_state = random_state
        self.error_score = error_score

        self._sampler = None

    # TODO: Vectorization.
    def evaluate(self, Y_pred, Y_true):
        """Bootstrap bias corrected cross-validation proposed by .

        Args:
            Y_pred (array-like): A matrix (N x C) containing out-of-sample
                predictions for N samples and C hyperparameter configurations.
                Thus, scores[i, j] denotes the out-of-sample prediction of on
                the i-th sample of the j-th configuration.
            Y_true ():

        Returns:
            (dict):

        """
        if self._sampler is None:
            self._sampler = utils.sampling.OOBSampler(
                self.oob, self.random_state
            )
        bbc_scores = []
        # Sample rows with replacement from the prediction matrix.
        for train_idx, test_idx in self._sampler.split(Y_true, Y_pred):
            optimal_config = self.criterion(
                Y_true[train_idx, :], Y_pred[train_idx, :]
            )
            bbc_scores.append(
                self._score(
                    Y_true[test_idx, optimal_config],
                    Y_pred[test_idx, optimal_config]
                )
            )
        return {
            'oob_avg_score': np.nanmean(bbc_scores),
            'oob_std_score': np.nanstd(bbc_scores),
            'oob_oob_ci': self.bootstrap_ci(bbc_scores),
        }

    def _score(self, y_true, y_pred):
        # Score function error mechanism.
        try:
            output = self.score_func(y_true, y_pred)
        except:
            output = self.error_score

        return output

    def criterion(self, Y_true, Y_pred):
        """Given a set of selected samples of predictions and ground truths,
        determine the optimal configuration index from the sample subset.

        Returns:
            (int): Index of the optimal configuration according to the
                score function.

        """
        _, num_configs = np.shape(Y_true)
        losses = np.ones(num_configs, dtype=float) * np.nan
        for num in range(num_configs):
            # Calculate the loss for each configuration. Returns <float> or
            # error score (1 - NaN = NaN).
            losses[num] = 1.0 - self._score(Y_true[:, num], Y_pred[:, num])
        # Select the configuration corresponding to the minimum loss.
        return np.nanargmin(losses)

    def bootstrap_ci(self, scores):
        """Calculate the bootstrap confidence interval from sample data."""

        upper_idx = (1 - self.alpha / 2) * len(scores)
        lower_idx = self.alpha / 2 * len(scores)

        asc_scores = sorted(scores)
        return asc_scores[int(lower_idx)], asc_scores[int(upper_idx)]


# TODO:
# * Multi scoring with
#   >>> scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
def nested_kfold(
    X, y,
    experiment_id,
    model,
    hparam_space,
    score_func,
    n_splits,
    max_evals,
    shuffle=True,
    verbose=1,
    output_dir=None,
    random_state=None,
    path_tmp_results=None,
    error_score=np.nan,
):
    if verbose > 0:
        start_search = datetime.now()
        print('Entering parameter search')

    optimizer = SMACSearch(
        model=model,
        hparam_space=hparam_space,
        score_func=score_func,
        n_splits=n_splits,
        shuffle=shuffle,
        verbose=verbose,
        max_evals=max_evals,
        random_state=random_state,
        output_dir=output_dir
    )
    optimizer.fit(X, y)

    if verbose > 0:
        end_search = datetime.now() - start_search
        print('Parameter search finished in {}'.format(end_search))

    test_scores, train_scores = [], []
    folds = StratifiedKFold(n_splits, shuffle, random_state)
    for train_idx, test_idx in folds.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        _model = deepcopy(model)
        _model.set_params(**optimizer.best_config)
        _model.fit(X_train, y_train)

        test_scores.append(
            score_func(y_test, np.squeeze(_model.predict(X_test)))
        )
        train_scores.append(
            score_func(y_train, np.squeeze(_model.predict(X_train)))
        )
    if verbose > 0:
        print('Score CV finished in {}'.format(datetime.now() - end_search))

    # NOTE: Consider using median rather than mean.
    return OrderedDict(
        [
            ('test_score', np.nanmedian(test_scores)),
            ('train_score', np.nanmedian(train_scores)),
            ('test_score_variance', np.nanvar(test_scores)),
            ('train_score_variance', np.nanvar(train_scores)),
        ]
    )


class SMACSearchCV:

    def __init__(
        self,
        cv=None,
        model=None,
        max_evals=None,
        score_func=None,
        hparam_space=None,
        run_objective='quality',
        random_state=None,
        shuffle=True,
        deterministic=True,
        output_dir=None,
        verbose=0,
        abort_first_run=False,
        early_stopping=50,
        store_predictions=False
    ):
        self.cv = cv
        self.model = model
        self.max_evals = max_evals
        self.score_func = score_func
        self.hparam_space = hparam_space
        self.run_objective = run_objective
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.output_dir = output_dir
        self.abort_first_run = abort_first_run

        self.early_stopping = early_stopping
        self.store_predictions = store_predictions
        if deterministic:
            self.deterministic = 'true'
        else:
            self.deterministic = 'false'

        self._rgen = None
        self._best_config = None
        self._current_min = None
        self._objective_func = None
        self._predictions = None

    @property
    def best_config(self):

        return self._best_config

    @property
    def best_model(self):

        _model = deepcopy(self.model)
        _model.set_params(**self.best_config)

        return _model

    def fit(self, X, y):

        # NB: Carefull!
        self.X, self.y = self._check_X_y(X, y)

        if self._rgen is None:
            self._rgen = np.random.RandomState(self.random_state)

        if self._current_min is None:
            self._current_min = float(np.inf)

        # NOTE: see https://github.com/automl/auto-sklearn/issues/345 for
        # info on `abort_on_first_run_crash`.
        scenario = Scenario(
            {
                'run_obj': self.run_objective,
                'runcount-limit': self.max_evals,
                'cs': self.hparam_space,
                'deterministic': self.deterministic,
                'output_dir': self.output_dir,
                'abort_on_first_run_crash': self.abort_first_run
             }
        )
        smac = SMAC(
            scenario=scenario,
            rng=self._rgen,
            tae_runner=self.cv_objective
        )
        self._best_config = smac.optimize()

        return self

    def cv_objective(self, hparams):

        if self.early_stopping < 1:
            warnings.warn('Exiting by early stopping.')
            return self._best_params

        test_scores = []
        folds = StratifiedKFold(self.cv, self.shuffle, self.random_state)
        for train_idx, test_idx in folds.split(self.X, self.y):

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            _model = deepcopy(self.model)
            _model.set_params(**hparams)
            _model.fit(X_train, y_train)

            test_scores.append(
                self.score_func(y_test, np.squeeze(_model.predict(X_test)))
            )
        loss = 1.0 - np.mean(test_scores)
        # Early stopping mechanism.
        if self._current_min < loss:
            self.early_stopping = self.early_stopping - 1
        else:
            self._current_min = loss

        return loss

    @staticmethod
    def _check_X_y(X, y):
        # Wrapping the sklearn formatter function.
        return check_X_y(X, y)


class SMACSearchOOBCV:

    def __init__(self):

        # Ground truths and predictions for BBC-CV procedure.
        self.Y_pred = None
        self.Y_test = None
        self._sample_lim = None

    def _setup_pred_containers(self):

        nrows, _ = np.shape(self.X)
        # With scikit-learn CV:
        # * The first n_samples % n_splits folds have size
        #   n_samples // n_splits + 1, other folds have size
        #   n_samples // n_splits, where n_samples is the number of samples.
        # If stratified CV:
        # * Train and test sizes may be different in each fold,
        #   with a difference of at most n_classes
        # Adjust the sample limit with be the number of target classes to
        # ensure all predictions have equal size when storing outputs for
        # BBC-CV procedure.
        self._sample_lim = nrows // self.cv - len(set(self.y))

        # Setup containers from predictions and corresponding ground truths to
        # be used with BBC-CV procedure. The containers must hold the set of
        # predictinos for each suggested hyperparamter configuration as the
        # optimal. That is, predictions must be stored each time the objective
        # function is called by the optimization algorithm.
        self.Y_pred = np.zeros((self._sample_lim, self.max_evals), dtype=int)
        self.Y_test = np.zeros((self._sample_lim, self.max_evals), dtype=int)

        return self

    def fit(self, X, y):

        # NB: Carefull!
        self.X, self.y = self._check_X_y(X, y)

        if self._rgen is None:
            self._rgen = np.random.RandomState(self.random_state)

        if self._current_min is None:
            self._current_min = float(np.inf)

        if self.Y_pred is None and self.Y_test is None:
            self._setup_pred_containers()

        # NOTE: see https://github.com/automl/auto-sklearn/issues/345 for
        # info on `abort_on_first_run_crash`.
        scenario = Scenario(
            {
                'run_obj': self.run_objective,
                'runcount-limit': self.max_evals,
                'cs': self.hparam_space,
                'deterministic': self.deterministic,
                'output_dir': self.output_dir,
                'abort_on_first_run_crash': self.abort_first_run
             }
        )
        smac = SMAC(
            scenario=scenario,
            rng=self._rgen,
            tae_runner=self.cv_objective
        )
        self._best_config = smac.optimize()

        return self

    @property
    def oos_pairs(self):
        """Returns a tuple with ground truths and corresponding out-of-sample
        predictions."""

        preds = np.transpose(
            [items['y_pred'] for items in self.trials.results]
        )
        trues = np.transpose(
            [items['y_true'] for items in self.trials.results]
        )
        return preds, trues
