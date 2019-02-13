
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

from mlxtend.feature_selection import SequentialFeatureSelector


def model_selection(
        X, y,
        experiment_id,
        hparam_space,
        workflow,
        score_func,
        cv=5,
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
        output = {'exp_id': random_state, 'experiment_id': experiment_id}
        if verbose > 0:
            print('Running experiment: {}'.format(random_state))
            start_time = datetime.now()

        optimizer = SMACSearchCV(
            cv=cv,
            workflow=workflow,
            max_evals=max_evals,
            score_func=score_func,
            random_state=random_state,
            shuffle=shuffle,
            verbose=verbose,
            output_dir=output_dir,
            store_predictions=True
        )
        optimizer.fit(X, y)

        _workflow = deepcopy(workflow)
        _workflow.set_params(**optimizer.best_config)

        # Estimate average performance of best model.
        results = cross_val_score(
            X, y, cv, shuffle, random_state, _workflow, score_func
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


def cross_val_score(X, y, cv, shuffle, random_state, workflow, score_func):

    test_scores, train_scores = [], []
    folds = StratifiedKFold(cv, shuffle, random_state)
    for train_idx, test_idx in folds.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        workflow.fit(X_train, y_train)

        test_scores.append(
            score_func(y_test, np.squeeze(workflow.predict(X_test)))
        )
        train_scores.append(
            score_func(y_train, np.squeeze(workflow.predict(X_train)))
        )
    return OrderedDict(
        [
            ('test_score', np.mean(test_scores)),
            ('train_score', np.mean(train_scores)),
            ('test_score_variance', np.var(test_scores)),
            ('train_score_variance', np.var(train_scores)),
        ]
    )


class SMACSearchCV:

    def __init__(
        self,
        cv=None,
        workflow=None,
        max_evals=None,
        score_func=None,
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
        self.workflow = workflow
        self.max_evals = max_evals
        self.score_func = score_func
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
    def best_workflow(self):

        _workflow = deepcopy(self.workflow)
        _workflow.set_params(**self.best_config)

        return _workflow

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
                'cs': self.workflow.hparams,
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

            _workflow = deepcopy(self.workflow)
            _workflow.set_params(**hparams)
            _workflow.fit(X_train, y_train)

            test_scores.append(
                self.score_func(y_test, np.squeeze(_workflow.predict(X_test)))
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
