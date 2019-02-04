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


def nested_kfold_selection(
    X, y,
    experiment_id,
    model,
    hparam_space,
    score_func,
    cv,
    execdir,
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
            cv=cv,
            execdir=execdir,
            max_evals=max_evals,
            verbose=verbose,
            shuffle=shuffle,
            random_state=None,
            path_tmp_results=None,
            error_score=np.nan,
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


# TODO:
# * Multi scoring with
#   >>> scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
def nested_kfold(
    X, y,
    experiment_id,
    model,
    hparam_space,
    score_func,
    cv,
    max_evals,
    shuffle=True,
    verbose=1,
    execdir=None,
    random_state=None,
    path_tmp_results=None,
    error_score=np.nan,
):
    if verbose > 0:
        start_search = datetime.now()
        print('Entering parameter search')

    optimizer = SMACSearchCV(
        model=model,
        hparam_space=hparam_space,
        score_func=score_func,
        cv=cv,
        shuffle=shuffle,
        verbose=verbose,
        max_evals=max_evals,
        random_state=random_state,
        execdir=execdir
    )
    optimizer.fit(X, y)

    if verbose > 0:
        end_search = datetime.now() - start_search
        print('Parameter search finished in {}'.format(end_search))

    test_scores, train_scores = [], []
    folds = StratifiedKFold(cv, shuffle, random_state)
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
            #('hparams', )
        ]
    )


class SMACSearchCV:

    def __init__(
        self,
        model=None,
        hparam_space=None,
        score_func=None,
        cv=4,
        verbose=0,
        objective='quality',
        max_evals=2,
        random_state=None,
        shuffle=True,
        deterministic=True,
        execdir='./outputs',
        early_stopping=25
    ):
        self.model = model
        self.hparam_space = hparam_space
        self.score_func = score_func
        self.cv = cv
        self.shuffle = shuffle
        self.verbose = verbose
        # Optimization objective (quality or runtime).
        self.run_obj = objective
        # Maximum function evaluations.
        self.max_evals = max_evals
        # Configuration space.
        self.model = model
        self.execdir = execdir
        self.random_state = random_state
        self.early_stopping = early_stopping

        if deterministic:
            self.deterministic = 'true'
        else:
            self.deterministic = 'false'

        self._rgen = None
        self._best_config = None
        self._current_min = None

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

        # Scenario object.
        scenario = Scenario(
            {
                'run_obj': self.run_obj,
                'runcount-limit': self.max_evals,
                'cs': self.hparam_space,
                'deterministic': self.deterministic,
                'execdir': self.execdir
             }
        )
        # Optimize using a SMAC-object.
        smac = SMAC(
            scenario=scenario,
            rng=np.random.RandomState(self.random_state),
            tae_runner=self.objective
        )
        self._best_config = smac.optimize()

        return self

    def objective(self, hparams):

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
        if self._current_min < loss:
            self.early_stopping = self.early_stopping - 1
        else:
            self._current_min = loss

        return loss

    @staticmethod
    def _check_X_y(X, y):
        # Wrapping the sklearn formatter function.
        return check_X_y(X, y)


class TPESearchCV:
    """Perform K-fold cross-validated hyperparameter search with the Bayesian
    optimization Tree Parzen Estimator.

    Args:
        model ():
        space ():
        ...

    """

    def __init__(
        self,
        algo,
        model,
        space,
        score_func,
        cv=4,
        verbose=0,
        max_evals=100,
        shuffle=True,
        random_state=None,
        error_score=np.nan,
        balancing=True,
        early_stopping=30
    ):
        self.algo = algo
        self.model = model
        self.space = space
        self.verbose = verbose
        self.shuffle = shuffle
        self.score_func = score_func
        self.error_score = error_score
        self.cv = cv
        self.max_evals = max_evals
        self.random_state = random_state
        self.balancing = balancing
        self.early_stopping = early_stopping

        # NOTE: Attributes updated with instance.
        self.X = None
        self.y = None
        self.trials = None

        self._rgen = None
        self._best_params = None
        self._prev_score = float(np.inf)

    @property
    def params(self):

        params = {
            num: res['hparams']
            for num, res in enumerate(self.trials.best_trial)
        }
        return {'param_search_eval_params': params}

    @property
    def best_results(self):

        return min(self.trials.results, key=lambda item: item['loss'])

    @property
    def best_config(self):
        """Returns the optimal hyperparameter configuration."""

        return self.best_results['hparams']

    @property
    def best_model(self):
        """Returns an instance of the estimator with the optimal
        hyperparameters."""

        _model = deepcopy(self.model)
        _model.set_params(**self.best_config)

        return _model

    @property
    def train_loss(self):
        """Returns """

        losses = {
            num: res['train_loss']
            for num, res in enumerate(self.trials.results)
        }
        return {'param_search_training_loss': losses}

    @property
    def test_loss(self):
        """Returns """

        losses = {
            num: res['loss'] for num, res in enumerate(self.trials.results)
        }
        return {'param_search_test_loss': losses}

    @property
    def train_loss_var(self):
        """Returns a dict with the variance of each hyperparameter
        configuration for each K-fold cross-validated training loss."""

        losses = {
            num: res['train_loss_variance']
            for num, res in enumerate(self.trials.results)
        }
        return {'param_search_training_loss_var': losses}

    @property
    def test_loss_var(self):
        """Returns a dict with the variance of each hyperparameter
        configuration for each K-fold cross-validated test loss."""

        losses = {
            num: res['loss_variance']
            for num, res in enumerate(self.trials.results)
        }
        return {'param_search_test_loss_var': losses}

    def fit(self, X, y):
        """Optimal hyperparameter search.

        Args:
            X (array-like):
            y (array-like):

        """
        self.X, self.y = self._check_X_y(X, y)

        if self._rgen is None:
            self._rgen = np.random.RandomState(self.random_state)

        if self.trials is None:
            self.trials = Trials()

        # Passing random state to optimization algorithm renders randomly
        # selected seeds from hyperopt sampling reproduciable.
        self._best_params = fmin(
            self.objective,
            self.space,
            algo=self.algo,
            rstate=self._rgen,
            trials=self.trials,
            max_evals=self.max_evals,
        )
        return self

    def objective(self, hparams):
        """Objective function to minimize.

        Args:
            hparams (dict): Hyperparameter configuration.

        Returns:
            (dict): Outputs stored in the hyperopt trials object.

        """
        if self.early_stopping < 1:
            warnings.warn('Exiting by early stopping.')
            return self._best_params

        test_loss, train_loss = [], []
        folds = StratifiedKFold(self.cv, self.shuffle, self.random_state)
        for train_idx, test_idx in folds.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            _model = deepcopy(self.model)
            _model.set_params(**hparams)
            _model.fit(X_train, y_train)

            pred_y_test = np.squeeze(_model.predict(X_test))
            pred_y_train = np.squeeze(_model.predict(X_train))

            test_loss.append(1.0 - self.score_func(y_test, pred_y_test))
            train_loss.append(1.0 - self.score_func(y_train, pred_y_train))

        # NOTE: Consider using median rather than mean.
        self._best_params = OrderedDict(
            [
                ('status', STATUS_OK),
                ('eval_time', time.time()),
                ('loss', np.nanmean(test_loss)),
                ('loss_variance', np.nanvar(test_loss)),
                ('train_loss', np.nanmean(train_loss)),
                ('train_loss_variance', np.nanvar(train_loss)),
                ('hparams', hparams)
            ]
        )
        # Record the minimum loss to monitor if diverging from optimum.
        if self._prev_score < self._best_params['loss']:
            self.early_stopping = self.early_stopping - 1
            #warnings.warn('Reduced buffer for eacly stopping to {}'
            #              ''.format(self.early_stopping))
        else:
            self._prev_score = self._best_params['loss']

        return self._best_params

    @staticmethod
    def _check_X_y(X, y):
        # Wrapping the sklearn formatter function.
        return check_X_y(X, y)
