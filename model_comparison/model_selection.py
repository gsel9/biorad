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


def nested_selection(
    X, y,
    experiment_id,
    model,
    hparam_space,
    score_func,
    n_splits,
    selection_scheme,
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

        if selection_scheme == '.632+':
            results = nested_point632plus(
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
        elif selection_scheme == 'k-fold':
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
        else:
            raise ValueError('Invalid selection scheme {}'
                             ''.format(selection_scheme))

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
def nested_point632plus(
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
        performance_scheme='632+',
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

    sampler = OOBGenerator(
        n_splits=n_splits, random_state=random_state
    )
    train_scores, test_scores = [], []
    for train_idx, test_idx in sampler.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        _model = deepcopy(model)
        _model.set_params(**optimizer.best_config)
        _model.fit(X_train, y_train)

        y_train_pred = _model.predict(X_train)
        y_test_pred = _model.predict(X_test)
        train_score = score_func(y_train, y_train_pred)
        test_score = score_func(y_test, y_test_pred)
        test_scores.append(
            point632plus_score(
                y_train, y_train_pred, train_score, test_score
            )
        )
        train_scores.append(
            point632plus_score(
                y_test, y_test_pred, train_score, test_score
            )
        )
    if verbose > 0:
        print('Best model error eval finished in {}'
              ''.format(datetime.now() - end_search))

    return OrderedDict(
        [
            ('test_score', np.nanmedian(test_scores)),
            ('train_score', np.nanmedian(train_scores)),
            ('test_score_variance', np.nanvar(test_scores)),
            ('train_score_variance', np.nanvar(train_scores)),
            #('hparams', )
        ]
    )


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
    performance_scheme='k-fold',
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
        performance_scheme='k-fold',
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
            #('hparams', )
        ]
    )


class SMACSearch:

    def __init__(
        self,
        model=None,
        hparam_space=None,
        score_func=None,
        n_splits=None,
        performance_scheme=None,
        run_objective='quality',
        max_evals=None,
        random_state=None,
        shuffle=True,
        deterministic=True,
        output_dir=None,
        verbose=0,
        early_stopping=30
    ):
        self.model = model
        self.hparam_space = hparam_space
        self.score_func = score_func
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.verbose = verbose
        self.performance_scheme = performance_scheme
        # Optimization objective (quality or runtime).
        self.run_objective = run_objective
        # Maximum function evaluations.
        self.max_evals = max_evals
        # Configuration space.
        self.model = model
        self.output_dir = output_dir
        self.random_state = random_state
        self.early_stopping = early_stopping
        if deterministic:
            self.deterministic = 'true'
        else:
            self.deterministic = 'false'

        self._rgen = None
        self._best_config = None
        self._current_min = None
        self._objective_func = None

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
        # Setup objective function.
        if self.performance_scheme == '632+':
            self._objective_func = self.point632plus_objective
        elif self.performance_scheme == 'k-fold':
            self._objective_func = self.cv_objective
        else:
            raise ValueError('Invalid performance scheme {}'
                             ''.format(self.performance_scheme))

        # Setup scenario object and opimitze.
        scenario = Scenario(
            {
                'run_obj': self.run_objective,
                'runcount-limit': self.max_evals,
                'cs': self.hparam_space,
                'deterministic': self.deterministic,
                'output_dir': self.output_dir
             }
        )
        smac = SMAC(
            scenario=scenario,
            rng=self._rgen,
            tae_runner=self._objective_func
        )
        self._best_config = smac.optimize()

        return self

    def cv_objective(self, hparams):

        if self.early_stopping < 1:
            warnings.warn('Exiting by early stopping.')
            return self._best_params

        test_scores = []
        folds = StratifiedKFold(self.n_splits, self.shuffle, self.random_state)
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

    def point632plus_objective(self, hparams):

        if self.early_stopping < 1:
            warnings.warn('Exiting by early stopping.')
            return self._best_params

        sampler = OOBGenerator(
            n_splits=self.n_splits, random_state=self.random_state
        )
        test_scores = []
        for train_idx, test_idx in sampler.split(self.X, self.y):

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            _model = deepcopy(self.model)
            _model.set_params(**hparams)
            _model.fit(X_train, y_train)

            y_train_pred = _model.predict(X_train)
            y_test_pred = _model.predict(X_test)
            train_score = self.score_func(y_train, y_train_pred)
            test_score = self.score_func(y_test, y_test_pred)
            test_scores.append(
                point632plus_score(
                    y_test, y_test_pred, train_score, test_score
                )
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


class OOBGenerator:
    """A bootstrap Out-of-Bag resampler.

    Args:
        n_splits (int): The number of resamplings to perform.
        random_state (int): Seed for the pseudo-random number generator.

    """

    def __init__(self, n_splits, random_state):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, **kwargs):
        """Generates Out-of-Bag samples.

        Args:
            X (array-like): The predictor data.
            y (array-like): The target data.

        Returns:
            (generator): An iterable with X and y sample indicators.

        """
        rgen = np.random.RandomState(self.random_state)

        nrows, _ = np.shape(X)
        sample_indicators = np.arange(nrows)
        for _ in range(self.n_splits):
            train_idx = rgen.choice(
                sample_indicators, size=nrows, replace=True
            )
            test_idx = np.array(
                list(set(sample_indicators) - set(train_idx)), dtype=int
            )
            yield train_idx, test_idx


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
    r_marked = _relative_overfit_rate(train_score, test_score, gamma)

    return _point632plus(train_score, test_score, r_marked, test_score_marked)


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
