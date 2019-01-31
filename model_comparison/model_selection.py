# -*- coding: utf-8 -*-
#
# model_selection.py
#

"""
hyperopt with sklearn
http://steventhornton.ca/hyperparameter-tuning-with-hyperopt-in-python/

Parallelizing Evaluations During Search via MongoDB
https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB

Practical notes on SGD: https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use

Checkout for plots ++:
* https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
* https://github.com/tmadl/highdimensional-decision-boundary-plot

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


def nested_kfold_selection(
    X, y,
    algo,
    model_id,
    model,
    param_space,
    score_func,
    cv,
    max_evals,
    shuffle,
    verbose=1,
    random_state=None,
    balancing=True,
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
                random_state, model_id
            )
        )
    else:
        path_case_file = ''

    # Determine if results already exists.
    if os.path.isfile(path_case_file):
        output = utils.ioutil.read_prelim_result(path_case_file)
        print('Reloading results from: {}'.format(path_case_file))
    else:
        output = {'exp_id': random_state, 'model_id': model_id}
        if verbose > 0:
            print('Running experiment: {}'.format(random_state))
            start_time = datetime.now()

        results = nested_kfold(
            X, y,
            algo,
            model_id,
            model,
            param_space,
            score_func,
            cv,
            max_evals,
            shuffle,
            verbose=1,
            random_state=None,
            balancing=True,
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
    algo,
    model_id,
    model,
    param_space,
    score_func,
    cv,
    max_evals,
    shuffle,
    scaling=True,
    verbose=1,
    random_state=None,
    balancing=True,
    path_tmp_results=None,
    error_score=np.nan,
):

    """
    test_loss, train_loss, Y_test, Y_pred = [], [], [], []
    _cv = StratifiedKFold(cv, shuffle, random_state)
    for num, (train_idx, test_idx) in enumerate(_cv.split(X, y)):

        if verbose > 0:
            # Adjusting to Python counting logic.
            print('Outer loop iteration number {}'.format(num + 1))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start_time = datetime.now()
        optimizer = ParameterSearchCV(
            algo=algo,
            model=model,
            space=param_space,
            score_func=score_func,
            cv=cv,
            verbose=verbose,
            max_evals=max_evals,
            shuffle=shuffle,
            random_state=random_state,
            error_score=error_score,
            balancing=balancing
        )
        optimizer.fit(X_train, y_train)

        if verbose > 0:
            print('Parameter search finished in {}'
                  ''.format(datetime.now() - start_time))

        _model = optimizer.best_model
        _model.fit(X_test)

        pred_y_test = np.squeeze(_model.predict(X_test))
        pred_y_train = np.squeeze(_model.predict(X_train))

        test_loss.append(1.0 - self.score_func(y_test, pred_y_test))
        train_loss.append(1.0 - self.score_func(y_train, pred_y_train))
    """
    if verbose > 0:
        start_search = datetime.now()

    optimizer = BayesianSearchCV(
        algo=algo,
        model=model,
        space=param_space,
        score_func=score_func,
        cv=cv,
        verbose=verbose,
        max_evals=max_evals,
        shuffle=shuffle,
        random_state=random_state,
        error_score=error_score,
        balancing=balancing
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

        # Setup best model with hyperparameters from parameter search.
        _model = deepcopy(optimizer.best_model)
        _model.fit(X_train, y_train)

        # Tracking scores for each fold.
        test_scores.append(
            score_func(y_test, np.squeeze(_model.predict(X_test)))
        )
        train_scores.append(
            score_func(y_train, np.squeeze(_model.predict(X_train)))
        )
    if verbose > 0:
        print('Score CV finished in {}'.format(datetime.now() - end_search))

    return OrderedDict(
        [
            ('test_score', np.nanmedian(test_scores)),
            ('train_score', np.nanmedian(train_scores)),
            ('test_score_variance', np.nanvar(test_scores)),
            ('train_score_variance', np.nanvar(train_scores)),
        ]
    )


class BayesianSearchCV:
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
        cv=10,
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
        _model.set_params(**self.best_results['hparams'])

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

        self._best_params = OrderedDict(
            [
                ('status', STATUS_OK),
                ('eval_time', time.time()),
                ('loss', np.nanmedian(test_loss)),
                ('loss_variance', np.nanvar(test_loss)),
                ('train_loss', np.nanmedian(train_loss)),
                ('train_loss_variance', np.nanvar(train_loss)),
                ('hparams', hparams)
            ]
        )
        # Record the minimum loss to monitor if diverging from optimum.
        if self._prev_score < self._best_params['loss']:
            self.early_stopping = self.early_stopping - 1
            warnings.warn('Reduced buffer for eacly stopping to {}'
                          ''.format(self.early_stopping))
        else:
            print(self._best_params['loss'])
            self._prev_score = self._best_params['loss']

        return self._best_params

    @staticmethod
    def _check_X_y(X, y):
        # Wrapping the sklearn formatter function.
        return check_X_y(X, y)

    @staticmethod
    def safe_predict(y_pred):

        if np.ndim(y_pred) > 1:
            y_pred = np.squeeze(y_pred)

        return y_pred
