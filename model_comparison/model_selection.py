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

import numpy as np

from copy import deepcopy
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
    oob,
    max_evals,
    shuffle,
    verbose=1,
    random_state=None,
    alpha=0.05,
    balancing=True,
    path_tmp_results=None,
    error_score=np.nan,
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
        # Experimental results container.
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
            oob,
            max_evals,
            shuffle,
            verbose=1,
            random_state=None,
            alpha=0.05,
            balancing=True,
            path_tmp_results=None,
            error_score=np.nan,
        )
        if path_tmp_results is not None:
            print('Writing results...')
            utils.ioutil.write_prelim_results(path_case_file, output)

        if verbose > 0:
            durat = datetime.now() - start_time
            print('Experiment {} completed in {}'.format(random_state, durat))
            output['exp_duration'] = durat

        return output


def nested_kfold(
    X, y,
    algo,
    model_id,
    model,
    param_space,
    score_func,
    cv,
    oob,
    max_evals,
    shuffle,
    verbose=1,
    random_state=None,
    alpha=0.05,
    balancing=True,
    path_tmp_results=None,
    error_score=np.nan,
):

    test_loss, train_loss, Y_test, Y_pred = [], [], [], []
    _cv = StratifiedKFold(self.cv, self.shuffle, self.random_state)

    start_time = datetime.now()
    for num, (train_idx, test_idx) in enumerate(_cv.split(self.X, self.y)):

        X_train, X_test = self.X[train_idx], self.X[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]

        # Perform cross-validated hyperparameter optimization.
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

        _model = optimizer.best_model
        _model.train(X_test)

        pred_y_test = np.squeeze(_model.predict(X_test))
        pred_y_train = np.squeeze(_model.predict(X_train))

        test_loss.append(1.0 - self.score_func(y_test, pred_y_test))
        train_loss.append(1.0 - self.score_func(y_train, pred_y_train))

    return OrderedDict(
        [
            ('loss', np.nanmedian(test_loss)),
            ('loss_variance', np.nanvar(test_loss)),
            ('train_loss', np.nanmedian(train_loss)),
            ('train_loss_variance', np.nanvar(train_loss)),
        ]
    )


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
        oob=10,
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
        for train_idx, test_idx in self._sampler.split(Y_true, Y_pred):
            # Sample rows with replacement from the prediction matrix.
            opt_config = self.criterion(
                Y_true[train_idx, :], Y_pred[train_idx, :]
            )
            bbc_scores.append(
                self._score(
                    Y_true[test_idx, opt_config], Y_pred[test_idx, opt_config]
                )
            )
        return {
            'oob_avg_score': np.nanmean(bbc_scores),
            'oob_std_score': np.nanstd(bbc_scores),
            'oob_median_score': np.nanmedian(bbc_scores),
            'oob_score_ci': self.bootstrap_ci(bbc_scores),
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


class ParameterSearchCV:
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
        balancing=True
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

        # NOTE: Attributes updated with instance.
        self.X = None
        self.y = None
        self.trials = None

        self._rgen = None
        self._best_params = None

    @property
    def params(self):

        params = {
            num: res['hparams'] for num, res in enumerate(self.trials.results)
        }
        return {'param_search_eval_params': params}

    @property
    def best_model(self):
        """Returns an instance of the estimator with the optimal
        hyperparameters."""

        return self.model.set_params(**self.best_params)

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

        if self.Y_pred is None and self.Y_test is None:
            self._setup_pred_containers()
        # The Trials object stores information of each iteration.
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
        if self.verbose > 1:
            self._num_evals = self._num_evals + 1
            print('Evaluating objective at round {}'.format(self._num_evals))

        test_loss, train_loss, Y_test, Y_pred = [], [], [], []
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

        return OrderedDict(
            [
                ('status', STATUS_OK),
                ('eval_time': time.time()),
                ('loss', np.nanmedian(test_loss)),
                ('loss_variance', np.nanvar(test_loss)),
                ('train_loss', np.nanmedian(train_loss)),
                ('train_loss_variance', np.nanvar(train_loss)),
            ]
        )

    @staticmethod
    def _check_X_y(X, y):
        # Wrapping the sklearn formatter function.
        return check_X_y(X, y)

    @staticmethod
    def safe_predict(y_pred):

        if np.ndim(y_pred) > 1:
            y_pred = np.squeeze(y_pred)

        return y_pred


if __name__ == '__main__':

    import sys
    sys.path.append('./../experiment')

    import os
    import backend
    import model_selection
    import comparison

    import numpy as np
    import pandas as pd

    from configs.selector_configs import selectors
    from configs.estimator_configs import classifiers

    from hyperopt import tpe

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import precision_recall_fscore_support

    # TEMP:
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    # TODO: To utils?
    def load_target(path_to_target, index_col=0):

        var = pd.read_csv(path_to_target, index_col=index_col)
        return np.squeeze(var.values).astype(np.float32)


    # TODO: To utils?
    def load_predictors(path_to_data, index_col=0, regex=None):

        data = pd.read_csv(path_to_data, index_col=index_col)
        if regex is None:
            return np.array(data.values, dtype=np.float32)
        else:
            target_features = data.filter(regex=regex)
            return np.array(data.loc[:, target_features].values, dtype=np.float32)

    # FEATURE SET:
    X = load_predictors('./../../data_source/to_analysis/complete_decorr.csv')

    # TARGET:
    y = load_target('./../../data_source/to_analysis/target_lrr.csv')

    # SETUP:
    CV = 3
    OOB = 10
    MAX_EVALS = 7
    SCORING = roc_auc_score

    # Generate pipelines from config elements.
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    name = 'MRMRSelection_SVC'
    pipe, params = pipes_and_params[name]

    results = bbc_cv_selection(
        X, y,
        tpe.suggest,
        name,
        pipe,
        params,
        SCORING,
        CV,
        OOB,
        MAX_EVALS,
        shuffle=True,
        verbose=0,
        random_state=0,
        alpha=0.05,
        balancing=True,
        error_score=np.nan,
        path_tmp_results=None,
    )
