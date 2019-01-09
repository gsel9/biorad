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

from scipy import stats
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


# TODO:
def point_632plus_selection():
    pass


def bbc_cv_selection(
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
        output = {'exp_id': random_state}

        if verbose > 0:
            print('Initiating parameter search: {}'.format(random_state))
            start_time = datetime.now()

        # Perform cross-validated hyperparameter optimization.
        optimizer = ParameterSearchCV(
            algo=algo,
            model=model,
            space=param_space,
            score_func=score_func,
            cv=cv,
            max_evals=max_evals,
            shuffle=shuffle,
            random_state=random_state,
            error_score=error_score,
            balancing=balancing
        )
        # Error handling
        optimizer.fit(X, y)

        if verbose > 0:
            _duration = datetime.now() - start_time
            print('Finished parameter search in: {}'.format(_duration))
            print('Initiating BBC-CV')

        # Evaluate model performance with BBC-CV method.
        bbc_cv = BootstrapBiasCorrectedCV(
            random_state=random_state,
            score_func=score_func,
            alpha=alpha,
            oob=oob,
        )
        # Add results to output.
        output.update(bbc_cv.evaluate(*optimizer.oos_pairs))
        output.update(optimizer.test_loss)
        output.update(optimizer.train_loss)
        output.update(optimizer.test_loss_var)
        output.update(optimizer.train_loss_var)
        output.update(optimizer.best_params)
        output.update(optimizer.params)

        if verbose > 0:
            print('Finished BBC-CV in {}'.format(datetime.now() - _duration))

        if path_tmp_results is not None:
            print('Writing results...')
            utils.ioutil.write_prelim_results(path_case_file, output)

        if verbose > 0:
            duration = datetime.now() - start_time
            print('Experiment completed in {}'.format(duration))
            output['exp_duration'] = duration

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
        for sample_idx, oos_idx in self._sampler.split(Y_true, Y_pred):
            best_config = self.criterion(
                Y_true[sample_idx, :], Y_pred[sample_idx, :]
            )
            bbc_scores.append(
                self._score(
                    Y_true[oos_idx, best_config], Y_pred[oos_idx, best_config]
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
        """

        Returns:
            (int): Index of the optimal configuration according to the
                score function.

        """
        _, num_configs = np.shape(Y_true)

        losses = np.ones(num_configs, dtype=float) * np.nan
        for num in range(num_configs):
            # Returns <float> or error score (1 - NaN = NaN).
            losses[num] = 1.0 - self._score(Y_true[:, num], Y_pred[:, num])

        # Select the configuration with the minimum loss ignoring NaNs.
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

    # NOTE: For pickling results stored in hyperopt Trials() object.
    TEMP_RESULTS_FILE = './tmp_trials.p'

    def __init__(
        self,
        algo,
        model,
        space,
        score_func,
        cv=5,
        max_evals=10,
        shuffle=True,
        random_state=None,
        error_score=np.nan,
        balancing=True
    ):
        self.algo = algo
        self.model = model
        self.space = space
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
        # Ground truths and predictions for BBC-CV procedure.
        self.Y_pred = None
        self.Y_test = None

        # The hyperopt trails object stores information from each iteration.
        self.trials = None
        self._rgen = None
        self._sample_lim = None
        self._best_params = None

    @property
    def best_params(self):
        """Returns the optimal hyperparameters."""

        return {'param_search_best_params': self._best_params}

    @property
    def params(self):

        params = {
            num: res['hparams'] for num, res in enumerate(self.trials.results)
        }
        return {'param_search_params': params}

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
        return {'param_search_trainig_loss': losses}

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
        # For saving prelim results: https://github.com/hyperopt/hyperopt/issues/267
        # pickle.dump(optimizer, open(TEMP_RESULTS_FILE, 'wb'))
        # trials = pickle.load(open('TEMP_RESULTS_FILE', 'rb'))
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

    def objective(self, hparams):
        """Objective function to minimize.

        Args:
            hparams (dict): Hyperparameter configuration.

        Returns:
            (dict): Outputs stored in the hyperopt trials object.

        """
        start_time = datetime.now()

        _cv = StratifiedKFold(self.cv, self.shuffle, self.random_state)

        test_loss, train_loss = [], []
        Y_test, Y_pred = [], []
        for num, (train_idx, test_idx) in enumerate(_cv.split(self.X, self.y)):

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            # Balance target class distributions with SMOTE oversampling.
            if self.balancing:
                X_train, y_train = utils.sampling.balance_data(
                    X_train, y_train, self.random_state
                )
            # Clone model to ensure independency between folds. With deepcopy,
            # changes made to the object copy does not affect the original
            # object version.
            _model = deepcopy(self.model)
            # Configure model with provided hyperparamter setting and train.
            _model.set_params(**hparams)
            _model.fit(X_train, y_train)
            # Ensure predictions are properly formatted vectors.
            pred_y_test = self.safe_predict(_model.predict(X_test))
            pred_y_train = self.safe_predict(_model.predict(X_train))

            test_loss.append(1.0 - self.score_func(y_test, pred_y_test))
            train_loss.append(1.0 - self.score_func(y_train, pred_y_train))
            # Collect ground truths and predictions for BBC-CV procedure.
            Y_pred.append(pred_y_test[:self._sample_lim])
            Y_test.append(y_test[:self._sample_lim])

        return OrderedDict(
            [
                ('status', STATUS_OK),
                ('eval_time', datetime.now() - start_time),
                ('loss', np.median(test_loss)),
                ('train_loss', np.median(train_loss)),
                ('loss_variance', np.var(test_loss)),
                ('train_loss_variance', np.var(train_loss)),
                ('hparams', hparams),
                # Stack predictions of each fold into a vector representing the
                # predictions for this particular configuration.
                ('y_true', np.hstack(Y_test,)),
                ('y_pred', np.hstack(Y_pred,)),
            ]
        )

    @staticmethod
    def _check_X_y(X, y):
        # A wrapper around sklearn formatter.

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
