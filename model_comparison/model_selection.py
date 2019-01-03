# hyperopt with sklearn
# http://steventhornton.ca/hyperparameter-tuning-with-hyperopt-in-python/

# Parallelizing Evaluations During Search via MongoDB
# https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB

# Practical notes on SGD: https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use

# Checkout for plots ++: https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce

import os
import time
import utils
import pickle
import logging

import numpy as np

from scipy import stats

from datetime import datetime
from collections import OrderedDict

from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
from hyperopt import space_eval
from hyperopt import STATUS_OK

from hyperopt.pyll.base import scope

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def model_selection(
    X, y,
    algo,
    model,
    space,
    score_func,
    path_tmp_results,
    cv,
    oob,
    max_evals,
    shuffle,
    verbose=0,
    random_state=None,
    alpha=0.05,
    balancing=True,
    error_score=np.nan,
):
    """
    Work function for parallelizable model selection experiments.

    Args:

        cv (int): The number of folds in stratified k-fold cross-validation.
        oob (int): The number of samples in out-of-bag bootstrap re-sampling.
        max_evals (int): The number of iterations in hyperparameter search.

    """
    path_case_file = os.path.join(
        path_tmp_results, 'experiment_{}'.format(random_state)
    )
    # Determine if results already produced, or if initiating new experiment.
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        print('Reloading results from: {}'.format(path_case_file))
    else:
        # Balance target class distributions with SMOTE procedure.
        if balancing:
            X, y = utils.sampling.balance_data(X, y, random_state)

        # Experimental results container.
        outputs = {'exp_id': random_state}

        if verbose > 0:
            print('Initiating experiment: {}'.format(random_state))
            start_time = datetime.now()

        # Perform cross-validated hyperparameter optimization.
        optimizer = ParameterSearchCV(
            algo=algo,
            model=model,
            space=space,
            score_func=score_func,
            cv=cv,
            max_evals=max_evals,
            shuffle=shuffle,
            random_state=random_state,
            error_score=error_score,
        )
        optimizer.fit(X, y)

        # Include results.
        #outputs.update(optimizer.trials.results)

        # Evaluate model performance with BBC-CV method.
        bbc_cv = BootstrapBiasCorrectedCV(
            random_state=random_state,
            score_func=score_func,
            alpha=alpha,
            oob=oob,
        )
        # Returns results directly.
        outputs.update(bbc_cv.evaluate(*optimizer.oos_pairs))

        if verbose > 0:
            durat = datetime.now() - start_time
            outputs['exp_duration'] = durat
            print('Experiment {} completed in {}'.format(random_state, durat))

    return outputs


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
        self.random_state = random_state
        self.score_func = score_func
        self.error_score = error_score
        self.oob = oob
        self.alpha = alpha

        self._sampler = None

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
        # Generate bootstrapped matrices.
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
            'avg_score': np.mean(bbc_scores),
            'std_score': np.std(bbc_scores),
            'median_score': np.median(bbc_scores),
            'bootstrap_ci': self.bootstrap_ci(bbc_scores),
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

        asc_scores = sorted(scores)

        upper_idx = (1 - self.alpha / 2) * len(scores)
        lower_idx = self.alpha / 2 * len(scores)

        return asc_scores[int(lower_idx)], asc_scores[int(upper_idx)]


class ParameterSearchCV:
    """Perform K-fold cross-validated hyperparameter search with the Bayesian
    optimization Tree Parzen Estimator.

    Args:
        model ():
        space ():
        ...

    """

    # NOTE: For pickling.
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
    ):
        self.algo = algo
        self.model = model
        self.space = space
        self.shuffle = shuffle
        self.score_func = score_func
        self.error_score = error_score

        self.cv = int(cv)
        self.max_evals = int(max_evals)
        self.random_state = int(random_state)

        self.X = None
        self.y = None
        self.trials = None
        self._best_params = None

    @property
    def best_params(self):
        """Returns the optimal hyperparameters."""

        return self._best_params

    @property
    def best_model(self):
        """Returns an instance of the estimator with the optimal
        hyperparameters."""

        return self.model.set_params(**self.best_params)

    @property
    def train_loss(self):
        """Returns """

        test_losses = [results['train_loss'] for results in self.trials.results]
        return np.array(test_losses, dtype=float)

    @property
    def test_loss(self):
        """Returns """

        test_losses = [results['loss'] for results in self.trials.results]
        return np.array(test_losses, dtype=float)

    @property
    def train_loss_var(self):
        """Returns the variance of each hyperparameter configuration of K-fold
        cross-validated training loss."""

        test_losses = [
            results['train_loss_variance'] for results in self.trials.results
        ]
        return np.array(test_losses, dtype=float)

    @property
    def test_loss_var(self):
        """Returns the variance of each hyperparameter configuration of K-fold
        cross-validated test loss."""

        test_losses = [
            results['loss_variance'] for results in self.trials.results
        ]
        return np.array(test_losses, dtype=float)

    @property
    def oos_pairs(self):
        """Returns a tuple with ground truths and corresponding out-of-sample
        predictions."""

        preds = np.transpose(
            [items['y_preds'] for items in self.trials.results]
        )
        trues = np.transpose(
            [items['y_trues'] for items in self.trials.results]
        )
        return trues, preds

    def fit(self, X, y):
        """Optimal hyperparameter search.

        Args:
            X (array-like):
            y (array-like):

        """
        self.X, self.y = self._check_X_y(X, y)

        # The Trials object stores information of each iteration.
        if self.trials is None:
            self.trials = Trials()

        # For saving prelim results: https://github.com/hyperopt/hyperopt/issues/267
        #pickle.dump(optimizer, open(TEMP_RESULTS_FILE, 'wb'))
        #trials = pickle.load(open('TEMP_RESULTS_FILE', 'rb'))

        # Run the hyperparameter search.
        self._best_params = fmin(
            self.objective,
            self.space,
            algo=self.algo,
            max_evals=self.max_evals,
            trials=self.trials,
            rstate=np.random.RandomState(self.random_state)
        )
        return self

    def objective(self, hparams):
        """Objective function to minimize.

        Args:
            hparams (dict): Hyperparameter configuration.

        Returns:
            (dict): Outputs stored in the hyperopt trials object.

        """
        start_time = datetime.now()

        kfolds = StratifiedKFold(self.cv, self.shuffle, self.random_state)

        test_loss, train_loss, _preds, _trues = [], [], [], []
        for train_index, test_index in kfolds.split(self.X, self.y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Clone model to ensure independency.
            _model = clone(self.model)
            # Error handling mechanism.
            try:
                _model.set_params(**hparams)
                _model.fit(X_train, y_train)
            except:
                # Use error score
                pass

            _y_test = _model.predict(X_test)
            _y_train = _model.predict(X_train)

            test_loss.append(1.0 - self.score_func(y_test, _y_test))
            train_loss.append(1.0 - self.score_func(y_train, _y_train))

            # Collect ground truths and predictions for BBC-CV procedure.
            _trues = np.hstack((_trues, y_test))
            _preds = np.hstack((_preds, _y_test))

        return OrderedDict(
            [
                ('status', STATUS_OK),
                ('eval_time', datetime.now() - start_time),
                ('loss', np.median(test_loss)),
                ('train_loss', np.median(train_loss)),
                ('loss_variance', np.var(test_loss)),
                ('train_loss_variance', np.var(train_loss)),
                ('y_trues', _trues,),
                ('y_preds', _preds,),
                ('hparams', hparams)
            ]
        )

    # TODO:
    @staticmethod
    def _check_X_y(X, y):
        # Type checking of predictor and target data.

        return X, y


if __name__ == '__main__':
    # TODO:
    # * Write work function sewing together param search and BBC-CV class that can
    #   be passed to model_comparison.


    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # TEMP:
    from sklearn.feature_selection.univariate_selection import SelectPercentile, chi2
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    pipe = Pipeline([
        ('kbest', SelectPercentile(chi2)),
        ('clf_scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=0))
    ])

    # Can specify hparam distr in config files that acn direclty be read into
    # Python dict with hyperopt distirbutions?

    # Parameter search space
    space = {}
    # Random number between 50 and 100
    space['kbest__percentile'] = hp.uniform('kbest__percentile', 50, 100)
    # Random number between 0 and 1
    #space['clf__l1_ratio'] = hp.uniform('clf__l1_ratio', 0.0, 1.0)
    # Log-uniform between 1e-9 and 1e-4
    #space['clf__alpha'] = hp.loguniform('clf__alpha', -9*np.log(10), -4*np.log(10))
    # Random integer in 20:5:80
    #space['clf__n_iter'] = 20 + 5 * hp.randint('clf__n_iter', 12)
    # Random number between 50 and 100
    space['clf__class_weight'] = hp.choice('clf__class_weight', [None,]) #'balanced']),
    # Discrete uniform distribution
    space['clf__max_leaf_nodes'] = scope.int(hp.quniform('clf__max_leaf_nodes', 30, 150, 1))
    # Discrete uniform distribution
    space['clf__min_samples_leaf'] = scope.int(hp.quniform('clf__min_samples_leaf', 20, 500, 5))

    results = model_selection(
        X_train, y_train,
        algo=tpe.suggest,
        model=pipe,
        space=space,
        score_func=roc_auc_score,
        path_tmp_results='./',
        cv=5,
        oob=5,
        max_evals=7,
        shuffle=True,
        verbose=1,
        random_state=0,
        alpha=0.05,
        balancing=True,
        error_score=np.nan
    )
