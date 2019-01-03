# hyperopt with sklearn
# http://steventhornton.ca/hyperparameter-tuning-with-hyperopt-in-python/

# Parallelizing Evaluations During Search via MongoDB
# https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB

# Practical notes on SGD: https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use

import time
import logging
import numpy as np

from datetime import datetime
from collections import OrderedDict

from scipy import stats

from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
from hyperopt import space_eval
from hyperopt import STATUS_OK

from hyperopt.pyll.base import scope

from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold


def model_selection():

    pass


class BBCCV:
    """

    Args:
        score_func (function):
        n_iter (int):
        random_state (int):

    """

    def __init__(
        self,
        score_func,
        n_iter=5,
        alpha=0.05,
        random_state=None,
    ):

        self.score_func = score_func
        self.n_iter = n_iter
        self.alpha = alpha
        self.random_state = random_state

        self._sampler = None

    def fit(self, Y_pred, Y_true):
        """bootstrap_bias_corrected_cv

        Args:
            Y_pred (array-like): A matrix (N x C) containing out-of-sample
                predictions for N samples and C hyperparameter configurations.
                Thus, scores[i, j] denotes the out-of-sample prediction of on
                the i-th sample of the j-th configuration.
            Y_true ():

        Returns:
            (dict):


        """

        # Bootstrapped matrices.
        if self._sampler is None:
            self._sampler = OOBSampler(self.n_iter, self.random_state)

        bbc_scores = []
        for sample_idx, oos_idx in sampler.split(Y_true, Y_pred):
            best_config = criterion(
                Y_true[sample_idx, :], Y_pred[sample_idx, :], self.score_func
            )
            bbc_scores.append(
                self.score_func(
                    Y_true[oos_idx, best_config], Y_pred[oos_idx, best_config]
                )
            )
        return {
            'avg_score': np.mean(bbc_scores),
            'std_score': np.std(bbc_scores),
            'median_score': np.median(bbc_scores),
            'avg_ci': self.mean_ci(bbc_scores),
            'bootstrap_ci': self.bootstrap_ci(bbc_scores),
        }

    def criterion(self, Y_true, Y_pred):
        """

        Returns:
            (int): Index of the optimal configuration according to the
                score function.

        """

        _, nconfigs = np.shape(Y_true)

        losses = np.ones(nconfigs, dtype=float) * np.nan
        for num in range(nconfigs):
            losses[num] = 1 - self.score_func(Y_true[:, num], Y_pred[:, num])

        # Select the configuration with the minimum loss.
        return np.argmin(losses)

    # QUESTION: How many degrees of freedom in standard error?
    def mean_ci(self, samples):
        """Calculate the mean confidence interval from sample data."""

        # The standard error of the mean.
        mean, mean_se  = np.mean(samples), stats.sem(samples)

        # Percent point function (inverse of cdf — percentiles).
        deviation = mean_se * stats.t.ppf(1 - self.alpha / 2, len(samples) - 1)

        return mean, mean - deviation, mean + deviation

    def bootstrap_ci(self, scores):
        """Calculate the bootstrap confidence interval from sample data."""

        asc_scores = sorted(scores)

        upper_idx = (1 - self.alpha / 2) * len(scores)
        lower_idx = self.alpha / 2 * len(scores)

        return asc_scores[int(lower_idx)], asc_scores[int(upper_idx)]


class OOBSampler:
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
            # Oberervations not part of training set defines test set.
            test_idx = np.array(
                list(set(sample_indicators) - set(train_idx)), dtype=int
            )
            yield train_idx, test_idx


class ParameterSearchCV:
    """Represetation of a K-fold cross-validated hyperparameter search.

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
        n_splits=10,
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

        self.n_splits = int(n_splits)
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

        # Run the hyperparameter search.
        self._best_params = fmin(
            self.objective,
            self.space,
            algo=self.algo,
            max_evals=self.max_evals,
            trials=self.trials
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

        kfolds = StratifiedKFold(
            self.n_splits, self.shuffle, self.random_state
        )
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

            # Collect ground truths and predictions to BBC-CV procedure.
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
            ]
        )

    # TODO:
    @staticmethod
    def _check_X_y(X, y):
        # Type checking of predictor and target data.

        return X, y



if __name__ == '__main__':
    # TODO:
    # * Create BBC-CV class (tensorflow GPU/numpy-based)
    # * Write work function sewing together param search and BBC-CV class that can
    #   be passed to model_comparison.
    # * Calc sample sizes in each fold with 10-fold CV and 198 patients.

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
        X, y, test_size=0.2, random_state=42
    )
    pipe = Pipeline([
        ('kbest', SelectPercentile(chi2)),
        ('clf_scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=0))
    ])

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

    optimizer = ParameterSearchCV(
        tpe.suggest,pipe, space, score_func=roc_auc_score, random_state=0
    )
    optimizer.fit(X_train, y_train)

    Y_true, Y_pred = optimizer.oos_pairs

    #correction = BBCCV(random_state=0)
    #correction.loss(searcher.predictions, y_train)
