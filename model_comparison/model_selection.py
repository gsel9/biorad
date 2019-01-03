# hyperopt with sklearn
# http://steventhornton.ca/hyperparameter-tuning-with-hyperopt-in-python/

# Parallelizing Evaluations During Search via MongoDB
# https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB

# Practical notes on SGD: https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use

import time
import logging
import numpy as np

from datetime import datetime

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


class BBCCV:
    """

    Args:
        score_func (function):
        n_iter (int):
        random_state (int):

    """

    def __init__(self, score_func=None, n_iter=5, random_state=None):

        self.score_func = score_func
        self.n_iter = n_iter
        self.random_state = random_state

        self._sampler = OOBSampler(self.n_iter, self.random_state)

    def loss(self, y_pred, y_true):
        """bootstrap_bias_corrected_cv

        Args:
            scores (array-like): A matrix (N x C) containing out-of-sample
                predictions for N samples and C hyperparameter configurations.
                Thus, scores[i, j] denotes the out-of-sample prediction of on
                the i-th sample of the j-th configuration.

        Returns:
            (int): Index of the best performing configuration according to the
                loss criterion.


        """
        nrows = np.size(y_pred)

        loss = 0
        for sample_idxs, oob_idxs in self._sampler.split(y_pred):
            # Apply configuration selection method to OOB scores.
            loss = 2

        return loss / self.n_iter

    @staticmethod
    def criterion(scores):

        # Select the configuration with the maximum average score.
        return np.argmax(scores, axis=0)


class OOBSampler:
    """A bootstrap Out-of-Bag resampler.

    Args:
        n_splits (int): The number of resamplings to perform.
        random_state (int): Seed for the pseudo-random number generator.

    """

    def __init__(self, n_splits, random_state):

        self.n_splits = n_splits
        self.rgen = np.random.RandomState(random_state)

    def split(self, X, **kwargs):
        """Generates Out-of-Bag samples.

        Args:
            X (array-like): The predictor data.

        Returns:
            (genrator): An iterable with X and y sample indicators.

        """
        nrows = np.size(X)
        sample_idxs = np.arange(nrows, dtype=int)
        for _ in range(self.n_splits):
            train_idx = self.rgen.choice(
                sample_idxs, size=nrows, replace=True
            )
            test_idx = np.array(
                list(set(sample_idxs) - set(train_idx)), dtype=int
            )
            yield train_idx, test_idx


class ParameterSearchCV:

    def __init__(
        self,
        model, space,
        score_func=None,
        max_evals=10,
        n_splits=10,
        shuffle=True,
        cv=5,
        algo=tpe.suggest,
        error_score=np.nan,
        random_state=None
    ):
        self.model = model
        self.space = space

        self.shuffle = shuffle
        self.algo = algo
        self.error_score = error_score

        self.score_func = score_func
        self.n_splits = int(n_splits)
        self.max_evals = int(max_evals)
        self.cv = int(cv)
        self.random_state = int(random_state)

        self.X = None
        self.y = None
        self.trials = None
        self.results = None
        self.run_time = None

        # Keys: scores, preds, best_params, best_model
        self._outputs = None
        self._best_params = None
        self._preds = None
        self._loss = None

    @property
    def opt_params(self):
        # Get the values of the optimal parameters

        #return self._outputs['opt_params']
        pass

    @property
    def opt_model(self):

        #return self.model.set_params(**self.opt_params)
        pass

    @property
    def loss(self):

        #return np.transpose(np.array(self._scores, dtype=float))
        return self._loss

    @property
    def preds(self):
        """Returns out-of-sample predictions."""

        return np.transpose(self._preds)

    def fit(self, X, y):
        """Perform hyperparameter search.

        Args:
            X (array-like):
            y (array-like):

        """

        self.X, self.y = self._check_X_y(X, y)

        # The Trials object will store details of each iteration.
        if self.trials is None:
            self.trials = Trials()

        if self._preds is None:
            self._preds = []

        # Run the hyperparameter search.
        start_time = datetime.now()
        best_params = fmin(
            self.objective,
            self.space,
            algo=self.algo,
            max_evals=self.max_evals,
            trials=self.trials
        )
        self.run_time = datetime.now() - start_time

        print(self.trials.results)

        return self

    def objective(self, hparams):
        """Objective function to minimize.

        Args:
            hparams (dict): Model hyperparameter configuration.

        Returns:
            (float): Error score.

        """

        start_time = datetime.now()

        kfolds = StratifiedKFold(
            self.n_splits, self.shuffle, self.random_state
        )
        test_loss, train_loss = [], []
        for train_index, test_index in kfolds.split(self.X, self.y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Clone model to ensure independency.
            _model = clone(self.model)
            _model.set_params(**hparams)
            # TODO: Error handling
            try:
                _model.fit(X_train, y_train)
            except:
                # Use error score
                pass

            _y_test = _model.predict(X_test)
            _y_train = _model.predict(X_train)

            # Collect predictions to BBC-CV procedure.
            self._preds.append(_y_test)

            test_loss.append(1.0 - self.score_func(y_test, _y_test))
            train_loss.append(1.0 - self.score_func(y_train, _y_train))

        return {
            'status': STATUS_OK,
            'eval_time': datetime.now() - start_time,
            'loss': np.median(test_loss),
            'train_loss': np.median(train_loss),
            'loss_variance': np.var(test_loss),
            'train_loss_variance': np.var(train_loss),
        }

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

    searcher = ParameterSearchCV(
        pipe, space, score_func=roc_auc_score, random_state=0
    )
    searcher.fit(X_train, y_train)
    #correction = BBCCV(random_state=0)
    #correction.loss(searcher.predictions, y_train)
