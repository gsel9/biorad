# hyperopt with sklearn
# http://steventhornton.ca/hyperparameter-tuning-with-hyperopt-in-python/

# Parallelizing Evaluations During Search via MongoDB
# https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB

# Practical notes on SGD: https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use

import logging
import numpy as np

from datetime import datetime

from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
from hyperopt import space_eval

from hyperopt.pyll.base import scope


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


class BBCCV:
    """

    Args:
        score_func (function):
        n_iter (int):
        random_state (int):

    """

    def __init__(self, n_iter=5, random_state=None):

        self.sampler = OOBSampler(n_iter, random_state)

    def loss(self, scores):
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
        nrows, _ = np.shape(scores)

        loss = 0
        for sample_idxs, oob_idxs in self.sampler.split(scores):
            # Apply configuration selection method to OOB scores.
            idx = int(self.criterion(scores[sample_idxs, :]))
            #loss = loss + scores[oob_idxs, idx]
            #print(scores[oob_idxs, idx])

        return loss / self.n_iter

    @staticmethod
    def criterion(scores):

        # NOTE: Maximizing score instead of minimizing error as originally in
        # paper.
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
        nrows, _ = np.shape(X)
        sample_idxs = np.arange(nrows, dtype=int)
        for _ in range(self.n_splits):
            train_idx = self.rgen.choice(
                sample_idxs, size=nrows, replace=True
            )
            test_idx = np.array(
                list(set(sample_idxs) - set(train_idx)), dtype=int
            )
            yield train_idx, test_idx


class ParameterSearch:

    def __init__(
        self,
        model, space,
        cv=5,
        score_func=None,
        n_jobs=-1,
        max_evals=10,
        algo=tpe.suggest
    ):
        self.model = model
        self.space = space

        self.cv = cv
        self.algo = algo
        self.n_jobs = n_jobs
        self.max_evals = max_evals
        self.score_func = score_func

        self.X = None
        self.y = None
        self.trails = None
        self.results = None
        self.run_time = None

        self._errors = None
        self._y_preds = None

    @property
    def opt_hparams(self):
        # Get the values of the optimal parameters

        return space_eval(self.space, self.results)

    @property
    def opt_model(self):

        return self.model.set_params(**self.opt_hparams)

    @property
    def errors(self):

        return np.array(self._errors, dtype=float)

    @property
    def predictions(self):

        return np.array(self._y_preds, dtype=float).T

    def fit(self, X, y):
        """Perform hyperparameter search.

        Args:
            X (array-like):
            y (array-like):

        """

        self.X, self.y = self._check_X_y(X, y)

        # The Trials object will store details of each iteration.
        if self.trails is None:
            self.trials = Trials()

        if self._y_preds is None:
            self._y_preds = []

        if self._errors is None:
            self._errors = []

        # Run the hyperparameter search.
        start_time = datetime.now()
        self.results = fmin(
            self.objective,
            self.space,
            algo=self.algo,
            max_evals=self.max_evals,
            trials=self.trials
        )
        self.run_time = datetime.now() - start_time

        return self

    def objective(self, hparams):
        """Objective function to minimize.

        Args:
            hparams (dict): Model hyperparameter configuration.

        Returns:
            (float): Error score.

        """
        # NOTE: Assumes standard model API.
        self.model.set_params(**hparams)

        # Stratified K-fold cross-validation.
        y_preds = cross_val_predict(
            self.model,
            self.X, self.y,
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        error = 1.0 - np.median(self.score_func(self.y, y_preds))
        # Save scores for BBC-CV procedure, and errors for inspection.
        self._y_preds.append(y_preds), self._errors.append(error)

        return error

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

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    # TEMP:
    from sklearn.feature_selection.univariate_selection import SelectPercentile, chi2
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    kbest = SelectPercentile(chi2)
    clf = RandomForestClassifier(random_state=0)
    pipe = Pipeline([('kbest', kbest), ('clf', clf)])


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

    searcher = ParameterSearch(pipe, space, score_func=roc_auc_score)
    searcher.fit(X_train, y_train)
    print(searcher.predictions)
    print(searcher.errors)
    #correction = BBCCV(random_state=0)
    #correction.loss(searcher.scores)
