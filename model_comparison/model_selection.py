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

from sklearn.model_selection import cross_val_score


class BBCCV:

    # TODO: Refactor to recieve scores and apply argmin to these (from objective function).
    def criterion(self):
        """Configuration selection strategy as described by Tsamardinos and
        Greasidou (2018).

        Args:
        results (array-like): A matrix (N x C) containing out-of-sample
            predictions for N sapmles and C hyperparameter configurations. Thus,
            results[i, j] denotes the out-of-sample prediction of on the i-th
            sample of the j-th configuration.
        y_true (array-like): Ground truths corresponding to results.
        loss (function): Score criterion.

        Returns:
            (int): Index of the best performing configuration according to the
                loss criterion.
        """
        # NOTE: can implement other selection criteria that consider, not only the
        # out-of-sample loss, but also the complexity of the models produced by
        # each configuration.

        # Compute scores for each column set of predictions.
        # NOTE: Passing ground truths as additional argument to loss function.
        scores = np.apply_along_axis(loss, axis=0, arr=results, y_true=y_true)
        # Select the configuration with the minimum average loss
        return np.argmin(scores)


class ParameterSearch:

    def __init__(
        self,
        model, space,
        cv=5, scoring='roc_auc',
        n_jobs=-1, max_evals=5, algo=tpe.suggest
    ):
        self.model = model
        self.space = space

        self.cv = cv
        self.algo = algo
        self.n_jobs = n_jobs
        self.max_evals = max_evals
        self.scoring = scoring

        self.X = None
        self.y = None
        self.trails = None
        self.scores = None
        self.results = None
        self.run_time = None

        self._runs = None

    @property
    def optimal_hparams(self):
        # Get the values of the optimal parameters

        return space_eval(self.space, self.results)

    @property
    def optimal_model(self):

        return self.model.set_params(self.optimal_hparams)

    def fit(self, X, y):

        self.X, self.y = self._check_X_y(X, y)

        # The Trials object will store details of each iteration
        if self.trails is None:
            self.trials = Trials()

        if self.scores is None:
            # TODO: Chak refactor to [np.size(y) x max_evals]
            self.scores = np.array((np.size(y), self.max_evals), dtype=float)

        # Reset num calls to objective = num hparam configurations counter.
        self._runs = 0

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

        # Sanity check.
        assert np.shape(self.scores) == (np.size(y), self._runs)

        return self

    # TODO: Return tensor of (scores x configs = num calls) to be used with BBC-CV criterion.
    def objective(self, hparams):

        # NOTE: Assumes standard model API.
        self.model.set_params(**hparams)

        self._runs = self._runs + 1

        # Stratified K-fold cross-validation.
        scores = cross_val_score(
            pipe,
            self.X, self.y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )
        np.append(self.scores, scores, axis=1)

        return 1.0 - np.median(scores)

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

    searcher = ParameterSearch(pipe, space)
    searcher.fit(X, y)
    print(searcher.run_time)
