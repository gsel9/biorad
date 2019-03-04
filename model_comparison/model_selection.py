# -*- coding: utf-8 -*-
#
# model_selection.py
#

"""
Work function for model selection experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

from copy import deepcopy
from collections import OrderedDict
from datetime import datetime
import os
from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
import numpy as np
import utils


def model_selection(
    X, y,
    experiment_id,
    workflow,
    score_func,
    cv: int=10,
    output_dir=None,
    max_evals: int=100,
    verbose: int=1,
    shuffle: bool=True,
    random_state=None,
    path_tmp_results: str=None,
):
    """
    Nested cross-validtion model comparison.

    Args:

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
        output = {'exp_id': random_state, 'experiment_id': experiment_id}
        if verbose > 0:
            start_time = datetime.now()
            print('Running experiment {} with {}'
                  ''.format(random_state, experiment_id))

        pipeline, hparam_space = workflow
        _pipeline = deepcopy(pipeline)

        optimizer = SMACSearchCV(
            cv=cv,
            experiment_id=experiment_id,
            workflow=_pipeline,
            hparam_space=hparam_space,
            max_evals=max_evals,
            score_func=score_func,
            random_state=random_state,
            shuffle=shuffle,
            verbose=verbose,
            output_dir=output_dir
        )
        optimizer.fit(X, y)
        output.update(**optimizer.best_config)

        _pipeline = deepcopy(pipeline)
        _pipeline.set_params(**optimizer.best_config)

        # Estimate average performance of best model.
        results = cross_val_score(
            X, y, cv, shuffle, random_state, _pipeline, score_func
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


def cross_val_score(X, y, cv, shuffle, random_state, workflow, score_func):

    feature_votes = np.zeros(X.shape[1], dtype=int)

    test_scores, train_scores = [], []
    folds = StratifiedKFold(cv, shuffle, random_state)
    for train_idx, test_idx in folds.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        workflow.fit(X_train, y_train)

        _, model = workflow.steps[-1]
        feature_votes[model.support] += 1

        test_scores.append(
            score_func(y_test, np.squeeze(workflow.predict(X_test)))
        )
        train_scores.append(
            score_func(y_train, np.squeeze(workflow.predict(X_train)))
        )
    return OrderedDict(
        [
            ('test_score', np.mean(test_scores)),
            ('train_score', np.mean(train_scores)),
            ('test_score_variance', np.var(test_scores)),
            ('train_score_variance', np.var(train_scores)),
            ('feature_votes', np.array(feature_votes, dtype=int))
        ]
    )


class SMACSearchCV:

    def __init__(
        self,
        cv=None,
        experiment_id=None,
        workflow=None,
        hparam_space=None,
        max_evals=None,
        score_func=None,
        random_state=None,
        shuffle=True,
        deterministic=True,
        output_dir=None,
        verbose=0,
        abort_first_run=True
    ):
        self.cv = cv
        self.experiment_id = experiment_id
        self.workflow = workflow
        self.hparam_space = hparam_space
        self.max_evals = max_evals
        self.score_func = score_func
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.output_dir = output_dir
        self.abort_first_run = abort_first_run

        if deterministic:
            self.deterministic = 'true'
        else:
            self.deterministic = 'false'

        self._best_config = None

    @property
    def best_config(self):

        return self._best_config

    @property
    def best_workflow(self):

        _workflow = deepcopy(self.workflow)
        _workflow.set_params(**self.best_config)

        return _workflow

    def fit(self, X, y):

        # NB: Carefull!
        self.X, self.y = self._check_X_y(X, y)

        # Location to store metadata from hyperparameter search.
        _output_dir = os.path.join(
            self.output_dir, '{}_{}'.format(self.experiment_id, self.random_state)
        )
        if not os.path.isdir(_output_dir):
            os.makedirs(_output_dir)

        # NOTE: See https://automl.github.io/SMAC3/dev/options.html for
        # options.
        scenario = Scenario(
            {
                'run_obj': 'quality',
                'runcount-limit': self.max_evals,
                'cs': self.hparam_space,
                'deterministic': self.deterministic,
                'output_dir': _output_dir,
                'abort_on_first_run_crash': self.abort_first_run,
                'wallclock_limit': float(500),
                'use_ta_time': True,
             }
        )
        smac = SMAC(
            scenario=scenario,
            rng=np.random.RandomState(self.random_state),
            tae_runner=self.cv_objective
        )
        self._best_config = smac.optimize()

        return self

    def cv_objective(self, hparams):

        test_scores = []
        folds = StratifiedKFold(self.cv, self.shuffle, self.random_state)
        for train_idx, test_idx in folds.split(self.X, self.y):

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            _workflow = deepcopy(self.workflow)
            _workflow.set_params(**hparams)
            _workflow.fit(X_train, y_train)

            test_scores.append(
                self.score_func(y_test, np.squeeze(_workflow.predict(X_test)))
            )
        return 1.0 - np.mean(test_scores)

    @staticmethod
    def _check_X_y(X, y):
        # Wrapping the sklearn formatter function.
        return check_X_y(X, y)
