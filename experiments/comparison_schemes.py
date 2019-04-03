# -*- coding: utf-8 -*-
#
# comparison_schemesn.py
#

"""
Schemes of model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os

from copy import deepcopy
from collections import OrderedDict
from datetime import datetime

from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

import numpy as np

# Local imports.
from utils import ioutil


def nested_cross_validation_smac(
    X, y,
    experiment_id,
    workflow,
    score_func,
    cv: int = 10,
    output_dir = None,
    max_evals: int = 100,
    verbose: int = 1,
    random_state = None,
    path_tmp_results: str = None,
):
    """
    Nested cross-validtion model comparison.

    Args:

    Returns:
        (dict):

    """
    # Determine if should write prelim results, and if some results already
    # exists and needs to be reloaded into memory for output results inclusion.
    if path_tmp_results is None:
        path_case_file = ''
    else:
        path_case_file = os.path.join(
            path_tmp_results, f'experiment_{random_state}_{experiment_id}'
        )
    if os.path.isfile(path_case_file):
        output = ioutil.read_prelim_result(path_case_file)
        print(f'Reloading results from: {path_case_file}')
    else:
        output = {'exp_id': random_state, 'experiment_id': experiment_id}
        if verbose > 0:
            start_time = datetime.now()
            print(f'Running experiment {random_state} with {experiment_id}')
        # Unpack workflow elements and copy pipeline for fresh start.
        pipeline, hparam_space = workflow
        pipeline_cp = deepcopy(pipeline)
        # Run hyperparameter optimization protocol.
        optimizer = SMACSearchCV(
            cv=cv,
            experiment_id=experiment_id,
            workflow=pipeline_cp,
            hparam_space=hparam_space,
            max_evals=max_evals,
            score_func=score_func,
            random_state=random_state,
            verbose=verbose,
            output_dir=output_dir
        )
        optimizer.fit(X, y)
        # Include best hyperparameter config in output records.
        output.update(**optimizer.best_config)
        pipeline_cp = deepcopy(pipeline)
        pipeline_cp.set_params(**optimizer.best_config)
        # Estimate average performance of best model in outer CV loop.
        results = cross_val_score(
            X, y,
            cv,
            random_state,
            pipeline_cp,
            score_func
        )
        output.update(results)
        if path_tmp_results is not None:
            print('Writing results temporary results.')
            ioutil.write_prelim_results(path_case_file, output)
        if verbose > 0:
            duration = datetime.now() - start_time
            print(f'Experiment {random_state} completed in {duration}')
            output['exp_duration'] = duration

    return output


def cross_val_score(
    X, y,
    cv: int,
    random_state: int,
    pipeline,
    score_func,
    shuffle: bool = True,
):
    """Represents the outer K-fold cross-validation (CV) loop of a nested CV
    scheme.

    Returns:
        (collections.OrderedDict): The average and variance of training and
            validation scores, and the number times each feature was
            selected.

    """
    feature_votes = np.zeros(X.shape[1], dtype=int)

    test_scores, train_scores = [], []
    kfolds = StratifiedKFold(cv, shuffle, random_state)
    for train_idx, test_idx in kfolds.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        # Count feature votes with the support attribute.
        feature_votes[pipeline.steps[-2][-1].support] += 1
        test_scores.append(
            score_func(y_test, np.squeeze(pipeline.predict(X_test)))
        )
        train_scores.append(
            score_func(y_train, np.squeeze(pipeline.predict(X_train)))
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
    """Hyperparameter optimization by Sequential Model-based Algorithm
    Configuration (SMAC).

    """
    def __init__(
        self,
        cv: int = None,
        experiment_id = None,
        workflow = None,
        hparam_space = None,
        max_evals: int = None,
        score_func = None,
        random_state: int = None,
        shuffle: bool = True,
        output_dir: str = None,
        verbose: int = 0,
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

        # NOTE: Attribute set with instance.
        self._y = None
        self._X = None
        self._best_config = None

    @property
    def best_config(self):
        """Returns the optimal workflow configuration."""
        return self._best_config

    @property
    def best_workflow(self):
        """Returns the optimally configures workflow."""
        workflow = deepcopy(self.workflow)
        workflow.set_params(**self.best_config)

        return workflow

    def fit(self, X, y):
        """Execture hyperparameter search.

        Args:
            X (array-like): Predictor matrix.
            y (array-like): Target variable vector.

        """
        # NB:
        self._X, self._y = self._check_X_y(X, y)
        # Location to store metadata from hyperparameter search.
        search_metadata_dir = os.path.join(
            self.output_dir, f'{self.experiment_id}_{self.random_state}'
        )
        if not os.path.isdir(search_metadata_dir):
            os.makedirs(search_metadata_dir)
        # NOTE: See https://automl.github.io/SMAC3/dev/options.html for configs.
        scenario = Scenario(
            {
                'use_ta_time': True,
                'wallclock_limit': float(500),
                'cs': self.hparam_space,
                'output_dir': search_metadata_dir,
                'runcount-limit': self.max_evals,
                'run_obj': 'quality',
                'deterministic': 'true',
                'abort_on_first_run_crash': 'true',
             }
        )
        smac = SMAC(
            scenario=scenario,
            rng=np.random.RandomState(self.random_state),
            tae_runner=self.cv_objective_fn
        )
        self._best_config = smac.optimize()

        return self

    def cv_objective_fn(self, hparams):
        """Optimization objective function.

        Args:
            hparams (dict): A hyperparameter configuration.

        Returns:
            (float): The loss score obtained with the input configuration.

        """
        test_scores = []
        kfolds = StratifiedKFold(self.cv, self.shuffle, self.random_state)
        for train_idx, test_idx in kfolds.split(self._X, self._y):
            X_train, X_test = self._X[train_idx], self._X[test_idx]
            y_train, y_test = self._y[train_idx], self._y[test_idx]
            # Copy workflow for fresh start with each fold.
            workflow = deepcopy(self.workflow)
            workflow.set_params(**hparams)
            print('Param config', hparams)
            workflow.fit(X_train, y_train)
            # Test scores as objective loss.
            test_scores.append(
                self.score_func(y_test, np.squeeze(workflow.predict(X_test)))
            )
        return 1.0 - np.mean(test_scores)

    @staticmethod
    def _check_X_y(X, y):
        # Wrapping the sklearn formatter function.
        return check_X_y(X, y)
