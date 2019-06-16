# -*- coding: utf-8 -*-
#
# main.py
#

"""
Run model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import sys
sys.path.append('./../')

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from model_comparison import model_comparison_fn

from algorithms import feature_selection
from algorithms import classification
from utils import ioutil


def prep_pipeline(estimators, selectors=None):
    """Prepare modeling pipeline.

    Returns:
        (dict): A dictionary holding all elements of the modelling pipeline.

    """

    setup = {}

    if selectors is None:
        for estimator in estimators:
            setup[f'{selector.NAME}_{estimator.NAME}'] = (
                (VarianceThreshold.__name__, VarianceThreshold()),
                (StandardScaler.__name__, StandardScaler()),
                (estimator.NAME, estimator)
            )
    else:
        for estimator in estimators:
            for selector in selectors:
                setup[f'{selector.NAME}_{estimator.NAME}'] = (
                    (VarianceThreshold.__name__, VarianceThreshold()),
                    (StandardScaler.__name__, StandardScaler()),
                    (selector.NAME, selector),
                    (estimator.NAME, estimator)
                )
    return setup


def balanced_roc_auc(y_true, y_pred):
    """Define a balanced ROC AUC optimization metric."""
    return roc_auc_score(y_true, y_pred, average='weighted')


def experiment():
    path_to_results = './results_all_features_icc.csv'
    path_to_target = './../../iss_original_images/dfs_original_images.csv'
    path_to_predictors = './../../iss_original_images/all_features_orig_images_icc.csv'

    CV = 5
    NUM_REPS = 40
    SCORE = balanced_roc_auc
    MAX_EVALS = 80
    VERBOSE = 1

    estimators = [
        classification.QuadraticDiscriminantEstimator(),
        classification.RidgeClassifierEstimator(),
        classification.ExtraTreesEstimator(),
        classification.XGBoosting(),
        classification.LightGBM(),
        classification.SVCEstimator(),
        classification.LogRegEstimator(),
        classification.RFEstimator(),
        classification.KNNEstimator(),
        classification.DTreeEstimator()
    ]
    selectors = [
        feature_selection.DummySelection(),
        feature_selection.WilcoxonSelection(),
        feature_selection.MultiSURFSelection(),
        feature_selection.ReliefFSelection(),
        feature_selection.ChiSquareSelection(),
        feature_selection.FisherScoreSelection(),
        feature_selection.MutualInformationSelection()
    ]

    y = ioutil.load_target_to_ndarray(path_to_target)
    X = ioutil.load_predictors_to_ndarray(path_to_predictors)

    np.random.seed(seed=0)
    random_states = np.random.choice(40, size=NUM_REPS)

    model_comparison_fn(
        experiments=prep_pipeline(estimators, selectors),
        path_final_results=path_to_results,
        output_dir='./parameter_search',
        random_states=random_states,
        score_func=SCORE,
        max_evals=MAX_EVALS,
        verbose=VERBOSE,
        cv=CV,
        X=X,
        y=y
    )


if __name__ == '__main__':
    experiment()
