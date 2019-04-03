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


def build_pipeline(estimators, selectors):
    """Build modeling pipeline."""
    setup = {}
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
    """Define a balanced ROC AUC score metric."""
    return roc_auc_score(y_true, y_pred, average='weighted')


def experiment(_):
    """Experimental setup."""

    np.random.seed(seed=0)
    random_states = np.random.choice(50, size=50)

    path_to_results = './all_features_original_images.csv'

    y = ioutil.load_target_to_ndarray(
        './../../original_images_data/dfs_original_images.csv'
    )
    X = ioutil.load_predictors_to_ndarray(
        './../../original_images_data/all_features_original_images.csv'
    )
    estimators = [
        classification.XGBoosting(),
        classification.LightGBM(),
        classification.ElasticNetEstimator(),
        classification.PLSREstimator(),
        classification.SVCEstimator(),
        classification.LogRegEstimator(),
        classification.RFEstimator(),
        classification.KNNEstimator(),
        classification.DTreeEstimator(),
        classification.ExtraTreeEstimator()
    ]
    selectors = [
        feature_selection.DummySelection(),
        feature_selection.StudentTTestSelection(),
        feature_selection.MutualInformationSelection(),
        feature_selection.FisherScoreSelection(),
        feature_selection.WilcoxonSelection(),
        feature_selection.ANOVAFvalueSelection(),
        feature_selection.ChiSquareSelection(),
        feature_selection.ReliefFSelection(),
    ]

    model_comparison_fn(
        experiments=build_pipeline(estimators, selectors),
        path_final_results=path_to_results,
        output_dir='./parameter_search',
        random_states=random_states,
        score_func=balanced_roc_auc,
        write_prelim=True,
        max_evals=80,
        verbose=1,
        cv=10,
        X=X,
        y=y
    )


if __name__ == '__main__':
    experiment(None)
