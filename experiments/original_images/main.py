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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
    random_states = np.random.choice(40, size=40)

    path_to_results = './all_features_original_images.csv'

    y = ioutil.load_target_to_ndarray(
        './../../../data_source/original_images_data/dfs_original_images.csv'
    )
    X = ioutil.load_predictors_to_ndarray(
        './../../../data_source/original_images_data/all_features_original_images.csv'
    )
    # NOTE: Cannot run group LASSO with feature selection because altering the
    # number of features in the feature matrix causes inconsistency in group
    # indicators.

    priors = [sum(y) / np.size(y), 1 - (sum(y) / np.size(y))]
    lda_model = LinearDiscriminantAnalysis(priors=priors, solver='lsqr')
    qda_model = QuadraticDiscriminantAnalysis(priors=priors)

    estimators = [
        classification.LinearDiscriminantEstimator(model=lda_model),
        classification.QuadraticDiscriminantEstimator(model=qda_model),
        classification.XGBoosting(),
        classification.LightGBM(),
        classification.SVCEstimator(),
        classification.LogRegEstimator(),
        classification.RFEstimator(),
        classification.KNNEstimator(),
        classification.DTreeEstimator(),
        classification.ExtraTreesEstimator()
    ]
    selectors = [
        feature_selection.DummySelection(),
        feature_selection.StudentTTestSelection(),
        feature_selection.MutualInformationSelection(),
        feature_selection.FisherScoreSelection(),
        feature_selection.WilcoxonSelection(),
        feature_selection.ChiSquareSelection(),
        feature_selection.ReliefFSelection(),
    ]

    model_comparison_fn(
        experiments=build_pipeline(estimators, selectors),
        path_final_results=path_to_results,
        output_dir='./parameter_search',
        random_states=random_states,
        score_func=balanced_roc_auc,
        max_evals=80,
        verbose=1,
        cv=10,
        X=X,
        y=y
    )


if __name__ == '__main__':
    experiment(None)
