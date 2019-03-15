# -*- coding: utf-8 -*-
#
# main.py
#

"""
Perform model comparison experiments.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import sys
sys.path.append('./../')
sys.path.append('./../../model_comparison')

import comparison
import model_selection

import pandas as pd
import numpy as np

from algorithms.feature_selection import GeneralizedFisherScore

from algorithms.classification import LogRegEstimator
from algorithms.classification import DTreeEstimator
from algorithms.classification import PLSREstimator
from algorithms.classification import SVCEstimator
from algorithms.classification import KNNEstimator
from algorithms.classification import GNBEstimator
from algorithms.classification import RFEstimator

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from backend import utils


def balanced_roc_auc(y_true, y_pred):

    return roc_auc_score(y_true, y_pred, average='weighted')


def build_setup(estimators, selectors):

    setup = {}
    for estimator_id, estimator in estimators.items():
        for selector_id, selector in selectors.items():
            label = 'f{selector_id}_{estimator_id}'
            setup[label] = (
                (VarianceThreshold.__name__, VarianceThreshold()),
                (StandardScaler.__name__, StandardScaler()),
                (selector_id, selector),
                (estimator_id, estimator)
            )
    return setup


def experiment(_):

    np.random.seed(seed=0)
    random_states = np.random.choice(50, size=50)

    path_to_results = './baseline_nofilter_dfs.csv'
    dropped = [38, 45, 82]
    y = pd.read_csv('./../../../data_source/to_analysis/target_dfs.csv', index_col=0)
    y.drop(dropped, inplace=True)
    y = np.squeeze(y.values).astype(np.int32)

    X = utils.load_predictors('./../../../data_source/to_analysis/anomaly_filtered_concat.csv')

    estimators = {
        #PLSREstimator.__name__: PLSREstimator(),
        #SVCEstimator.__name__: SVCEstimator(),
        #LogRegEstimator.__name__: LogRegEstimator(),
        #GNBEstimator.__name__: GNBEstimator(),
        #RFEstimator.__name__: RFEstimator(),
        #KNNEstimator.__name__: KNNEstimator(),
        DTreeEstimator.__name__: DTreeEstimator(),
    }
    selectors = {
        GeneralizedFisherScore.__name__: GeneralizedFisherScore(num_classes=2),
    }
    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=build_setup(estimators, selectors),
        score_func=balanced_roc_auc,
        cv=10,
        write_prelim=True,
        max_evals=50,
        n_jobs=-1,
        output_dir='./parameter_search',
        random_states=random_states,
        path_final_results=path_to_results,
        verbose=1
    )


if __name__ == '__main__':
    experiment(None)
