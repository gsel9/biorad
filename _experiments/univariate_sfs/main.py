# -*- coding: utf-8 -*-
#
# main.py
#

"""
Perform model comparison experiments.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import pandas as pd
import numpy as np
import sys
sys.path.append('./../')
sys.path.append('./../../model_comparison')

import os

import comparison
import model_selection

import numpy as np
import pandas as pd

from algorithms.feature_selection import CorrelationEnsembleSelection
from algorithms.feature_selection import MutualInformationSelection
from algorithms.feature_selection import StudentTTestSelection
from algorithms.feature_selection import ANOVAFvalueSelection
from algorithms.feature_selection import WilcoxonSelection
from algorithms.feature_selection import ReliefFSelection
from algorithms.feature_selection import FScoreSelection
from algorithms.feature_selection import Chi2Selection
from algorithms.feature_selection import MRMRSelection

from algorithms.classification import LogRegEstimator
from algorithms.classification import DTreeEstimator
from algorithms.classification import PLSREstimator
from algorithms.classification import SVCEstimator
from algorithms.classification import KNNEstimator
from algorithms.classification import GNBEstimator
from algorithms.classification import RFEstimator

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def balanced_roc_auc(y_true, y_pred):

    return roc_auc_score(y_true, y_pred, average='weighted')


# TODO: To utils!
def load_target(path_to_target, index_col=0, classify=True):
    """

    """
    var = pd.read_csv(path_to_target, index_col=index_col)
    if classify:
        return np.squeeze(var.values).astype(np.int32)
    else:
        return np.squeeze(var.values).astype(np.float32)


# TODO: To utils!
def load_predictors(path_to_data, index_col=0, regex=None):
    """

    """
    data = pd.read_csv(path_to_data, index_col=index_col)
    if regex is None:
        return np.array(data.values, dtype=np.float32)
    else:
        target_features = data.filter(regex=regex)
        return np.array(data.loc[:, target_features].values, dtype=np.float32)


def build_setup(estimators, selectors):

    setup = {}
    for estimator_id, estimator in estimators.items():
        for selector_id, selector in selectors.items():
            label = '{}_{}'.format(selector_id, estimator_id)
            setup[label] = (
                (VarianceThreshold.__name__, VarianceThreshold()),
                (StandardScaler.__name__, StandardScaler()),
                (selector_id, selector),
                (estimator_id, estimator)
            )
    return setup


def univariate_forward_selection(X, y):

    path_to_results = './50evals_50reps_dtree_univariate_sfs_dfs.csv'

    rgen = np.random.RandomState(0)
    random_states = rgen.choice(50, size=50)

    estimators = {
        DTreeEstimator.__name__: DTreeEstimator(
            with_selection=True,
            forward=True,
            floating=False
        ),
    }
    selectors = {
        StudentTTestSelection.__name__: StudentTTestSelection(),
        MutualInformationSelection.__name__: MutualInformationSelection(),
        FScoreSelection.__name__: FScoreSelection(),
        WilcoxonSelection.__name__: WilcoxonSelection(),
        ANOVAFvalueSelection.__name__: ANOVAFvalueSelection(),
        Chi2Selection.__name__: Chi2Selection(),
        MRMRSelection.__name__: MRMRSelection(),
        ReliefFSelection.__name__: ReliefFSelection(),
    }

    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=build_setup(estimators, selectors),
        score_func=balanced_roc_auc,
        cv=10,
 	n_jobs=-1,
        write_prelim=True,
        max_evals=50,
        output_dir='./dtree_parameter_search',
        random_states=random_states,
        path_final_results=path_to_results,
        verbose=1
    )


if __name__ == '__main__':
    X = load_predictors('./../../../anomaly_filtered_concat.csv')

    dropped = [38, 45, 82]
    y = pd.read_csv('./../../../target_dfs.csv', index_col=0)
    y.drop(dropped, inplace=True)
    y = np.squeeze(y.values)

    univariate_forward_selection(X, y)
    
