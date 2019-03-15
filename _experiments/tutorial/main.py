# -*- coding: utf-8 -*-
#
# main.py
#

"""
Tutorial on predictive modeling.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


# TEMP:
import sys

import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import comparison
import model_selection

from backend import utils

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

# TEMP:
sys.path.append('./../')
sys.path.append('./../../model_comparison')


def build_setup(estimators, selectors):
    """Setup model pipeline elements."""
    setup = {}
    for estimator_id, estimator in estimators.items():
        for selector_id, selector in selectors.items():
            label = f'{selector_id}_{estimator_id}'
            setup[label] = (
                (VarianceThreshold.__name__, VarianceThreshold()),
                (StandardScaler.__name__, StandardScaler()),
                (selector_id, selector),
                (estimator_id, estimator)
            )
    return setup


def balanced_roc_auc(y_true, y_pred):
    """Score function."""
    return roc_auc_score(y_true, y_pred, average='weighted')


def experiment(_):
    """Dummy experiment for tutorial purposes."""
    # Determines the number of experimental repeats across random seeds for
    # pseudo random generators affecting stochastic behaviour.
    np.random.seed(seed=0)
    random_states = np.random.choice(50, size=50)
    # Location to store results.
    path_to_results = './baseline_nofilter_dfs.csv'
    # Load target vector and predictor matrix.
    y = utils.load_target('./data_source/to_analysis/target_dfs.csv')
    X = utils.load_predictors('./data_source/to_analysis/no_filter_concat.csv')
    # Define classifiers.
    estimators = {
        PLSREstimator.__name__: PLSREstimator(),
        SVCEstimator.__name__: SVCEstimator(),
        LogRegEstimator.__name__: LogRegEstimator(),
        GNBEstimator.__name__: GNBEstimator(),
        RFEstimator.__name__: RFEstimator(),
        KNNEstimator.__name__: KNNEstimator(),
        DTreeEstimator.__name__: DTreeEstimator(),
    }
    # Define feature selection algorithms.
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
    # Define optimization and evaluation setup.
    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=build_setup(estimators, selectors),
        score_func=balanced_roc_auc,
        cv=10,
        write_prelim=True,
        max_evals=50,
        output_dir='./parameter_search',
        random_states=random_states,
        path_final_results=path_to_results,
        verbose=1
    )
    return None


if __name__ == '__main__':
    # Demo run.
    experiment(None)
