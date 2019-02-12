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
                (Whitening.__name__, Whitening(method='zca-cor')),
                (selector_id, selector),
                (estimator_id, estimator)
            )
    return setup


if __name__ == '__main__':
    # TEMP:
    import sys
    sys.path.append('./../')
    sys.path.append('./../../model_comparison')

    import os

    import comparison
    import model_selection

    import numpy as np
    import pandas as pd

    from sklearn.svm import SVC

    from algorithms.transforms import Whitening

    from algorithms.feature_selection import MutualInformationSelection
    from algorithms.feature_selection import ANOVAFvalueSelection
    from algorithms.feature_selection import SequentialSelection
    from algorithms.feature_selection import WilcoxonSelection
    from algorithms.feature_selection import ReliefFSelection
    from algorithms.feature_selection import FScoreSelection
    from algorithms.feature_selection import Chi2Selection
    from algorithms.feature_selection import MRMRSelection

    from algorithms.classification import LogRegEstimator
    from algorithms.classification import PLSREstimator
    from algorithms.classification import SVCEstimator
    from algorithms.classification import GNBEstimator
    from algorithms.classification import KNNEstimator
    from algorithms.classification import RFEstimator

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold


    def balanced_roc_auc(y_true, y_pred, **kwargs):

        # Wrapper for weighted ROC AUC score function.
        return roc_auc_score(y_true, y_pred, average='weighted')


    np.random.seed(0)
    random_states = np.random.randint(1000, size=2)

    path_to_results = './baseline_meta_nofilter_dfs.csv'
    y = load_target('./../../../data_source/to_analysis/target_dfs.csv')
    X = load_predictors('./../../../data_source/to_analysis/no_filter_concat.csv')

    setup = {
        'wlcx_sffs_svc': (
            (VarianceThreshold.__name__, VarianceThreshold()),
            (StandardScaler.__name__, StandardScaler()),
            (Whitening.__name__, Whitening()),
            (WilcoxonSelection.__name__, WilcoxonSelection()),
            (
                SequentialSelection.__name__,
                SequentialSelection(
                    scoring='roc_auc',
                    model_name=SVCEstimator.__name__,
                    model=SVC(
                        class_weight='balanced',
                        verbose=False,
                        cache_size=500,
                        max_iter=-1,
                        decision_function_shape='ovr',
                    ),
                )
            ),
            (SVCEstimator.__name__, SVCEstimator())
        )
    }
    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=setup,
        score_func=balanced_roc_auc,
        cv=5,
        write_prelim=True,
        max_evals=9,
        output_dir='./parameter_search',
        random_states=random_states,
        path_final_results=path_to_results
    )
