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


def build_setup(estimators):

    setup = {}
    for estimator_id, estimator in estimators.items():
        setup[f'{estimator_id}'] = (
            (VarianceThreshold.__name__, VarianceThreshold()),
            (StandardScaler.__name__, StandardScaler()),
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

    from algorithms.classification import LightGBM
    from algorithms.classification import XGBoosting

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


    def balanced_roc_auc(y_true, y_pred):

        return roc_auc_score(y_true, y_pred, average='weighted')

    np.random.seed(seed=0)
    random_states = np.random.choice(100, size=100)

    path_to_results = './80evals_100eps_boosting_dfs.csv'
    X = load_predictors('./../../../data_source/to_analysis/anomaly_filtered_concat.csv')

    dropped = [38, 45, 82]
    y = pd.read_csv('./../../../data_source/to_analysis/target_dfs.csv', index_col=0)
    y.drop(dropped, inplace=True)
    y = np.squeeze(y.values)

    estimators = {
        LightGBM.__name__: LightGBM(),
        XGBoosting.__name__: XGBoosting(),
    }
    # NB: Unable to run LightGBM in parallell. Maybe because LightGBM n_jobs=-1
    # by default?
    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=build_setup(estimators),
        score_func=balanced_roc_auc,
        cv=10,
        write_prelim=True,
        max_evals=80,
        output_dir='./parameter_search',
        random_states=random_states,
        path_final_results=path_to_results,
        n_jobs=1,
        verbose=1
    )
