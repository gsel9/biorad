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

    from algorithms.classification import GroupLASSO

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def balanced_roc_auc(y_true, y_pred):

        return roc_auc_score(y_true, y_pred, average='weighted')

    np.random.seed(seed=0)
    random_states = np.random.choice(50, size=50)

    path_to_results = './50evals_50reps_grouplasso_dfs.csv'
    X = load_predictors('./../../../data_source/to_analysis/anomaly_filtered_concat.csv')

    dropped = [38, 45, 82]
    y = pd.read_csv('./../../../data_source/to_analysis/target_dfs.csv', index_col=0)
    y.drop(dropped, inplace=True)
    y = np.squeeze(y.values).astype(np.int32)

    # Groups indicated from 1, ...
    group_idx = np.load('./../../../data_source/to_analysis/feature_categories.npy')

    estimators = {
        GroupLASSO.__name__: GroupLASSO(group_idx=group_idx),
    }
    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=build_setup(estimators),
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
