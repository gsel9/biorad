# -*- coding: utf-8 -*-
#
# main.py
#

"""
Perform model comparison experiments.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


from collections import OrderedDict
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

from smac.configspace import ConfigurationSpace

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from algorithms.feature_selection import ForwardFloatingSelection


# Globals.
SEED = 0


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


# TODO: To utils!
def config_experiments(experiments):
    """

    """
    global SEED

    pipes_and_params = OrderedDict()
    for (experiment_id, setup) in experiments.items():
        config_space = ConfigurationSpace()
        config_space.seed(SEED)
        for name, algorithm in setup:
            # Avoid transformers without hyperparameters.
            try:
                config_space.add_configuration_space(
                    prefix=name,
                    configuration_space=algorithm.config_space,
                    delimiter='__'
                )
            except:
                pass
        pipes_and_params[experiment_id] = (Pipeline(setup), config_space)

    return pipes_and_params


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

    from algorithms.feature_selection import MutualInformationSelection
    from algorithms.feature_selection import ANOVAFvalueSelection
    from algorithms.feature_selection import WilcoxonSelection
    from algorithms.feature_selection import ReliefFSelection
    from algorithms.feature_selection import FScoreSelection
    from algorithms.feature_selection import Chi2Selection
    from algorithms.feature_selection import MRMRSelection

    from algorithms.meta_estimator import LogRegMetaEstimator
    from algorithms.meta_estimator import PLSRMetaEstimator
    from algorithms.meta_estimator import SVCMetaEstimator
    from algorithms.meta_estimator import GNBMetaEstimator
    from algorithms.meta_estimator import RFMetaEstimator
    from algorithms.meta_estimator import KNNMetaEstimator


    def balanced_roc_auc(y_true, y_pred):
        # Wrapper for weighted ROC AUC score function.
        return roc_auc_score(y_true, y_pred, average='weighted')


    """
    Experiment 1:
    -------------
    * ZCA-cor transformed feature matrix.
    * Nested 5-fold CV.
    * 60 objective evals

    """

    np.random.seed(0)
    random_states = np.random.randint(1000, size=10)

    path_to_results = './baseline_meta_estimator_nofilter_zca_cor_dfs.csv'
    y = load_target('./../../../data_source/to_analysis/target_dfs.csv')
    X = load_predictors('./../../../data_source/to_analysis/no_filter_concat_zca_cor.csv')

    setup = {
        'wlcx_sfs_svc': (
            (VarianceThreshold.__name__, VarianceThreshold()),
            (StandardScaler.__name__, StandardScaler()),
            (WilcoxonSelection.__name__: WilcoxonSelection()),
            (SVCMetaEstimator.__name__: SVCMetaEstimator())
        ),
    }

    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=config_experiments(setup),
        score_func=balanced_roc_auc,
        cv=5,
        oob=None,
        write_prelim=True,
        max_evals=60,
        output_dir='./parameter_search_zca',
        random_states=random_states,
        path_final_results=path_to_results
    )
