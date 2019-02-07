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
        spaces = []
        for _, algorithm in setup:
            # Avoid transformers without hyperparameters.
            try:
                spaces.extend(algorithm.hparam_space)
            except:
                pass
        config_space = ConfigurationSpace()
        # Set seed for config space random sampler.
        config_space.seed(SEED)
        # Merge algorithms config spaces.
        config_space.add_hyperparameters(spaces)

        pipes_and_params[experiment_id] = (Pipeline(setup), config_space)

    return pipes_and_params


def balanced_roc_auc(y_true, y_pred):

    return roc_auc_score(y_true, y_pred, average='weighted')


if __name__ == '__main__':
    import sys
    sys.path.append('./../')

    import os
    # TEMP:
    sys.path.append('./../../model_comparison')

    import comparison
    import model_selection

    import numpy as np
    import pandas as pd

    from algorithms.feature_selection import ReliefFSelection
    from algorithms.feature_selection import FeatureScreening
    from algorithms.classification import PLSREstimator
    from algorithms.classification import SVCEstimator
    from algorithms.classification import SVCEstimator

    from sklearn.preprocessing import StandardScaler


    # TODO:
    # * Setup Fisher score and Chi2 feature screening experiments.


    np.random.seed(0)
    random_states = np.random.randint(1000, size=30)

    X = load_predictors('./../../../data_source/to_analysis/no_filter_concat.csv')
    y = load_target('./../../../data_source/to_analysis/target_dfs.csv')

    path_to_results = './test.csv'
    #path_to_results = './baseline_nofilter_dfs.csv' # 0.5091642924976257
    #path_to_results = './rfs_nofilter_dfs.csv'
    #path_to_results = './dgufs_nofilter_dfs.csv'

    # Possible to define multiple experiments (e.g. all possible combos of a
    # clf and a FS).
    setup = {
        #'rfs_relieff_plsr': (
        #    ('1_{}'.format(StandardScaler.__name__), StandardScaler()),
        #    (FeatureScreening.__name__, FeatureScreening()),
        #    ('2_{}'.format(StandardScaler.__name__), StandardScaler()),
        #    (ReliefFSelection.__name__, ReliefFSelection()),
        #    (PLSREstimator.__name__, PLSREstimator())
        #),
        #'relieff_svc': (
        #    (StandardScaler.__name__, StandardScaler()),
        #    (ReliefFSelection.__name__, ReliefFSelection()),
        #    (SVCEstimator.__name__, SVCEstimator())
        #),
        'relieff_logreg': (
            (StandardScaler.__name__, StandardScaler()),
            (ReliefFSelection.__name__, ReliefFSelection()),
            (SVCEstimator.__name__, SVCEstimator())
        ),
    }
    # On F-beta score: https://stats.stackexchange.com/questions/221997/why-f-beta-score-define-beta-like-that
    # On AUC vs precision/recall: https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba
    # TODO: Write prelim results!!!
    comparison.model_comparison(
        comparison_scheme=model_selection.nested_selection,
        X=X, y=y,
        experiments=config_experiments(setup),
        score_func=balanced_roc_auc,
        selection_scheme='k-fold',
        n_splits=5,
        max_evals=25,
        output_dir='./test',
        random_states=random_states,
        path_final_results=path_to_results
    )
    res = pd.read_csv(path_to_results, index_col=0)
    print(np.mean(res['test_score']))
