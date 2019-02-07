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


def balanced_roc_auc(y_true, y_pred):

    return roc_auc_score(y_true, y_pred, average='weighted')


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

    from algorithms.classification import LogRegEstimator
    from algorithms.classification import PLSREstimator
    from algorithms.classification import SVCEstimator
    from algorithms.classification import GNBEstimator
    from algorithms.classification import RFEstimator
    from algorithms.classification import KNNEstimator

    from sklearn.preprocessing import StandardScaler

    """
    setup = {
        'rfs_relieff_plsr': (
            (StandardScaler.__name__, StandardScaler()),
            #(ReliefFSelection.__name__, ReliefFSelection()),
            (MutualInformationSelection.__name__, MutualInformationSelection()),
            (PLSREstimator.__name__, PLSREstimator())
        ),
        'relieff_svc': (
            (StandardScaler.__name__, StandardScaler()),
            #(ReliefFSelection.__name__, ReliefFSelection()),
            (MutualInformationSelection.__name__, MutualInformationSelection()),
            (SVCEstimator.__name__, SVCEstimator())
        ),
        'relieff_logreg': (
            (StandardScaler.__name__, StandardScaler()),
            #(ReliefFSelection.__name__, ReliefFSelection()),
            (MutualInformationSelection.__name__, MutualInformationSelection()),
            (LogRegEstimator.__name__, LogRegEstimator())
        )
    }
    """

    # On F-beta score: https://stats.stackexchange.com/questions/221997/why-f-beta-score-define-beta-like-that
    # On AUC vs precision/recall: https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba

    np.random.seed(0)
    random_states = np.random.randint(1000, size=2)

    path_to_results = './baseline_nofilter_dfs.csv'
    y = load_target('./../../../data_source/to_analysis/target_dfs.csv')
    X = load_predictors('./../../../data_source/to_analysis/no_filter_concat.csv')

    estimators = {
        PLSREstimator.__name__: PLSREstimator(),
        SVCEstimator.__name__: SVCEstimator(),
        LogRegEstimator.__name__: LogRegEstimator(),
        GNBEstimator.__name__: GNBEstimator(),
        RFEstimator.__name__: RFEstimator(),
        KNNEstimator.__name__: KNNEstimator()
    }

    selectors = {
        ReliefFSelection.__name__: ReliefFSelection(),
        MutualInformationSelection.__name__: MutualInformationSelection(),
        FScoreSelection.__name__: FScoreSelection(),
        WilcoxonSelection.__name__: WilcoxonSelection(),
        ANOVAFvalueSelection.__name__: ANOVAFvalueSelection(),
        Chi2Selection.__name__: Chi2Selection(),
        # NOTE: MRMR is really slow!
        #MRMRSelection.__name__: MRMRSelection(),
    }
    # Wrap up estimators and selectors in experiment.
    setup = {}
    for estimator_id, estimator in estimators.items():
        for selector_id, selector in selectors.items():
            label = '{}_{}'.format(selector_id, estimator_id)
            setup[label] = (
                (StandardScaler.__name__, StandardScaler()),
                (selector_id, selector),
                (estimator_id, estimator)
            )

    comparison.model_comparison(
        comparison_scheme=model_selection.nested_selection,
        X=X, y=y,
        experiments=config_experiments(setup),
        score_func=balanced_roc_auc,
        selection_scheme='k-fold',
        n_splits=2,
        write_prelim=True,
        max_evals=3,
        output_dir='./parameter_search',
        random_states=random_states,
        path_final_results=path_to_results
    )
