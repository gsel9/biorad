# -*- coding: utf-8 -*-
#
# main.py
#

"""
Execute radiomic model comparison experiments.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


from collections import OrderedDict
from sklearn.pipeline import Pipeline

from smac.configspace import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter


# TODO: To utils!
def load_target(path_to_target, index_col=0, classify=True):

    var = pd.read_csv(path_to_target, index_col=index_col)
    if classify:
        return np.squeeze(var.values).astype(np.int32)
    else:
        return np.squeeze(var.values).astype(np.float32)


# TODO: To utils!
def load_predictors(path_to_data, index_col=0, regex=None):

    data = pd.read_csv(path_to_data, index_col=index_col)
    if regex is None:
        return np.array(data.values, dtype=np.float32)
    else:
        target_features = data.filter(regex=regex)
        return np.array(data.loc[:, target_features].values, dtype=np.float32)


# TODO: To utils!
def config_experiments(experiments):

    pipes_and_params = OrderedDict()
    for (experiment_id, setup) in experiments.items():
        spaces = []
        for _, algorithm in setup:
            # Avoid transformers without hyperparameters.
            try:
                spaces.extend(algorithm.hparam_space)
            except:
                pass
        # Merge algorithms config spaces.
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters(spaces)

        pipes_and_params[experiment_id] = (Pipeline(setup), config_space)

    return pipes_and_params


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
    from algorithms.classification import PLSREstimator

    from sklearn.preprocessing import StandardScaler

    from sklearn.metrics import roc_auc_score

    # FEATURE SET:
    # QUESTION: Something in the feature extraction or preprocessing procedure
    # conducted by Alise that rendering features superior?
    #X = load_predictors('./../../../data_source/to_analysis/no_filter_concat.csv')
    # Score (DFS): 0.5299233140225786
    # Score (LRR):

    #X = load_predictors('./../../../data_source/to_analysis/alise_setup.csv')
    # Score (DFS): 0.5338309566250743
    # Score (LRR): 0.5012103196313723

    X = load_predictors('./../../../data_source/to_analysis/sqroot_concat.csv')
    # Score (DFS):
    # Score (LRR): 0.4963584789242684

    y = load_target('./../../../data_source/to_analysis/target_dfs.csv')

    path_to_results = './test.csv'

    # Possible to define multiple experiments (e.g. all possible combos of a
    # clf and a fs.)
    setup = {
        'relieff_plsr': (
            (StandardScaler.__name__, StandardScaler()),
            (ReliefFSelection.__name__, ReliefFSelection()),
            (PLSREstimator.__name__, PLSREstimator())
        ),
    }

    np.random.seed(0)
    random_states = np.random.randint(1000, size=30)

    comparison.model_comparison(
        comparison_scheme=model_selection.nested_kfold_selection,
        X=X, y=y,
        experiments=config_experiments(setup),
        score_func=roc_auc_score,
        cv=4,
        max_evals=25,
        execdir='./outputs',
        random_states=random_states,
        path_final_results=path_to_results
    )

    res = pd.read_csv(path_to_results, index_col=0)
    print(np.mean(res['test_score']))

    """

    data_raw_df = pd.read_excel(xls, sheet_name='tilbakefall_siste', index_col=0)
    X = data_raw_df.values
    #load_predictors('./../../data_source/to_analysis/alise_setup.csv')
    #X = load_predictors('./../../data_source/to_analysis/sqroot_concat.csv')

    # TARGET:
    #y = load_target('./../../data_source/to_analysis/target_dfs.csv')
    #y = load_target('./../../data_source/to_analysis/target_lrr.csv')
    y_dfs = pd.read_csv('./alise/target_dfs.csv', index_col=0)
    y_dfs = np.squeeze(y_dfs.iloc[np.squeeze(np.where(np.isin(y_dfs.index.values, data_raw_df.index.values))), :].values)

    #y_lrr = pd.read_csv('./alise/target_lrr.csv', index_col=0)
    #y_lrr = np.squeeze(y_lrr.iloc[np.squeeze(np.where(np.isin(y_lrr.index.values, data_raw_df.index.values))), :].values)



    # RESULTS LOCATION:
    # Categorical/continous
    #path_to_results = './chi2_mi.csv' (slow)
    #path_to_results = './chi2_anova.csv' #(fast)
    #path_to_results = './mi_mi.csv' (slow)
    #path_to_results = './mi_anovatest.csv' #(fast) + winner!
    #path_to_results = './mi_mrmr.csv' #(fast) + winner!
    #path_to_results = './zca_corr_mi_anovatest.csv.csv'
    #path_to_results = './pca_mi_anovatest.csv.csv'
    #path_to_results = './mi_dgufs.csv' (slow)
    path_to_results = './test.csv'

    # mRMR feature screening is very slow, but indicates the best results.

    #path_to_results = './../data/experiments/no_filter_concat_dfs.csv'
    #path_to_results = './../data/experiments/complete_decorr_lrr.csv'

    # EXPERIMENTAL SETUP:
    CV = 4
    MAX_EVALS = 200
    NUM_EXP_REPS = 10
    SCORING = roc_auc_score

    # Generate seeds for pseudo-random generators to use in each experiment.
    np.random.seed(0)
    random_states = np.random.randint(1000, size=NUM_EXP_REPS)

    # Generate pipelines from config elements.
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    # Parameters to tune the TPE algorithm.
    tpe = partial(
        hyperopt.tpe.suggest,
        # Sample 1000 candidates and select the candidate with highest
        # Expected Improvement (EI).
        n_EI_candidates=1000,
        # Use 20 % of best observations to estimate next set of parameters.
        gamma=0.2,
        # First 20 trials are going to be random (include probability theory
        # for 90 % CI with this setup).
        n_startup_jobs=60,
    )
    comparison.model_comparison(
        model_selection.nested_kfold_selection,
        X, y,
        tpe,
        {'ReliefFSelection_PLSRegression': pipes_and_params['ReliefFSelection_PLSRegression']},
        SCORING,
        CV,
        MAX_EVALS,
        shuffle=True,
        verbose=1,
        random_states=random_states,
        balancing=False,
        write_prelim=True,
        error_score='all',
        n_jobs=None,
        path_final_results=path_to_results
    )
    res = pd.read_csv(path_to_results, index_col=0)
    print(np.mean(res['test_score']))
    """
