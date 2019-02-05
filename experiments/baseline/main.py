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

from smac.configspace import ConfigurationSpace
#from ConfigSpace.conditions import InCondition
#from ConfigSpace.hyperparameters import CategoricalHyperparameter
#from ConfigSpace.hyperparameters import UniformFloatHyperparameter
#from ConfigSpace.hyperparameters import UniformIntegerHyperparameter


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

    # ERROR: Issue with reproduciability (probably random states).

    #X = load_predictors('./../../../data_source/to_analysis/no_filter_concat.csv')
    # Score (DFS): 0.5299233140225786
    # Score (LRR):

    #X_orig = pd.read_csv('./../../../data_source/to_analysis/no_filter_concat.csv', index_col=0)
    #y_orig = pd.read_csv('./../../../data_source/to_analysis/target_dfs.csv', index_col=0)
    #data = pd.read_excel('./../../../data_source/to_analysis/alise_orig.xlsx', index_col=0)
    #target_samples = [idx for idx in X_orig.index if idx in data.index]
    #X = X_orig.loc[target_samples, :].values
    #y = np.squeeze(y_orig.loc[target_samples].values)
    # Score (DFS): 0.6611585922247688
    # Score (LRR):

    #X = load_predictors('./../../../data_source/to_analysis/gauss05_concat.csv')
    #y = load_target('./../../../data_source/to_analysis/target_dfs.csv')
    # Score (DFS): 0.5612211267082591 (compared to all samples: 0.6972678465325525)
    #X = X_orig.loc[target_samples, :].values
    #y = np.squeeze(y_orig.loc[target_samples].values)
    #X_orig = pd.read_csv('./../../../data_source/to_analysis/gauss05_concat.csv', index_col=0)
    #y_orig = pd.read_csv('./../../../data_source/to_analysis/target_dfs.csv', index_col=0)
    #data = pd.read_excel('./../../../data_source/to_analysis/alise_orig.xlsx', index_col=0)
    #target_samples = [idx for idx in X_orig.index if idx in data.index]
    #X = X_orig.loc[target_samples, :].values
    #y = np.squeeze(y_orig.loc[target_samples].values)
    # Score (DFS): 0.6972678465325525

    #data = pd.read_excel('./../../../data_source/to_analysis/alise_orig.xlsx', index_col=0)
    #y = np.squeeze(data['Toklasser'].values)
    #X = data.drop('Toklasser', 1).values
    # Score (DFS): 0.801366474620151/0.8068724270011033/0.7968821619556914

    # Alises samples are of better quality? YES!
    #X = load_predictors('./../../../data_source/to_analysis/alise_setup.csv')
    # Score (DFS): 0.5338309566250743
    # Score (LRR): 0.5012103196313723
    #X_orig = pd.read_csv('./../../../data_source/to_analysis/alise_setup.csv', index_col=0)
    #y_orig = pd.read_csv('./../../../data_source/to_analysis/target_dfs.csv', index_col=0)
    #data = pd.read_excel('./../../../data_source/to_analysis/alise_orig.xlsx', index_col=0)
    #target_samples = [idx for idx in X_orig.index if idx in data.index]
    #X = X_orig.loc[target_samples, :].values
    #y = np.squeeze(y_orig.loc[target_samples].values)
    # Score (DFS): 0.7081126389949919 (compared to all samples: 0.5338309566250743)

    # Check if my targets give different results from Alises targets.
    #y_orig = pd.read_csv('./../../../data_source/to_analysis/target_dfs.csv', index_col=0)
    #data = pd.read_excel('./../../../data_source/to_analysis/alise_orig.xlsx', index_col=0)
    #X = data.drop('Toklasser', 1).values
    #target_samples = [idx for idx in y_orig.index if idx in data.index]
    #y = np.squeeze(y_orig.loc[target_samples].values)
    # Score (DFS): 0.8114627100840335/0.8137376920465156

    # Alises samples are of better quality? YES!
    #X = load_predictors('./../../../data_source/to_analysis/sqroot_concat.csv')
    # Score (DFS): 0.5539652035056447
    # Score (LRR): 0.4963584789242684
    #X_orig = pd.read_csv('./../../../data_source/to_analysis/sqroot_concat.csv', index_col=0)
    #y_orig = pd.read_csv('./../../../data_source/to_analysis/target_dfs.csv', index_col=0)
    #data = pd.read_excel('./../../../data_source/to_analysis/alise_orig.xlsx', index_col=0)
    #target_samples = [idx for idx in X_orig.index if idx in data.index]
    #X = X_orig.loc[target_samples, :].values
    #y = np.squeeze(y_orig.loc[target_samples].values)
    # Score (DFS): 0.65200136872931 (compared to all samples: 0.5539652035056447)

    # Difference between using Alises target vector and my target vector?


    path_to_results = './test.csv'

    # Possible to define multiple experiments (e.g. all possible combos of a
    # clf and a fs.)
    setup = {
        'relieff_plsr': (
            (StandardScaler.__name__, StandardScaler()),
            (ReliefFSelection.__name__, ReliefFSelection()),
            (PLSREstimator.__name__, PLSREstimator())
        ),
        #'relieff_plsr': (
        #    (StandardScaler.__name__, StandardScaler()),
        #    (ReliefFSelection.__name__, ReliefFSelection()),
        #    (PLSREstimator.__name__, PLSREstimator())
        #),
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
        output_dir='./testing',
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
