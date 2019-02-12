# -*- coding: utf-8 -*-
#
# main.py
#

"""
Perform model comparison experiments.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


from sklearn.metrics import roc_auc_score


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
                # NOTE: Univaraite feature selection reported constant features
                # in some folds. Remove features constant in folds consulting a
                # variance threshold of zero variance.
                # NB: Variance threshold before scaling to unit feature
                # variances.
                (VarianceThreshold.__name__, VarianceThreshold()),
                (StandardScaler.__name__, StandardScaler()),
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
    from sklearn.feature_selection import VarianceThreshold

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
    def balanced_roc_auc(y_true, y_pred):
        # Wrapper for weighted ROC AUC score function.
        return roc_auc_score(y_true, y_pred, average='weighted')


    """
    Temp experiment:
    -------------
    * ZCA-cor transformed feature matrix.
    * Nested 5-fold CV.
    * 60 objective evals

    Final experiment:
    -------------
    * ZCA-cor transformed feature matrix.
    * Nested 10-fold CV.
    * 60 objective evals
    * Most used radiomics models and feature selectors (see summary paper)

    Follow up:
    -------------
    * Multiple ways to construct data analysis pipelines, but not sure
      hypothesized information is readily available in the images. Use
      general approximators (NN, but Tsetlin Machines may be more promising
      considering small amounts of data.

    """

    np.random.seed(0)
    random_states = np.random.randint(1000, size=10)

    # TODO:
    # * Create a heat map/cluster map (Seaborn) of feature matrix highlighting outliers.
    # * Include a test for normality just to verify whether or not any assumption of
    #   normality breaks or holds.

    #path_to_results = './baseline_nofilter_dfs.csv'
    path_to_results = './baseline_nofilter_zca_cor_dfs.csv'
    y = load_target('./../../../data_source/to_analysis/target_dfs.csv')
    #X = load_predictors('./../../../data_source/to_analysis/no_filter_concat.csv')
    # TODO: Verify data satisfies distribution assumptions of co-variance
    # estimator (data is Z-score transformed in befonre co-variance estimated
    # in whitening procedure).
    # Checkout:
    # * https://mathoverflow.net/questions/290490/how-to-measure-distribution-of-high-dimensional-data
    # * https://www.cs.umd.edu/~hjs/mkbook/chapter4.pdf
    X = load_predictors('./../../../data_source/to_analysis/no_filter_concat_zca_cor.csv')

    estimators = {
        PLSREstimator.__name__: PLSREstimator(),
        SVCEstimator.__name__: SVCEstimator(),
        LogRegEstimator.__name__: LogRegEstimator(),
        GNBEstimator.__name__: GNBEstimator(),
        RFEstimator.__name__: RFEstimator(),
        #KNNEstimator.__name__: KNNEstimator()
    }
    selectors = {
        ReliefFSelection.__name__: ReliefFSelection(),
        MutualInformationSelection.__name__: MutualInformationSelection(),
        FScoreSelection.__name__: FScoreSelection(),
        WilcoxonSelection.__name__: WilcoxonSelection(),
        ANOVAFvalueSelection.__name__: ANOVAFvalueSelection(),
        Chi2Selection.__name__: Chi2Selection(),
        #MRMRSelection.__name__: MRMRSelection(),
    }

    setup = build_setup(estimators, selectors)

    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=setup,
        score_func=balanced_roc_auc,
        cv=5,
        oob=None,
        write_prelim=True,
        max_evals=60,
        output_dir='./parameter_search_zca',
        random_states=random_states,
        path_final_results=path_to_results
    )
