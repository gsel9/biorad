"""
TODO:
* Shift clf to y axis and FS to x-axis.

"""


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import gridspec

from stat_tests import test_normality

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1 import colorbar

import fig_config as CONFIG


CONFIG.plot_setup()


def format_selector_labels(labels):

    mapper = {
        'ChiSquareSelection': 'Chi Square',
        'DummySelection': 'No Feature\nSelection',
        'WilcoxonSelection': 'Wilcoxon',
        'FisherScoreSelection': 'Fisher Score',
        'MultiSURFSelection': 'MultiSURF',
        'MutualInformationSelection': 'Mutual\nInformation',
        'ReliefFSelection': 'ReliefF'
    }
    new_labels = []
    for label in labels:
        new_labels.append(mapper[label])

    return new_labels


def format_estimator_labels(labels):

    mapper = {
        'QuadraticDiscriminantEstimator': 'QDA',
        'ExtraTreesEstimator': 'ETrees',
        'KNNEstimator': 'KNN',
        'LightGBM': 'LGBM',
        'LogRegEstimator': 'LogReg',
        'RFEstimator': 'RForest',
        'SVCEstimator': 'SVC',
        'DTreeEstimator': 'DTree',
        'XGBoost': 'XGB',
        'RidgeClassifier': 'Ridge'
    }
    new_labels = []
    for label in labels:
        new_labels.append(mapper[label])

    return new_labels



def gen_results_matrix(results, kind='test_score'):

    _selector_lbls, _estimator_lbls = [], []
    for label in np.unique(results['experiment_id']):

        selector_lbl, estimator_lbl = label.split('_')

        _selector_lbls.append(selector_lbl)
        _estimator_lbls.append(estimator_lbl)

    selector_lbls = np.unique(_selector_lbls)
    estimator_lbls = np.unique(_estimator_lbls)


    results_mat = np.zeros((len(estimator_lbls), len(selector_lbls)))
    for row_num, estimator_lbl in enumerate(estimator_lbls):
        for col_num, selector_lbl in enumerate(selector_lbls):

            label = f'{selector_lbl}_{estimator_lbl}'
            location = np.where(label == np.array(results['experiment_id']))
            scores = results.iloc[np.squeeze(location), :][kind]

            results_mat[row_num, col_num] = np.mean(scores)

    return results_mat, selector_lbls, estimator_lbls



def gen_heatmap(results, path_to_fig=None, kind='test_score', show=False):

    results_mat, selector_lbls, estimator_lbls = gen_results_matrix(
        results, kind=kind
    )
    plt.title('AUC', x=1.035, y=1.03)
    hmap = sns.heatmap(
        results_mat.T * 100,
        yticklabels=format_selector_labels(selector_lbls),
        xticklabels=format_estimator_labels(estimator_lbls),
        vmin=np.nanmin(results_mat) * 100 - 3,
        vmax=np.nanmax(results_mat) * 100 + 3,
        cmap=plt.cm.viridis,
        robust=True,
        annot=True,
        fmt='.1f',
        square=1,
        linewidth=.5,
        cbar=False,
        annot_kws={'size': 18},
    )
    plt.xlabel('Classification Algorithm', va='top', ha='center')
    plt.ylabel('Feature Selection Algorithm', va='bottom', ha='center')
    hmap.set_yticklabels(hmap.get_yticklabels(), rotation=0)
    hmap.set_xticklabels(
        hmap.get_xticklabels(), rotation=0, va='top', ha='center'
    )
    # labelpad=-40, y=1.05, rotation=0
    ax_divider = make_axes_locatable(hmap)
    cax = ax_divider.append_axes('right', size='3%', pad='2%')
    colorbar.colorbar(
        hmap.get_children()[0],
        cax=cax,
        orientation='vertical'
    )
    #cax.xaxis.set_label_text('AUC', fontname='Sans')
    #cax.xaxis.set_label_position('top')
    cbar_ticks = np.linspace(
        np.nanmin(results_mat) * 100 - 3,
        np.nanmax(results_mat) * 100 + 3,
        6
    )
    cax.yaxis.set_ticks(cbar_ticks)
    cax.yaxis.set_ticklabels([f'{num:.01f}' for num in cbar_ticks])

    if path_to_fig is not None:
        plt.savefig(
            path_to_fig,
            bbox_inches='tight',
            transparent=True,
            dpi=CONFIG.DPI,
        )
    if show:
        plt.show()


def hpv_b_clinical_only():

    path_to_fig = './../model_comparison_results/hmap_hpv_b_clinical.pdf'
    path_to_mod_comp_results = './../../data_source/results/hpv_splitting/results_hpv_b_clinical_only.csv'

    results = pd.read_csv(path_to_mod_comp_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_fig)


def baseline_clinical():

    path_to_fig = './../model_comparison_results/hmap_clinical_baseline.pdf'
    path_to_mod_comp_results = './../../data_source/results/original_images/results_full_clinical.csv'

    results = pd.read_csv(path_to_mod_comp_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_fig)


def baseline():

    path_to_fig = './../model_comparison_results/hmap_baseline.pdf'
    path_to_mod_comp_results = './../../data_source/results/original_images/results_all_features_original_images.csv'

    results = pd.read_csv(path_to_mod_comp_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_fig)


def removed_damaged_slices():

    path_to_fig = './../model_comparison_results/hmap_removed_damaged.pdf'
    path_to_mod_comp_results = './../../data_source/results/removed_broken_slices/results_all_features_removed_broken_slices.csv'

    results = pd.read_csv(path_to_mod_comp_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_fig)


def reduce_feature_redundancy():

    path_to_fig_hassan = './../model_comparison_results/hmap_hassan_reduced.pdf'
    path_to_fig_hassan_scc = './../model_comparison_results/hmap_hassan_scc_reduced.pdf'

    path_to_hassan_mod_comp_results = './../../data_source/results/hassan_original_images/results_all_features_icc.csv'
    path_to_hassan_scc_mod_comp_results = './../../data_source/results/dropped_corr/results_all_features_original_images_icc_dropped_correlated.csv'

    results = pd.read_csv(path_to_hassan_mod_comp_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_fig_hassan)

    results = pd.read_csv(path_to_hassan_scc_mod_comp_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_fig_hassan_scc)


def hpv_experiments():

    path_to_figure_hpv_a_all_feats = './../model_comparison_results/hmap_hpv_a_all_feats.pdf'
    path_to_figure_hpv_b_all_feats = './../model_comparison_results/hmap_hpv_b_all_feats.pdf'
    path_to_figure_hpv_a_reduced = './../model_comparison_results/hmap_hpv_a_reduced.pdf'
    path_to_figure_hpv_b_reduced = './../model_comparison_results/hmap_hpv_b_reduced.pdf'

    path_hpb_a_all_feats_results = './../../data_source/results/hpv_splitting/results_all_features_original_images_hpv_group_a.csv'
    path_hpb_b_all_feats_results = './../../data_source/results/hpv_splitting/results_all_features_original_images_hpv_group_b.csv'
    path_hpb_a_reduced_results = './../../data_source/results/hpv_splitting/results_all_features_hpv_group_a.csv'
    path_hpb_b_reduced_results = './../../data_source/results/hpv_splitting/results_all_features_hpv_group_b.csv'

    results = pd.read_csv(path_hpb_a_all_feats_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_figure_hpv_a_all_feats)

    results = pd.read_csv(path_hpb_b_all_feats_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_figure_hpv_b_all_feats)

    results = pd.read_csv(path_hpb_a_reduced_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_figure_hpv_a_reduced)

    results = pd.read_csv(path_hpb_b_reduced_results, index_col=0)

    fig = plt.figure()
    gen_heatmap(results, path_to_fig=path_to_figure_hpv_b_reduced)


def results_statistics():

    # Selected results:
    path_to_baseline_clinical_results = './../../data_source/results/original_images/results_full_clinical.csv'
    path_to_baseline_results = './../../data_source/results/original_images/results_all_features_original_images.csv'
    path_to_reduced_results = './../../data_source/results/removed_broken_slices/results_all_features_removed_broken_slices.csv'
    path_to_hassan_scc_mod_comp_results = './../../data_source/results/dropped_corr/results_all_features_original_images_icc_dropped_correlated.csv'
    path_hpb_b_reduced_results = './../../data_source/results/hpv_splitting/results_all_features_hpv_group_b.csv'
    path_hpb_b_clinical_results = './../../data_source/results/hpv_splitting/results_hpv_b_clinical_only.csv'


    def _confidence_ci(var, alpha=5.0):
        # Calculate 1 - alpha CI for the best model score (max statistic).
        np.random.seed(0)
        num_b_samples = np.size(var)
        scores = []
        for _ in range(1000):
        	# Bootstrap sample.
        	sample = np.random.choice(var, size=num_b_samples, replace=True)
        	scores.append(np.mean(sample))
        lower_p = alpha / 2.0
        lower = max(0.0, np.percentile(scores, lower_p))
        upper_p = (100 - alpha) + (alpha / 2.0)
        upper = min(1.0, np.percentile(scores, upper_p))
        return lower, upper


    result_items = {
        'BASELINE-CLINICAL': path_to_baseline_clinical_results,
        'BASELINE': path_to_baseline_results,
        'REDUCED': path_to_reduced_results,
        'REDUNDANCY': path_to_hassan_scc_mod_comp_results,
        'HPV-B-REDUNDANCY': path_hpb_b_reduced_results,
        'HPV-B-CLINICAL': path_hpb_b_clinical_results
    }
    for key, path in result_items.items():

        results = pd.read_csv(path, index_col=0)
        results_mat, selector_lbls, estimator_lbls = gen_results_matrix(results)
        x, y = np.squeeze(np.where(results_mat == np.max(results_mat)))

        print(key, '\n', '-' * 20)
        print('min', np.min(results_mat))
        print('max', np.max(results_mat))
        print('mean', np.mean(results_mat))
        print('std', np.std(results_mat))
        print('best model:', selector_lbls[y], estimator_lbls[x])
        best_model_lbl = ('_').join((selector_lbls[y], estimator_lbls[x]))
        best_model = results[results['experiment_id'] == best_model_lbl]
        print('CI', _confidence_ci(best_model['test_score']))
        print()


# TODO: Write p-values to file and store in appendix.
def test_auc_normality():

    path_to_baseline_results = './../../data_source/results/original_images/results_all_features_original_images.csv'
    path_to_reduced_results = './../../data_source/results/removed_broken_slices/results_all_features_removed_broken_slices.csv'
    # TODO:
    # path_to_hassan_mod_comp_results = './../../data_source/results/hassan_original_images/results_all_features_icc.csv'
    path_to_hassan_scc_mod_comp_results = './../../data_source/results/dropped_corr/results_all_features_original_images_icc_dropped_correlated.csv'
    path_hpb_b_reduced_results = './../../data_source/results/hpv_splitting/results_all_features_hpv_group_b.csv'
    path_hpb_b_clinical_results = './../../data_source/results/hpv_splitting/results_hpv_b_clinical_only.csv'

    result_items = {
        path_to_baseline_results: 'FisherScoreSelection_LightGBM',
        path_to_reduced_results: 'FisherScoreSelection_SVCEstimator',
        path_to_hassan_scc_mod_comp_results: 'FisherScoreSelection_LightGBM',
        path_hpb_b_reduced_results: 'MultiSURFSelection_XGBoost',
        path_hpb_b_clinical_results: 'MultiSURFSelection_SVCEstimator'
    }
    for path_to_result, target_model in result_items.items():

        results = pd.read_csv(path_to_result, index_col=0)

        idx = results.loc[:, 'experiment_id'] == target_model
        test_scores = results['test_score'][idx]

        print(target_model, '\n', '-' * 20)

        outcome = {}
        outcome = test_normality(test_scores, verbose=0, report=outcome)
        print(outcome)


if __name__ == '__main__':
    #baseline_clinical()
    #baseline()
    #removed_damaged_slices()
    #reduce_feature_redundancy()
    #hpv_experiments()
    results_statistics()
    #test_auc_normality()
