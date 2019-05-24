import numpy as np
import pandas as pd
import altair as alt

import concensus_clustering

from copy import deepcopy

from sklearn.cluster.bicluster import SpectralCoclustering
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1 import colorbar

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.colors import ListedColormap

import fig_config as CONFIG

CONFIG.plot_setup()


def _ticks(ticks):

    output = []
    for tick in ticks:
        if len(str(tick).split('.')[0]) > 1:
            output.append(f'{tick:.01f}')
        else:
            output.append(f'{tick:.02f}')

    return output


feature_categories = [
    'shape',
    'firstorder',
    'glcm',
    'glrlm',
    'glszm',
    'gldm',
    'ngtdm',
    'PETparam',
    'clinical'
]


def category_counts(_):

    return {
        'shape': 0,
        'firstorder': 0,
        'glcm': 0,
        'glrlm': 0,
        'glszm': 0,
        'gldm': 0,
        'ngtdm': 0,
        'PETparam': 0,
        'clinical': 0,
    }


def _update_count(pet_output, ct_output, key):

    if 'PET' in key:
        pet_output[key] += 1
    else:
        ct_output[key] += 1

    return pet_output, ct_output


def _norm_count(pet_output, ct_output, key, tot_counts):

    if 'PET' in key:
        pet_output[key] /= tot_counts[key]
    else:
        ct_output[key] /= tot_counts[key]

    return pet_output, ct_output


def to_feature_categories(cluster_indices, X):

    pet_output = category_counts(None)
    ct_output = category_counts(None)


    for label in X.columns[cluster_indices]:
        if 'shape' in label:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'shape')
        elif 'firstorder' in label:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'firstorder')
        elif 'glcm' in label:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'glcm')
        elif 'glrlm' in label:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'glrlm')
        elif 'glszm' in label:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'glszm')
        elif 'gldm' in label:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'gldm')
        elif 'ngtdm' in label:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'ngtdm')
        elif 'PETparam' in label:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'PETparam')
        else:
            pet_output, ct_output = _update_count(pet_output, ct_output, 'clinical')

    feature_counts = {
        'shape': 14,
        'firstorder': 46,
        'glcm': 132,
        'glrlm': 84,
        'glszm': 84,
        'gldm': 78,
        'ngtdm': 30,
        'PETparam': 3,
        'clinical': 42
    }
    for label in feature_counts.keys():
        pet_output, ct_output = _norm_count(pet_output, ct_output, label, feature_counts)

    return pet_output, ct_output


def biclusters(model, X, param_config):
    # Create Bicluster instances tracking detected clusters.

    # Start fresh with each clustering.
    _model = deepcopy(model)

    # Set number of clusters to detect and fit model to data.
    _model.set_params(**param_config)
    _model.fit(X)

    rows, cols = _model.rows_, _model.columns_
    # Sanity check.
    assert np.shape(rows)[0] == np.shape(cols)[0]

    biclusters = concensus_clustering.Biclusters(
        rows=rows, cols=cols, data=X
    )
    return biclusters


def bic_coords(model, num_clusters):
    # Collect coordinates for block diagonal biclusters.

    coords = pd.DataFrame(
        np.zeros((num_clusters, 4)),
        columns=('y1', 'y2', 'x1', 'x2')
    )
    prev_rows, prev_cols = 0, 0
    for num, row_bic in enumerate(model.rows_):
        num_rows = np.sum(row_bic)
        num_cols = np.sum(model.columns_[num])

        coords.iloc[num, 0] = prev_rows
        coords.iloc[num, 1] = prev_rows + num_rows
        coords.iloc[num, 2] = prev_cols
        coords.iloc[num, 3] = prev_cols + num_cols

        prev_rows += num_rows
        prev_cols += num_cols

    return coords


def format_feature_labels(labels):
    """Process raw feature labels."""
    prep_labels = []
    for label in labels:
        if 'shape' in label:
            prep_labels.append('Shape')
        elif 'PETparam' in label:
            prep_labels.append('PET Parameter')
        elif 'clinical' in label:
            prep_labels.append('Clinical')
        elif 'firstorder' in label:
            prep_labels.append('First Order')
        elif 'glcm' in label:
            prep_labels.append('GLCM')
        elif 'gldm' in label:
            prep_labels.append('GLDM')
        elif 'glrlm' in label:
            prep_labels.append('GLRLM')
        elif 'glszm' in label:
            prep_labels.append('GLSZM')
        elif 'ngtdm' in label:
            prep_labels.append('NGTDM')
        else:
            raise ValueError(f'Unknown label {label}')
    return prep_labels



def plot_biclusters():

    co_grid = ParameterGrid(
        {'n_clusters': np.arange(2, 10, 1), 'n_init': [20]}
    )
    _y = pd.read_csv('./../../data_source/to_analysis/original_images/dfs_original_images.csv', index_col=0)
    y_orig = np.squeeze(_y.values)

    X_orig = pd.read_csv('./../../data_source/to_analysis/original_images/all_features_original_images.csv', index_col=0)

    scaler = StandardScaler()
    X_orig_std = scaler.fit_transform(X_orig.values)

    #_run_experiment(co_grid, X_orig_std)

    df_avg_co_scores = pd.read_csv('bic_scores.csv', index_col=0)
    best_co_config = co_grid[
        np.argmin(df_avg_co_scores.loc[:, 'tvr'].values) - 1
    ]
    print(best_co_config, min(df_avg_co_scores.loc[:, 'tvr'].values))

    orig_co_model = SpectralCoclustering(random_state=0, svd_method='arpack')
    orig_co_model.set_params(**best_co_config)
    orig_co_model.fit(X_orig_std)

    #plt.figure()
    #_plot_tve(df_avg_co_scores, co_grid)

    plt.figure()
    _plot_bicmaps(X_orig_std, best_co_config)

    #plt.figure()
    #_plot_column_bics(orig_co_model, X_orig)

    #plt.figure()
    #_plot_row_bics(orig_co_model, y_orig)


def _run_experiment(co_grid, X_orig_std):

    np.random.seed(seed=0)
    random_states = np.random.choice(40, size=40)

    avg_co_scores = {}
    for num, co_param_config in enumerate(co_grid):
        orig_co_scores = []
        for random_state in random_states:
            orig_co_model = SpectralCoclustering(random_state=random_state, svd_method='arpack')
            # NOTE: Outputs a TVE score.
            orig_co_clusters = biclusters(orig_co_model, X_orig_std, co_param_config)
            orig_co_scores.append(orig_co_clusters.external_metrics.values)
        avg_co_scores[num] = np.nanmean(orig_co_scores, axis=0)

    avg_orig_co_scores = []
    for num, scores in enumerate(avg_co_scores.values()):
        avg_orig_co_scores.append(np.mean(scores, axis=0))

    df_avg_co_scores = pd.DataFrame(avg_orig_co_scores, columns=['tvr'])
    df_avg_co_scores.index.name = 'ConfigID'

    df_avg_co_scores.to_csv('bic_scores.csv')


def _plot_bicmaps(X_orig_std, best_co_config):

    # Train model with best config.
    orig_co_model = SpectralCoclustering(random_state=0, svd_method='arpack')
    orig_co_model.set_params(**best_co_config)
    orig_co_model.fit(X_orig_std)
    orig_co_row_sorted = X_orig_std[np.argsort(orig_co_model.row_labels_), :]
    orig_co_fit_data = orig_co_row_sorted[:, np.argsort(orig_co_model.column_labels_)]

    hmap = sns.heatmap(
        orig_co_fit_data,
        robust=True,
        cmap=plt.cm.viridis,
        fmt='f',
        vmin=np.min(orig_co_fit_data),
        vmax=np.max(orig_co_fit_data),
        cbar=False
    )
    coords = bic_coords(orig_co_model, best_co_config['n_clusters'])
    for num in coords.index:
        plt.plot(
            (coords.loc[num, ['x1', 'x2', 'x2', 'x1', 'x1']]),
            (coords.loc[num, ['y1', 'y1', 'y2', 'y2', 'y1']]),
            c='darkred'
    )
    plt.ylabel('Patients')
    plt.xlabel('Features')

    plt.yticks([], [])
    plt.xticks([], [])

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
        np.nanmin(orig_co_fit_data),
        np.nanmax(orig_co_fit_data),
        6
    )
    cax.yaxis.set_ticks(cbar_ticks)
    cax.yaxis.set_ticklabels([f'{num:.01f}' for num in cbar_ticks])

    plt.savefig(
        '../biclustering/bic_map_original_images.pdf',
        bbox_inches='tight',
        transparent=True,
        dpi=CONFIG.DPI,
    )


def _plot_tve(df_avg_co_scores, co_grid):

    color = CONFIG.base_palette(n=1)

    y_coords = df_avg_co_scores.loc[:, 'tvr']
    x_coords = y_coords.index
    plt.plot(
        x_coords, y_coords.values, color='blue', marker='o', linestyle=''
    )
    y_coords = np.linspace(
        np.min(df_avg_co_scores.loc[:, 'tvr']),
        np.max(df_avg_co_scores.loc[:, 'tvr']),
        6
    )
    y_ticks = [f'{tick:.02f}' for tick in df_avg_co_scores.loc[:, 'tvr']]
    plt.yticks(y_coords, y_ticks)

    plt.xticks(np.arange(len(co_grid)), np.arange(1, 9, 1, dtype=int))

    plt.ylabel('Transposed Virtual Error')
    plt.xlabel('Number of Biclusters')

    plt.savefig(
        '../biclustering/tve_by_param_config.pdf',
        bbox_inches='tight',
        dpi=CONFIG.DPI,
    )


def _row_bics(orig_co_model, y_orig):

    # Collect row cluster info.
    orig_row_idx = []
    orig_pfs_outcome, orig_not_pfs_outcome = [], []
    for bic_row_idx in np.unique(orig_co_model.row_labels_):
        # Store cluster index and ID samples belonging to current cluster.
        orig_row_idx.append(bic_row_idx)
        row_cluster_samples = np.where(orig_co_model.row_labels_ == bic_row_idx)

        # Store fractions of each outcome for current cluster.
        orig_pfs_outcome.append(sum(y_orig[row_cluster_samples] == 0) / np.size(y_orig))
        orig_not_pfs_outcome.append(sum(y_orig[row_cluster_samples] == 1) / np.size(y_orig))

    orig_sorted_cluster_idx = np.concatenate((orig_row_idx, orig_row_idx))
    orig_comb_results = np.concatenate((orig_pfs_outcome, orig_not_pfs_outcome))

    orig_results_id =  np.concatenate((
        ['Disease-Free Survival'] * len(orig_pfs_outcome),
        ['Other Event'] * len(orig_not_pfs_outcome)
    ))
    df_orig_row_clusters = pd.DataFrame(
        {'comb_results': orig_comb_results, 'results_id': orig_results_id},
        index=orig_sorted_cluster_idx,
    )
    return df_orig_row_clusters



def _column_bics(orig_co_model, X_orig):

    # Collect column cluster info.
    orig_column_clusters = {}
    for co_col_idx in np.unique(orig_co_model.column_labels_):
        # ID samples belonging to current cluster.
        col_cluster_samples = np.squeeze(np.where(orig_co_model.column_labels_ == co_col_idx))
        # Store fractions of present feature categories per modality.
        pet_output, ct_output = to_feature_categories(col_cluster_samples, X_orig)
        orig_column_clusters[co_col_idx] = {
            key: val_a + val_b
            for (key, val_a), (_, val_b)
            in zip(pet_output.items(), ct_output.items())
        }
    df_orig_column_clusters = pd.DataFrame(orig_column_clusters).T
    df_orig_column_clusters.columns = [
        'PET paremters', 'Clinical', 'First Order', 'GLCM', 'GLDM', 'GLRLM',
        'GLSZM', 'NGTDM', 'Shape'
    ]
    return df_orig_column_clusters


def _plot_column_bics(orig_co_model, X_orig):

    df_orig_column_clusters = _column_bics(orig_co_model, X_orig)

    fig = df_orig_column_clusters.plot(
        kind='bar',
        width=0.9,
        figsize=(15, 9.27),
        colormap=ListedColormap(
                CONFIG.base_palette(n=len(df_orig_column_clusters.columns))
            )
    )
    plt.xlabel('Column Cluster Indicator')
    plt.ylabel('Feature Category (%)')

    n = 3
    bar_width = 0.1
    # Adjust one strawling shitty bar.
    for bar_num, bar in enumerate(fig.patches):
        if bar_num == 26:
            bar.set_x(bar.get_x() - 0.1)
    """
        else:
            bar.set_x(bar.get_x() - 0.09)
        # NB:
        plt.setp(bar, width=bar_width)
    """

    plt.legend(
        loc='center right', bbox_to_anchor=(0.25, 0.75), ncol=1, fancybox=True,
        shadow=True
    )
    plt.xticks(np.arange(n), np.arange(1, n + 1), rotation=0)
    plt.yticks(np.linspace(0.0, 0.8, 6), _ticks(np.linspace(0.0, 80, 6)))

    plt.savefig(
        '../biclustering/column_bics.pdf', bbox_inches='tight',
        dpi=CONFIG.DPI,
    )



def _plot_row_bics(orig_co_model, y_orig):

    df_orig_row_clusters = _row_bics(orig_co_model, y_orig)
    colors = CONFIG.base_palette(n=2)

    data = {}
    for num, group in enumerate(['Disease-Free Survival', 'Other Event']):
        y = df_orig_row_clusters.loc[df_orig_row_clusters.loc[:, 'results_id'] == group, :]
        y = np.squeeze(y.loc[:, 'comb_results'].values)
        data[group] = y


    df = pd.DataFrame(data, index=np.arange(1, 4, 1))
    df.plot(
        kind='barh',
        figsize=(15, 9.27),
        colormap='viridis'
    )

    plt.ylabel('Row Cluster Indicator')
    plt.xlabel('Treatment outcome (%)')

    plt.savefig(
        '../biclustering/row_bics.pdf',
        bbox_inches='tight',
        dpi=CONFIG.DPI,
    )



def tve_stats():

    tve = pd.read_csv('bic_scores.csv', index_col=0)
    print(tve.mean())
    print(tve.std())
    print(tve)



if __name__ == '__main__':
    plot_biclusters()
