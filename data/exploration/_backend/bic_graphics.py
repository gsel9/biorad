# -*- coding: utf-8 -*-
#
# bic_graphics.py
#

"""
Apply a biclustering algorithm to predict biclusters from data.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def _checker_coords(model, num_clusters):
    # Collect coordinates for biclusters with a checkerborad structure.
    tot_num_clusters = num_clusters[0] * num_clusters[1]
    coords = pd.DataFrame(
        np.zeros((tot_num_clusters, 4)),
        columns=('y1', 'y2', 'x1', 'x2')
    )
    num, prev_rows = 0, 0
    for row_num in range(num_clusters[0]):
        nrows = np.sum(model.rows_[row_num])

        prev_cols = 0
        for col_num in range(num_clusters[1]):
            ncols = np.sum(model.columns_[col_num])

            coords.iloc[num, 0] = prev_rows + 1
            coords.iloc[num, 1] = prev_rows + nrows
            coords.iloc[num, 2] = prev_cols
            coords.iloc[num, 3] = prev_cols + ncols

            num += 1

            prev_cols += ncols
        prev_rows += nrows - 1

    return coords


def _bic_coords(model, num_clusters):
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


def plot_biclusters(model, data, num_clusters, path_to_fig=None):
    """Creates a heatmap with bounding boxes around biclusters.

    Args:
        model (sklearn.bicluster): The biclustering algorithm applied to
            data.
        data (array-like): The raw data.
        num_clusters (int or tuple):
        path_to_fig (str): Provide a path to a location if wanting to save the
            figure to disk.

    """

    if isinstance(data, pd.DataFrame):
        data = data.values

    if not isinstance(data, np.ndarray):
        raise TypeError('Input data should be <numpy.ndarray>, and not {}'
                        ''.format(type(data)))

    # Reshuffle data matrix according to predicted biclusters:
    row_sorted = data[np.argsort(model.row_labels_), :]
    fit_data = row_sorted[:, np.argsort(model.column_labels_)]

    # Compute coordinates of biclusters.
    if isinstance(num_clusters, int):
        coords = _bic_coords(model, num_clusters)
    else:
        coords = _checker_coords(model, num_clusters)

    # Generate a heatmap and plot bounding boxes of biclusters.
    fig, (cbar_ax, map_ax) = plt.subplots(
        nrows=2, figsize=(10, 10),
        gridspec_kw={'height_ratios':[0.025, 1]}
    )
    # Draw heatmap
    sns.heatmap(
        fit_data, ax=map_ax, robust=True,
        cmap=plt.cm.RdBu_r, fmt='f',
        vmin=np.min(fit_data),
        vmax=np.max(fit_data),
        cbar=False
    )
    # Add color bar
    fig.colorbar(
        map_ax.get_children()[0],
        cax=cbar_ax,
        orientation='horizontal'
    )
    for num in coords.index:
        plt.plot(
            (coords.loc[num, ['x1', 'x2', 'x2', 'x1', 'x1']]),
            (coords.loc[num, ['y1', 'y1', 'y2', 'y2', 'y1']]),
            linewidth=2, c='orangered' #darkred
        )
    plt.axis('off')
    if path_to_fig is not None:
        plt.savefig(
            path_to_fig,
            transparent=True,
            bbox_inches='tight',
            orientation='landscape'
        )
    return plt
