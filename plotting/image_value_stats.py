import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import pylab
import nrrd

from sklearn.neighbors import DistanceMetric

from sklearn.cluster import KMeans
from resizeimage import resizeimage
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from scipy import stats
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.cluster import hierarchy
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import StandardScaler

from ioutil import relative_paths
from cycler import cycler
import matplotlib.colors as mcolors
import fig_config as CONFIG
import random

from matplotlib.colors import ListedColormap


CONFIG.plot_setup()


def _ticks(coords):
    output = []
    for tick in coords:
        comps = str(tick).split('.')
        if 1 < len(comps[0]) < 3:
            output.append(f'{tick:.01f}')
        elif len(comps[0]) > 2:
            output.append(f'{tick:.02e}')
        else:
            output.append(f'{tick:.02f}')
    return output


def calc_image_vale_stats(images, patient_id):

    def image_stats(image):

        _image = np.copy(image)
        _image[image == 0] = np.nan

        return {
            'gl_mean': np.nanmean(_image),
            'gl_median': np.nanmedian(_image),
            'gl_min': np.nanmin(_image),
            'gl_max': np.nanmax(_image),
        }

    # Complete image statistics.
    stats = {}
    for num, image in enumerate(images.values()):
        stats[num] = image_stats(image)

    df_stats = pd.DataFrame(stats, columns=patient_id)

    return df_stats


def load_images(path_to_images, path_to_masks=None):

    output = {}
    for num, path_to_image in enumerate(path_to_images):

        image, _ = nrrd.read(path_to_image)

        if path_to_masks is not None:
            mask, _ = nrrd.read(path_to_masks[num])
            image = image * mask

        output[path_to_image] = image
    return output


def plot_img_value_stats(path_to_fig,
                    path_to_images,
                    path_to_masks=None,
                    show=False,
                    include_legend=False):
    """Calculate descriptive statistics of image values."""

    labels = ['Maximum', 'Mean', 'Median', 'Minimum']
    images = load_images(path_to_images, path_to_masks)
    patient_id = CONFIG.patient_axis_ticks()
    df_stats = calc_image_vale_stats(images, patient_id)

    palette = CONFIG.base_palette(n=4)
    # NOTE: linspace should be max value!
    for num, row_label in enumerate(df_stats.T):
        # Plot image stats.
        stat = np.squeeze(df_stats.loc[row_label, :].values)
        plt.scatter(
            x=np.squeeze(df_stats.columns.values),
            y=stat,
            label=labels[num],
            color=palette[num],
            alpha=CONFIG.ALPHA
        )
    if include_legend:
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.1),
            ncol=4,
            fancybox=True,
            shadow=True
        )
    y_coords = np.linspace(0, max(df_stats.max()), 6, dtype=int)
    y_ticks = _ticks(y_coords)
    plt.yticks(y_coords, y_ticks)

    x_coords = np.linspace(1, 198, 6, dtype=int)
    plt.xticks(x_coords, x_coords)

    plt.savefig(path_to_fig, bbox_inches='tight', dpi=400)
    if show:
        plt.show()


def clustering_ct_max_stats(images, patient_id, path_kmeans_distort):

    df_ct_stats = calc_image_vale_stats(images, patient_id)

    _plot_kmeans_distortions(df_ct_stats, path_kmeans_distort)

    """
    sns.scatterplot(
        np.arange(198),
        np.squeeze(df_ct_stats.loc['gl_max', :].values),
        hue=y_pred_kmeans==0,
        legend=False,
        palette=CONFIG.base_palette(n=2)
    )
    plt.ylabel('CT Image Value Maximum')
    plt.xlabel('Patient Indicator')

    plt.axhline(y=gl_max_thresh_kmeans)
    plt.legend(
        [f'Maximum: {int(gl_max_thresh_kmeans)}'],
        fontsize=17,
        loc='upper center', bbox_to_anchor=(0.13, 0.75),
        ncol=1, fancybox=True, shadow=True
    )
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.savefig(
        './../../figures/expl_analysis/kmeans_clustering_ct_max',
        bbox_inches='tight',
        transparent=True,
        dpi=100,
    )
    plt.tight_layout()
    """


def _plot_kmeans_distortions(df_ct_stats, path_to_figure=None):

    clusters = np.arange(1, 21)

    max_stat = df_ct_stats.loc['gl_max', :].dropna()
    max_stat = max_stat.values[:, np.newaxis]

    euc_dist = DistanceMetric.get_metric('euclidean')
    dist_mat = euc_dist.pairwise(max_stat)

    distortions = []
    for cluster in clusters:
        model = KMeans(n_clusters=cluster, random_state=0).fit(dist_mat)
        distortions.append(
            sum(np.min(cdist(dist_mat, model.cluster_centers_, 'euclidean'), axis=1)) / dist_mat.shape[0]
        )
    plt.figure()
    plt.plot(clusters, distortions, marker='o', color='yellow')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')

    #plt.xlim([0.01, 20.01])

    x_coords = np.linspace(1, np.size(clusters), 21, dtype=int)
    y_coords = np.linspace(min(distortions), max(distortions), 6)
    y_ticks = _ticks(y_coords)
    plt.xticks(x_coords, x_coords)
    plt.yticks(y_coords, y_ticks)

    if path_to_figure is not None:
        plt.savefig(path_to_figure, bbox_inches='tight', dpi=CONFIG.DPI,)


def plot_clustering_ct_max_stats():

    path_kmeans_distort = '../image_value_stats/kmeans_elbow.pdf'
    path_to_images = relative_paths('./../../data_source/images/ct_nrrd', target_format='nrrd')

    images = load_images(path_to_images)
    patient_id = CONFIG.patient_axis_ticks()

    clustering_ct_max_stats(images, patient_id, path_kmeans_distort)


def ct_img_value_stats():

    path_to_fig = '../image_value_stats/ct_statistics.pdf'
    path_to_images = relative_paths('./../../data_source/images/ct_nrrd', target_format='nrrd')

    plt.figure()
    plt.xlabel('Patient ID')
    plt.ylabel('CT Intensity Statistic')

    plot_img_value_stats(
        path_to_fig,
        path_to_images,
        include_legend=True
    )


def ct_cropped_img_value_stats():

    path_to_fig = '../image_value_stats/cropped_ct_statistics.pdf'
    path_to_images = relative_paths('./../../data_source/images/ct_nrrd', target_format='nrrd')
    path_to_masks = relative_paths('./../../data_source/images/masks_nrrd', target_format='nrrd')

    plt.figure()
    plt.xlabel('Patient ID')
    plt.ylabel('CT Intensity Statistic')

    plot_img_value_stats(
        path_to_fig,
        path_to_images,
        path_to_masks,
        include_legend=True
    )


def pet_img_value_stats():

    path_to_fig = '../image_value_stats/pet_statistics.pdf'
    path_to_images = relative_paths('./../../data_source/images/pet_nrrd', target_format='nrrd')

    plt.figure()
    plt.xlabel('Patient ID')
    plt.ylabel('PET Intensity Statistic')

    plot_img_value_stats(
        path_to_fig,
        path_to_images,
        include_legend=True
    )


def pet_cropped_img_value_stats():

    path_to_fig = '../image_value_stats/cropped_pet_statistics.pdf'
    path_to_images = relative_paths('./../../data_source/images/pet_nrrd', target_format='nrrd')
    path_to_masks = relative_paths('./../../data_source/images/masks_nrrd', target_format='nrrd')

    plt.figure()
    plt.xlabel('Patient ID')
    plt.ylabel('PET Intensity Statistic')

    plot_img_value_stats(
        path_to_fig,
        path_to_images,
        path_to_masks,
        include_legend=True
    )



if __name__ == '__main__':
    ct_img_value_stats()
    pet_img_value_stats()
    ct_cropped_img_value_stats()
    pet_cropped_img_value_stats()
    plot_clustering_ct_max_stats()
