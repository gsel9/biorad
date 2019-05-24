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


def plot_radiomics_feat_scatter():

    show = False

    path_to_figure = '../expl_analysis/'
    path_to_target = './../../data_source/to_analysis/original_images/dfs_original_images.csv'
    path_to_predictors = './../../data_source/to_analysis/original_images/all_features_original_images.csv'

    y = np.squeeze(pd.read_csv(path_to_target, index_col=0).values)
    X = pd.read_csv(path_to_predictors, index_col=0)

    X_shape = X.filter(regex='shape')
    X_PET = X.filter(regex='PET')
    X_CT = X.filter(regex='CT')

    scaler = StandardScaler()
    X_shape_std = scaler.fit_transform(X_shape)
    X_PET_std = scaler.fit_transform(X_PET)
    X_CT_std = scaler.fit_transform(X_CT)

    ylabels = ['Shape Feature', 'PET Feature', 'CT Feature']
    for num, dset in enumerate([X_shape_std, X_PET_std, X_CT_std]):

        palette = CONFIG.base_palette(n=dset.shape[1])
        x_coords = np.arange(1, np.size(y) + 1, dtype=int)

        plt.figure()
        for col_num, shape_col in enumerate(dset.T):
            plt.scatter(
                x_coords, shape_col, color=palette[col_num]
            )
        plt.xlabel('Patient ID')
        plt.ylabel(f'Z-scored {ylabels[num]} Values')

        x_coords = np.linspace(1, np.size(y), 6, dtype=int)
        y_coords = np.linspace(np.min(dset), np.max(dset), 6)
        plt.xticks(x_coords, x_coords)
        y_ticks = []
        for tick in y_coords:
            comps = str(tick).split('.')[0]
            if '-' in comps[0] and len(comps[0]) == 2:
                y_ticks.append(f'{tick:.02f}')
            elif len(comps[0]) > 1:
                y_ticks.append(f'{tick:.01f}')
            else:
                y_ticks.append(f'{tick:.02f}')
        plt.yticks(y_coords, y_ticks)

        _path_to_figure = f'{path_to_figure}rfeat_vals_{ylabels[num].split( )[0]}.pdf'
        plt.savefig(_path_to_figure, bbox_inches='tight', dpi=CONFIG.DPI)
        if show:
            plt.show()


if __name__ == '__main__':
    plot_radiomics_feat_scatter()
