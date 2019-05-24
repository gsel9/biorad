

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


DPI = 600
ALPHA = 0.7


def plot_setup():

    tick_size = 17
    font_size = 20
    fontlabel_size = 20

    params = {
        'font.family': 'Sans',
        'font.size': font_size,
        'mathtext.fontset': 'stix',
        'backend': 'wxAgg',
        'figure.autolayout': True,
        'figure.figsize': (15, 9.27),
        'lines.markersize': 2,
        'axes.labelsize': fontlabel_size,
        'axes.titlesize': 20,
        #'axes.titlepad': 20,
        'legend.fontsize': fontlabel_size,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        #'text.usetex': True,
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'image.cmap': 'viridis'
    }
    plt.rcParams.update(params)


def feature_categories_from_labels(labels, group_firstorder=True, group_texture=True):

    textures = ['glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']

    new_labels = []
    for label in labels:
        if 'shape' in label:
            new_labels.append('Shape')
        elif 'PETparam' in label:
            new_labels.append('PET Parameter')
        elif 'CT_original_firstorder' in label:
            label = 'First Order' if group_firstorder else 'CT First Order'
            new_labels.append(label)
        elif 'PET_original_firstorder' in label:
            label = 'First Order' if group_firstorder else 'PET First Order'
            new_labels.append(label)
        elif any(f'CT_original_{text}' in label for text in textures):
            label = 'Texture' if group_firstorder else 'CT Texture'
            new_labels.append(label)
        elif any(f'PET_original_{text}' in label for text in textures):
            label = 'Texture' if group_firstorder else 'PET Texture'
            new_labels.append(label)
        else:
            # Clinical.
            new_labels.append('Clinical')

    return new_labels


def patient_axis_ticks():

    path_to_file = './../../data_source/patient_id.npy'
    idx = np.load(path_to_file)

    return idx


# TODO: Get blue, green, orange, red from sns colorblind palette.
def descriptive_stats_colours():
    # http://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3

    return ['#2ecc71', '#34495e', '#3498db', '#e34a33']


def base_palette(n=7):
    base = plt.cm.get_cmap('viridis')
    palette = base(np.linspace(0, 1, n))

    return palette
