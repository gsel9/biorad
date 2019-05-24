import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ioutil import sample_paths
from stat_tests import wilcoxon_p_values

from sklearn.preprocessing import StandardScaler

import nrrd
import fig_config as CONFIG

CONFIG.plot_setup()


def gtv_reduction(plot=True, show=False, stats=False):

    path_orig_masks = './../../data_source/images/masks_nrrd'
    path_red_masks = './../../data_source/images/masks_removed_broken_slices_ct_size'

    path_to_files = sample_paths(path_orig_masks, path_red_masks, target_format='nrrd')

    red_volume = []
    for path_to_file in path_to_files:
        orig_mask, _ = nrrd.read(path_to_file['Image'])
        red_mask, _ = nrrd.read(path_to_file['Mask'])
        red_volume.append(np.sum(red_mask) / np.sum(orig_mask))

    red_volume = 100 - (np.array(red_volume) * 100)

    if plot:
        _plot_gtv_reduction(red_volume, show=show)
    if stats:
        _gtv_reduction_statistics(red_volume)


def _plot_gtv_reduction(red_volume, show=True):
    # The fraction of GTV removed by discarding damaged slices.

    path_to_figure = '../damaged_slices/frac_gtv_removed.pdf'

    palette = CONFIG.base_palette(n=4)
    grid = np.arange(len(red_volume))

    plt.figure()
    sns.scatterplot(grid, red_volume)
    markerline, stemlines, baseline = plt.stem(
        grid, red_volume, linefmt='-', markerfmt='o', bottom=0.0
    )
    plt.setp(stemlines, color=palette[3], linewidth=2)
    plt.setp(baseline, color=palette[3], linewidth=0.5)
    plt.setp(markerline, 'markerfacecolor', palette[-3])

    #plt.axhline(y=50, c=palette[-3])

    plt.xlabel('Patient ID')
    plt.ylabel('Reduction in Tumor Volume (%)')

    x_coords = np.linspace(1, len(red_volume), 5)
    x_ticks = [f'{int(tick)}' for tick in x_coords]
    plt.xticks(x_coords, x_ticks)

    y_coords = np.linspace(0.00, 50.0, 6)
    y_ticks = []
    for tick in y_coords:
        if len(str(tick).split('.')[0]) > 1:
            y_ticks.append(f'{tick:.01f}')
        else:
            y_ticks.append(f'{tick:.02f}')

    plt.yticks(y_coords, y_ticks)

    plt.savefig(
        path_to_figure,
        bbox_inches='tight',
        transparent=True,
        dpi=CONFIG.DPI,
    )
    if show:
        plt.show()


def _gtv_reduction_statistics(red_volume):

    _red_volume = np.copy(red_volume)
    _red_volume[red_volume == 0] = np.nan

    print('Num samples', len(red_volume))
    print('Min', np.nanmin(_red_volume))
    print('Mean', np.nanmean(_red_volume))
    print('Max', np.nanmax(_red_volume))


def X_orig_X_red(z_score=False):

    path_to_origial_features = './../../data_source/to_analysis/original_images/all_features_original_images.csv'
    path_to_reduced_features = './../../data_source/to_analysis/removed_broken_slices/all_features_removed_broken_slices.csv'

    X_orig = pd.read_csv(path_to_origial_features, index_col=0)
    X_red = pd.read_csv(path_to_reduced_features, index_col=0)

    if z_score:
        scaler = StandardScaler()
        X_orig_std = scaler.fit_transform(X_orig.values)
        X_red_std = scaler.fit_transform(X_red.values)

        X_orig = pd.DataFrame(X_orig_std, index=X_orig.index, columns=X_orig.columns)
        X_red = pd.DataFrame(X_red_std, index=X_red.index, columns=X_red.columns)

    return X_orig, X_red


def wilxocon_feature_stability():

    save_to_file = False

    def firstorder_texture(X, modal):
        if modal == 'CT':
            _CT = X.filter(regex='CT')
            _CT_fs = _CT.filter(regex='firstorder')
            _CT_text = _CT.drop(_CT_fs.columns, axis=1)
            return _CT_fs, _CT_text

        elif modal == 'PET':
            _PET = X.filter(regex='PET')
            _PET_fs = _PET.filter(regex='firstorder')
            _PET_text = _PET.drop(_PET_fs.columns, axis=1)
            # Skip PET params.
            pet_params = ['PETparam_SUVpeak', 'PETparam_MTV', 'PETparam_TLG']
            _PET_text = _PET_text.drop(pet_params, axis=1)
        return _PET_fs, _PET_text

    X_orig, X_red = X_orig_X_red(z_score=True)

    CT_orig_fs, CT_orig_text = firstorder_texture(X_orig, modal='CT')
    CT_red_fs, CT_red_text = firstorder_texture(X_red, modal='CT')

    PET_orig_fs, PET_orig_text = firstorder_texture(X_orig, modal='PET')
    PET_red_fs, PET_red_text = firstorder_texture(X_red, modal='PET')

    # NOTE: Reduced cohort size.
    CT_orig_fs = CT_orig_fs.loc[CT_red_fs.index, :]
    CT_orig_text = CT_orig_text.loc[CT_red_text.index, :]
    PET_orig_fs = PET_orig_fs.loc[PET_red_fs.index, :]
    PET_orig_text = PET_orig_text.loc[PET_red_text.index, :]

    # NOTE: Less PET FS features than CT features.
    ct_cols_to_keep = []
    pet_cols_to_keep = []
    for ct_col in CT_orig_fs.columns:
        ct_reg = ('_').join(ct_col.split('_')[1:])
        for pet_col in PET_red_fs.columns:
            pet_reg = ('_').join(pet_col.split('_')[1:])
            if ct_reg == pet_reg:
                ct_cols_to_keep.append(ct_col)
                pet_cols_to_keep.append(pet_col)

    CT_orig_fs = CT_orig_fs.loc[:, ct_cols_to_keep]
    CT_red_fs = CT_red_fs.loc[:, ct_cols_to_keep]
    PET_orig_fs = PET_orig_fs.loc[:, pet_cols_to_keep]
    PET_red_fs = PET_red_fs.loc[:, pet_cols_to_keep]

    CT_fs_same, CT_fs_diff, CT_fs_p_value = wilcoxon_p_values(
        CT_orig_fs, CT_red_fs, return_p_value=True
    )
    CT_text_same, CT_text_diff, CT_text_p_value = wilcoxon_p_values(
        CT_orig_text, CT_red_text, return_p_value=True
    )
    print('CT first order same', np.round((1 - len(CT_fs_same) / CT_orig_fs.shape[1]) * 100, 2))
    print('CT first order differed', np.round((1 - len(CT_fs_diff) / CT_orig_fs.shape[1]) * 100, 2))
    print('CT texture same', np.round((1 - len(CT_text_same) / CT_orig_text.shape[1]) * 100, 2))
    print('CT texture differed', np.round((1 - len(CT_text_diff) / CT_orig_text.shape[1]) * 100, 2))

    PET_fs_same, PET_fs_diff, PET_fs_p_value = wilcoxon_p_values(
        PET_orig_fs, PET_red_fs, return_p_value=True
    )
    PET_text_same, PET_text_diff, PET_text_p_value = wilcoxon_p_values(
        PET_orig_text, PET_red_text, return_p_value=True
    )
    print('PET first order same', np.round((1 - len(PET_fs_same) / PET_orig_fs.shape[1]) * 100, 2))
    print('PET first order differed', np.round((1 - len(PET_fs_diff) / PET_orig_fs.shape[1]) * 100, 2))
    print('PET texture same', np.round((1 - len(PET_text_same) / PET_orig_text.shape[1]) * 100, 2))
    print('PET texture differed', np.round((1 - len(PET_text_diff) / PET_orig_text.shape[1]) * 100, 2))

    """
    print(len(CT_fs_p_value))
    print(len(CT_text_p_value))
    print(len(PET_fs_p_value))
    print(len(PET_text_p_value))
    """

    if save_to_file:
        pass


if __name__ == '__main__':
    wilxocon_feature_stability()
