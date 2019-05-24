"""
TODO:
- Write bin widths to file.
- Calc ICC and check Hassan modifications.

"""


import os
import re
import nrrd

from ioutil import sample_paths

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

import seaborn as sns
import matplotlib.pyplot as plt


import fig_config as CONF

CONF.plot_setup()


# Ng: Number of graylevels.
hassan_gl_transforms = {
    'original_glcm_DifferenceEntropy': lambda Ng, feature: feature / np.log(Ng ** 2),
    'original_glcm_JointEntropy': lambda Ng, feature: feature / np.log(Ng ** 2),
    'original_glcm_SumEntropy': lambda Ng, feature: feature * Ng,
    'original_glcm_Contrast': lambda Ng, feature: feature / (Ng ** 2),
    'original_glcm_DifferenceVariance': lambda Ng, feature: feature / (Ng ** 2),
    'original_glcm_SumAverage': lambda Ng, feature: feature / Ng,
    'original_glcm_DifferenceAverage': lambda Ng, feature: feature / Ng,
    'original_glrlm_GrayLevelNonUniformity': lambda Ng, feature: feature * Ng,
    'original_glrlm_HighGrayLevelRunEmphasis': lambda Ng, feature: feature / (Ng ** 2),
    'original_glrlm_ShortRunHighGrayLevelEmphasis': lambda Ng, feature: feature / (Ng ** 2),
    'original_ngtdm_Contrast': lambda Ng, feature: feature / Ng,
    'original_ngtdm_Complexity': lambda Ng, feature: feature / (Ng ** 3),
    'original_ngtdm_Strength': lambda Ng, feature: feature / (Ng ** 2),
}

# Nv: Number of voxels in ROI.
hassan_roi_transforms = {
    'original_firstorder_Energy': lambda Nv, feature: feature * Nv,
    'original_firstorder_Entropy': lambda Nv, feature: feature / np.log(Nv),
    'original_firstorder_TotalEnergy': lambda Nv, feature: feature / Nv,
    'original_glcm_Contrast': lambda Nv, feature: feature / Nv,
    'original_glcm_InverseVariance': lambda Nv, feature: feature / Nv,
    'original_glcm_JointAverage': lambda Nv, feature: feature / Nv,
    'original_glrlm_GrayLevelNonUniformity': lambda Nv, feature: feature / Nv,
    'original_ngtdm_Coarsness': lambda Nv, feature: feature * Nv,
    'original_ngtdm_Strength': lambda Nv, feature: feature * Nv,
}


def get_palette_colour(label):

    palette = CONF.base_palette(n=9)

    mapping = {
        'Shape': palette[0],
        'Clinical': palette[1],
        'First Order': palette[2],
        'GLCM': palette[3],
        'GLDM': palette[4],
        'GLRLM': palette[5],
        'GLSZM': palette[6],
        'NGTDM': palette[7],
        'PET Parameter': palette[8],
    }
    return mapping[label]


def to_feature_categories(labels):
    """Process raw feature labels."""
    prep_labels = []
    for label in labels:
        if 'shape' in label:
            prep_labels.append('Shape')
        elif 'PETparam' in label:
            prep_labels.append('PET Parameter')
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
            prep_labels.append('Clinical')
    return prep_labels


def get_graylevels(mod='ct', kind='firstorder'):

    if mod == 'ct':
        if kind == 'firstorder':
            path_to_gl32 = 'graylevel_bins/ct_firstorder_gl32_bins.npy'
            path_to_gl64 = 'graylevel_bins/ct_firstorder_gl64_bins.npy'
            path_to_gl128 = 'graylevel_bins/ct_firstorder_gl128_bins.npy'
        else:
            path_to_gl32 = 'graylevel_bins/ct_firstorder_gl32_bins.npy'
            path_to_gl64 = 'graylevel_bins/ct_firstorder_gl64_bins.npy'
            path_to_gl128 = 'graylevel_bins/ct_firstorder_gl128_bins.npy'

    else:
        if kind == 'firstorder':
            path_to_gl32 = 'graylevel_bins/pet_firstorder_gl32_bins.npy'
            path_to_gl64 = 'graylevel_bins/pet_firstorder_gl64_bins.npy'
            path_to_gl128 = 'graylevel_bins/pet_firstorder_gl128_bins.npy'
        else:
            path_to_gl32 = 'graylevel_bins/pet_texture_gl32_bins.npy'
            path_to_gl64 = 'graylevel_bins/pet_texture_gl64_bins.npy'
            path_to_gl128 = 'graylevel_bins/pet_texture_gl128_bins.npy'

    output = (
        np.load(path_to_gl32), np.load(path_to_gl64), np.load(path_to_gl128)
    )
    return output


def transform(gl=True, voxel=True):

    path_to_features = '../../data_source/to_analysis/removed_broken_slices/all_features_removed_broken_slices.csv'
    X = pd.read_csv(X, index_col=0)
    print(X.shape)
    if gl:
        # TODO: Update with artifact removed images.
        gl_bins = gen_graylevels()
        for num, (key, transform) in enumerate(hassan_gl_transforms.items()):

            # Transform features.
            feats = X.filter(regex=key)
            X_transf = np.zeros_like(feats)
            for bin_num, (col, nbins) in enumerate(zip(feats.columns, gl_bins)):
                X_transf[:, bin_num] = transform(nbins, feats.loc[:, col].values)

            # Calculate ICC score and replace features.
            if icc(X_transf) >= 0.8:
                X.drop(feats.columns, axis=1, inplace=True)
                new_label = format_feature_labels(feats.columns)[0]
                X.iloc[:, new_label] = np.mean(X_transf[:, bin_num], axis=1)

    print(X.shape)
    if voxel:
        num_voxels = gen_num_voxels()
        for num, (key, transform) in enumerate(hassan_roi_transforms.items()):

            # Transform features.
            feats = X.filter(regex=key)
            X_transf = np.zeros_like(feats)
            for bin_num, (col, nvoxels) in enumerate(zip(feats.columns, gl_bins)):
                X_transf[:, bin_num] = transform(nbins, feats.loc[:, col].values)

            # Calculate ICC score and replace features.
            if abs(scc(X_transf, tumor_volume_feat)) > 0.5:
                X.drop(feats.columns, axis=1, inplace=True)
                new_label = format_feature_labels(feats.columns)[0]
                X.iloc[:, new_label] = np.mean(X_transf[:, bin_num], axis=1)



def gen_num_voxels():
    pass


def gen_graylevels(mode='orig'):

    # CT:
    if mode == 'orig':
        path_to_ct = './../../data_source/images/ct_nrrd/'
        path_to_ct_masks = './../../data_source/images/masks_nrrd/'
        ct_bin_widths = {
            'firstorder32': 41.41414141414142, #39.07286096256684, #
            'firstorder64': 20.70707070707071, #19.53643048128342, #
            'firstorder128': 10.353535353535355, #9.76821524064171, #
            'texture32': 0.1351010101010101, #0.1303475935828877, #
            'texture64': 0.06755050505050506, #0.06517379679144385, #
            'texture128': 0.03377525252525253, #0.032586898395721924 #
        }
        pet_bin_widths = {
            'firstorder32': 0.40561868686868685, #0.40240641711229946, #
            'firstorder64': 0.20280934343434343, #0.20120320855614973, #
            'firstorder128': 0.10140467171717171, #0.10060160427807487, #
            'texture32': 0.40561868686868685, #0.40240641711229946, #
            'texture64': 0.20280934343434343, #0.20120320855614973, #
            'texture128': 0.10140467171717171, #0.10060160427807487 #
        }
        path_to_pet = './../../data_source/images/pet_nrrd/'
        path_to_pet_masks = './../../data_source/images/masks_nrrd/'
    else:
        path_to_ct = './../../data_source/images/ct_removed_broken_slices/'
        path_to_ct_masks = './../../data_source/images/masks_removed_broken_slices_ct_size/'
        ct_bin_widths = {
            'firstorder32': 39.07286096256684,
            'firstorder64': 19.53643048128342,
            'firstorder128': 9.76821524064171,
            'texture32': 0.1303475935828877,
            'texture64': 0.06517379679144385,
            'texture128': 0.032586898395721924
        }
        pet_bin_widths = {
            'firstorder32': 0.40240641711229946, #
            'firstorder64': 0.20120320855614973, #
            'firstorder128': 0.10060160427807487, #
            'texture32': 0.40240641711229946, #
            'texture64': 0.20120320855614973, #
            'texture128': 0.10060160427807487 #
        }
        path_to_pet = './../../data_source/images/pet_removed_broken_slices/'
        path_to_pet_masks = './../../data_source/images/masks_removed_broken_slices_pet_size/'

    # First Order.
    path_to_ct_gl32 = 'graylevel_bins/ct_firstorder_gl32_bins'
    path_to_ct_gl64 = 'graylevel_bins/ct_firstorder_gl64_bins'
    path_to_ct_gl128 = 'graylevel_bins/ct_firstorder_gl128_bins'
    _get_graylevels(
        path_to_ct,
        path_to_ct_masks,
        width32=ct_bin_widths['firstorder32'],
        width64=ct_bin_widths['firstorder64'],
        width128=ct_bin_widths['firstorder128'],
        path_to_gl32=path_to_ct_gl32,
        path_to_gl64=path_to_ct_gl64,
        path_to_gl128=path_to_ct_gl128
    )
    # Texture.
    path_to_ct_gl32 = 'graylevel_bins/ct_texture_gl32_bins'
    path_to_ct_gl64 = 'graylevel_bins/ct_texture_gl64_bins'
    path_to_ct_gl128 = 'graylevel_bins/ct_texture_gl128_bins'
    _get_graylevels(
        path_to_ct,
        path_to_ct_masks,
        width32=ct_bin_widths['texture32'],
        width64=ct_bin_widths['texture64'],
        width128=ct_bin_widths['texture128'],
        path_to_gl32=path_to_ct_gl32,
        path_to_gl64=path_to_ct_gl64,
        path_to_gl128=path_to_ct_gl128,
        z_scoring=True
    )
    # First Order.
    path_to_pet_gl32 = 'graylevel_bins/pet_firstorder_gl32_bins'
    path_to_pet_gl64 = 'graylevel_bins/pet_firstorder_gl64_bins'
    path_to_pet_gl128 = 'graylevel_bins/pet_firstorder_gl128_bins'
    _get_graylevels(
        path_to_pet,
        path_to_pet_masks,
        width32=pet_bin_widths['firstorder32'],
        width64=pet_bin_widths['firstorder64'],
        width128=pet_bin_widths['firstorder128'],
        path_to_gl32=path_to_pet_gl32,
        path_to_gl64=path_to_pet_gl64,
        path_to_gl128=path_to_pet_gl128
    )
    # Texture.
    path_to_pet_gl32 = 'graylevel_bins/pet_texture_gl32_bins'
    path_to_pet_gl64 = 'graylevel_bins/pet_texture_gl64_bins'
    path_to_pet_gl128 = 'graylevel_bins/pet_texture_gl128_bins'
    _get_graylevels(
        path_to_pet,
        path_to_pet_masks,
        width32=pet_bin_widths['texture32'],
        width64=pet_bin_widths['texture64'],
        width128=pet_bin_widths['texture128'],
        path_to_gl32=path_to_pet_gl32,
        path_to_gl64=path_to_pet_gl64,
        path_to_gl128=path_to_pet_gl128,
    )


def _get_graylevels(path_to_images,
                    path_to_masks,
                    width32,
                    width64,
                    width128,
                    path_to_gl32,
                    path_to_gl64,
                    path_to_gl128,
                    z_scoring=False):

    path_to_stacks = sample_paths(
        path_to_images, path_to_masks, target_format='nrrd'
    )

    idx = []
    gl_32bins = np.zeros(len(path_to_stacks))
    gl_64bins = np.zeros(len(path_to_stacks))
    gl_128bins = np.zeros(len(path_to_stacks))
    for num, path_to_stack in enumerate(path_to_stacks):

        fname = os.path.basename(path_to_stack['Image'])
        idx_num = re.findall(r'\d+', fname.split('.')[0])[0]
        idx.append(int(idx_num))

        image, _ = nrrd.read(path_to_stack['Image'])
        mask, _ = nrrd.read(path_to_stack['Mask'])

        if z_scoring:
            image = (image - np.mean(image)) / (np.std(image) + 1e-12)

        cropped = image * mask
        data = cropped.ravel()

        # Binning operation as conducted in PyRadiomics.
        minimum = min(data)
        maximum = max(data)

        low_32_bound = minimum - (minimum % width32)
        low_64_bound = minimum - (minimum % width64)
        low_128_bound = minimum - (minimum % width128)

        high_32_bound = maximum + 2 * width32
        high_64_bound = maximum + 2 * width64
        high_128_bound = maximum + 2 * width128

        bin_32_edges = np.arange(low_32_bound, high_32_bound, width32)
        bin_64_edges = np.arange(low_64_bound, high_64_bound, width64)
        bin_128_edges = np.arange(low_128_bound, high_128_bound, width128)

        # Number of graylevels.
        gl_32bins[num] = np.size(bin_32_edges)
        gl_64bins[num] = np.size(bin_64_edges)
        gl_128bins[num] = np.size(bin_128_edges)

    # num graylevels x images.
    np.save(path_to_gl32, gl_32bins)
    np.save(path_to_gl64, gl_64bins)
    np.save(path_to_gl128, gl_128bins)



def icc(Y):
    """Calculate intra-class correlation coefficient (ICC).

    Reference:
        Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
        assessing rater reliability. Psychological bulletin, 86(2), 420.

    Args:
        X (array-like): Data matrix with observations on rows
            and measurements on columns.

    Returns:
        (float): Intraclass correlation coefficient.

    """
    n, k = np.shape(Y)

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k-1)
    dfr = n - 1

    # Sum Square Total
    Y_avg = np.mean(Y)
    SST = np.sum((Y - Y_avg) ** 2)

    # Create the design matrix for the different levels:
    # * Sessions:
    x = np.kron(np.eye(k), np.ones((n, 1)))
    # * Subjects:
    x0 = np.tile(np.eye(n), (k, 1))
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten('F')
    )
    residuals = Y.flatten('F') - predicted_Y
    SSE = np.sum(residuals ** 2)

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = np.sum((np.mean(Y, axis=0) - Y_avg) ** 2) * n
    MSC = SSC / dfc / n

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subject - mean square error) /
    # (mean square subject + (k-1)*mean square error)
    ICC = (MSR - MSE) / (MSR + (k-1) * MSE)

    return ICC


def icc_from_hassan_modified(X, gl_bins):
    """Apply Hassan transform and record ICC for original and
    transformed features.

    X (pandas.DataFrame):
    gl_bins (array-like):

    Returns:
        (pandas.DataFrame):

    """
    icc_orig_feat = np.zeros(len(hassan_gl_transforms.keys()))
    icc_norm_feat = np.zeros(len(hassan_gl_transforms.keys()))
    for num, (key, transform) in enumerate(hassan_gl_transforms.items()):

        feats = X.filter(regex=key)
        #print(feats.head())
        X_transf = np.zeros_like(feats)
        for bin_num, (col, nbins) in enumerate(zip(feats.columns, gl_bins)):
            X_transf[:, bin_num] = transform(nbins, feats.loc[:, col].values)

        icc_orig_feat[num] = icc(feats.values)
        icc_norm_feat[num] = icc(X_transf)

    df_icc_orig_feat = pd.DataFrame(
        icc_orig_feat, index=hassan_gl_transforms.keys(), columns=['Score']
    )
    df_icc_orig_feat['Kind'] = ['Original'] * len(hassan_gl_transforms.keys())

    df_icc_norm_feat = pd.DataFrame(
        icc_norm_feat, index=hassan_gl_transforms.keys(), columns=['Score']
    )
    df_icc_norm_feat['Kind'] = ['Modified'] * len(hassan_gl_transforms.keys())
    df_icc = pd.concat((df_icc_orig_feat, df_icc_norm_feat), axis=0)

    df_icc.sort_values(by=['Score'])

    return df_icc


def format_feature_labels(labels):
    """Process raw feature labels."""

    prep_labels = []
    for label in labels:
        comps = label.split('_')
        filter_type, feature_type, name = comps
        if name == 'GrayLevelNonUniformity':
            name = 'GLNU'
        elif name == 'HighGrayLevelRunEmphasis':
            name = 'HGLRE'
        elif name == 'ShortRunHighGrayLevelEmphasis':
            name = 'SRHGLE'
        new_label = f'{feature_type.upper()} {name}'

        prep_labels.append(new_label)

    return prep_labels


def gen_num_roi_voxels():
    pass


# TODO: Print to latex.
def plot_hassan_mod():
    # Applies to CT texture features originally. Extending to PET.

    #path_to_features = '../../data_source/to_analysis/removed_broken_slices/all_features_removed_broken_slices.csv'
    path_to_features = '../../data_source/to_analysis/original_images/all_features_original_images.csv'
    gen_graylevels(mode='ase')

    modal = 'ct'
    show = False

    if modal == 'pet':
        #path_to_figure = './../feature_redundancy/removed_broken_pet_hassan_modifications.pdf'
        path_to_figure = './../feature_redundancy/pet_hassan_modifications.pdf'
    else:
        #path_to_figure = './../feature_redundancy/removed_broken_ct_hassan_modifications.pdf'
        path_to_figure = './../feature_redundancy/ct_hassan_modifications.pdf'

    X = pd.read_csv(path_to_features, index_col=0)

    ct_text_bins32, ct_text_bins64, ct_text_bins128 = get_graylevels(mod='ct', kind='texture')
    pet_text_bins32, pet_text_bins64, pet_text_bins128 = get_graylevels(mod='pet', kind='texture')
    ct_fs_bins32, ct_fs_bins64, ct_fs_bins128 = get_graylevels(mod='ct', kind='firstorder')
    pet_fst_bins32, pet_fs_bins64, pet_fs_bins128 = get_graylevels(mod='pet', kind='firstorder')

    CT = X.filter(regex='CT')
    PET = X.filter(regex='PET')

    CT_fs = CT.filter(regex='firstorder')
    PET_fs = PET.filter(regex='firstorder')

    CT_text = CT.drop(CT_fs.columns, axis=1, inplace=False)
    PET.drop(PET_fs.columns, axis=1, inplace=True)
    PET_params = ['PETparam_SUVpeak', 'PETparam_MTV', 'PETparam_TLG']
    PET_text = PET.drop(PET_params, axis=1, inplace=False)

    #print(PET_text.head())
    #print(CT_text.head())

    if modal == 'ct':
        gl_bins = np.array([ct_text_bins32, ct_text_bins64, ct_text_bins128])
        df_icc = icc_from_hassan_modified(CT_text, gl_bins)
        print(df_icc)
    else:
        gl_bins = np.array([ct_text_bins32, ct_text_bins64, ct_text_bins128])
        df_icc = icc_from_hassan_modified(PET_text, gl_bins)
        print(df_icc)

    plt.figure()
    fig = sns.barplot(
        x=df_icc.index,
        y='Score',
        hue='Kind',
        data=df_icc,
        palette=CONF.base_palette(n=6)
    )
    for patch_num, patch in enumerate(fig.patches):
        current_width = patch.get_width()
        diff = current_width - 0.3
        patch.set_width(0.3)
        patch.set_x(patch.get_x() + diff * 0.5)

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        fancybox=True,
        shadow=True
    )
    labels = np.unique(format_feature_labels(hassan_gl_transforms.keys()))

    x_coords = np.arange(len(labels))
    fig.set_xticks(x_coords)
    fig.set_xticklabels(labels, rotation=30, ha='right')

    y_coords = np.linspace(0.0, 1.0, 6)
    y_ticks = [f'{tick:.02f}' for tick in y_coords]
    fig.set_yticks(y_coords)
    fig.set_yticklabels(y_ticks)
    plt.ylabel('Intraclass Correlation Coefficient')

    plt.savefig(path_to_figure, bbox_inches='tight', dpi=CONF.DPI)
    if show:
        plt.show()


def plot_scc():
    # Plot the SCC of all original features.

    show = False
    scc_thresh = False

    if scc_thresh:
        path_to_figure = './../feature_redundancy/thresh_max_feature_scc.pdf'
        path_to_features = '../../data_source/to_analysis/compressed_features/all_features_orig_images_icc.csv'
    else:
        path_to_figure = './../feature_redundancy/max_feature_scc.pdf'
        path_to_features = '../../data_source/to_analysis/original_images/all_features_original_images.csv'

    _X = pd.read_csv(path_to_features, index_col=0)

    # Drop categorical clinical variables.
    clinical_to_drop = []
    for col in _X:
        if len(np.unique(_X.loc[:, col])) < 3:
            clinical_to_drop.append(col)
    _X.drop(clinical_to_drop, axis=1, inplace=True)

    # Use Z-scored matrix since classifiers are applied to Z-scored data.
    scaler = StandardScaler()
    X = scaler.fit_transform(_X)
    X = pd.DataFrame(X, columns=_X.columns, index=_X.index)

    corr_matrix = X.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    if scc_thresh:
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        print(f'Dropping {len(to_drop)} features with SCC >= 0.95 out of {X.shape[1]}.')
        X = X.drop(to_drop, axis=1)
        upper = X.corr(method='spearman').abs()
        upper = upper.where(np.triu(np.ones(upper.shape), k=1).astype(np.bool))
    else:
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        print(f'{len(to_drop)} features with SCC >= 0.95 out of {X.shape[1]}.')

    # Get largest correlation between two features.
    max_feature_corr = []
    for num, col in enumerate(upper.columns):
        if not np.isnan(max(upper[col])):
            max_feature_corr.append(max(upper[col]))

    max_feature_corr = np.array(max_feature_corr)

    # SCC of all Z-scored features calculated from the original images according to feature category.
    feature_cats = np.array(to_feature_categories(upper.columns))
    sorted_cats_idx = np.argsort(feature_cats)
    feature_cats = feature_cats[sorted_cats_idx]

    to_drop = np.where(sorted_cats_idx == max(sorted_cats_idx))
    sorted_cats_idx = np.delete(sorted_cats_idx, to_drop)
    feature_cats = np.delete(feature_cats, to_drop)

    colors = []
    for label in feature_cats:
        colors.append(get_palette_colour(label))

    # The number of features with at least 0.95 SCC. Note that each feature
    # corr to another forms a pair meaning that the number of corr features
    #
    # print(sum(max_feature_corr[sorted_cats_idx] >= 0.95))

    plt.figure()
    plt.scatter(
        np.arange(np.size(max_feature_corr)),
        max_feature_corr[sorted_cats_idx],
        color=colors
    )
    handles = [
        mpatches.Patch(color=get_palette_colour(label), label=label)
        for label in np.unique(feature_cats)
    ]
    plt.legend(
        handles=handles, loc='upper center',
        bbox_to_anchor=(0.5, -0.05), ncol=4, fancybox=True, shadow=True
    )
    plt.ylabel("Spearman's Rank Correaltion Coefficient")
    plt.xlabel('Feature ID')
    plt.xticks([])

    y_coords = np.linspace(0.0, 1.0, 6)
    y_ticks = [f'{tick:.02f}' for tick in y_coords]
    plt.yticks(y_coords, y_ticks)

    plt.savefig(path_to_figure, bbox_inches='tight', dpi=CONF.DPI)
    if show:
        plt.show()


def plot_pet_parameter_corr():

    for plot_num in range(3):
        _plot_pet_parameter_corr(plot_num)


def _plot_pet_parameter_corr(plot_num):

    path_to_figure = f'./../../figures/feature_redundancy/pet_param_corr_{plot_num}.pdf'
    path_to_features = '../../data_source/to_analysis/original_images/all_features_original_images.csv'

    X = pd.read_csv(path_to_features, index_col=0)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(X_std, index=X.index, columns=X.columns)
    corr_matrix = X_std.corr(method='spearman').abs()

    y = [
        'PETparam_SUVpeak',
        'PETparam_MTV',
        'PETparam_TLG'
    ]
    x = [
        'PET_original_glszm_HighGrayLevelZoneEmphasis_128bins',
        'original_shape_VoxelVolume',
        'PET_original_firstorder_Energy'
    ]
    feature_cats = to_feature_categories(x[plot_num])
    plt.figure()
    plt.scatter(
        y=X_std.loc[:, y[plot_num]],
        x=X_std.loc[:, x[plot_num]],
        c=get_palette_colour(feature_cats[plot_num])
    )
    y_labels = [
        'SUV Peak',
        'Metabolic Tumor Volume',
        'Total Lesion Glycolysis'
    ]
    x_labels = [
        'PET GLSZM HighGrayLevelZoneEmphasis 128bins',
        'Shape VoxelVolume',
        'PET First Order Energy'
    ]
    score = np.round(corr_matrix.loc[y[plot_num], x[plot_num]], 2)
    plt.annotate(
        f'SCC: {score}',
        xy=(12, 520),
        xycoords='axes points',
        size=20,
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round', fc='w')
    )
    plt.xlabel(x_labels[plot_num])
    plt.ylabel(y_labels[plot_num])
    plt.xticks([], [])
    plt.yticks([], [])

    plt.savefig(path_to_figure, bbox_inches='tight', dpi=CONF.DPI)


def clinical_and_pet_corr():

    path_to_figure = './../../figures/feature_redundancy/pet_param_corr_2.pdf'
    path_to_features = '../../data_source/to_analysis/original_images/all_features_original_images.csv'
    path_to_clinical = './../../data_source/to_analysis/clinical_params.csv'

    clinical = pd.read_csv(path_to_clinical, index_col=0)
    _X = pd.read_csv(path_to_features, index_col=0)

    clinicals = ['Age', 'Years Smoking', 'Naxogin Days']
    to_drop = [col for col in clinical.columns if not col in clinicals]

    _X.drop(to_drop, axis=1, inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(_X)
    X = pd.DataFrame(X, columns=_X.columns, index=_X.index)

    corr_matrix = X.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    pet_params = list(X.filter(regex='PETparam').columns)
    pet_corr_pairs = {}
    for num, col in enumerate(upper.columns):
        if col in pet_params:
            idx = np.nanargmax(upper.loc[:, col])
            pair = upper.iloc[:, idx].name
            pet_corr_pairs[col] = (pair, upper.iloc[idx, num])
    print('PET parameter correlations:')
    print(pet_corr_pairs)

    clinical_corr_pairs = {}
    for num, col in enumerate(upper.columns):
        if col in clinicals:
            idx = np.nanargmax(upper.loc[col, :])
            pair = upper.iloc[:, idx].name
            clinical_corr_pairs[col] = (pair, upper.iloc[num, idx])
    print('Clinical correlations:')
    print(clinical_corr_pairs)
    print()


def plot_texture_volume_corr():
    # Plot
    # * CT fs entropy
    # * CT fs energy
    # * CT GLRLM RLNU
    # * CT NGDTM coarsness
    # as func of log(num voxels) in roi (size tumor).
    pass


if __name__ == '__main__':
    #plot_pet_parameter_corr()
    plot_hassan_mod()
    #clinical_and_pet_corr()
    #plot_texture_volume_corr()
    #transform()
