import shap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib import ticker
from sklearn.preprocessing import StandardScaler
from skfeature.function.similarity_based.fisher_score import fisher_score

import pandas as pd
import numpy as np

import fig_config as CONFIG

import pickle


CONFIG.plot_setup()


def feat_category_mappings(id_to_cat=False):

    if id_to_cat:
        return {
            1: 'Shape',
            2: 'CT First Order',
            3: 'CT Texture',#'CT GLCM',
            4: 'CT Texture',#'CT GLDM',
            5: 'CT Texture',#'CT GLRLM',
            6: 'CT Texture',#'CT GLSZM',
            7: 'CT Texture',#'CT NGTDM',
            8: 'PET First Order',
            9: 'PET Texture',#'PET GLCM', 'PET Texture',
            10: 'PET Texture',#'PET GLDM',
            11: 'PET Texture',#'PET GLRLM',
            12: 'PET Texture',#'PET GLZSM',
            13: 'PET Texture',#'PET NGTDM',
            14: 'PET Parameter',
            0: 'Clinical'
        }
    return {
        'clinical': 0,
        'shape': 1,
        'CT_original_firstorder': 2,
        'CT_original_glcm': 3,
        'CT_original_gldm': 4,
        'CT_original_glrlm': 5,
        'CT_original_glszm': 6,
        'CT_original_ngtdm': 7,
        'PET_original_firstorder': 8,
        'PET_original_glcm': 9,
        'PET_original_gldm': 10,
        'PET_original_glrlm': 11,
        'PET_original_glszm': 12,
        'PET_original_ngtdm': 13,
        'PETparam': 14
    }


def get_palette_colour(label):

    palette = CONFIG.base_palette(n=9)
    mapping = {
        'Shape': palette[0],
        'Clinical': palette[1],
        'First Order': palette[2],
        'GLCM': palette[3],
        'GLRLM': palette[4],
        'GLSZM': palette[5],
        'GLDM': palette[6],
        'NGTDM': palette[7],
        'PET parameter': palette[8]
    }
    return mapping[label]


def feature_categories_from_labels(labels):
    """Translates feature labels into feature categories."""
    keys = []
    for label in labels:
        if 'shape' in label:
            keys.append('Shape')
        elif 'firstorder' in label:
            keys.append('First Order')
        elif 'glcm' in label:
            keys.append('GLCM')
        elif 'glrlm' in label:
            keys.append('GLRLM')
        elif 'glszm' in label:
            keys.append('GLSZM')
        elif 'gldm' in label:
            keys.append('GLDM')
        elif 'ngtdm' in label:
            keys.append('NGTDM')
        elif 'PETparam' in label:
            keys.append('PET parameter')
        else:
            keys.append('Clinical')

    return keys


def format_feature_labels(labels):
    """Process raw feature labels."""
    prep_labels = []
    for label in labels:
        if label == 'CT_original_gldm_DependenceNonUniformityNormalized_32bins':
            prep_labels.append('CT DNUN')
        elif label == 'ECOG':
            prep_labels.append('ECOG')
        elif label == 'CT_original_firstorder_MeanAbsoluteDeviation':
            prep_labels.append('CT MAD')
        elif label == 'CT_original_gldm_SmallDependenceEmphasis':
            prep_labels.append('CT SDE')
        elif label == 'CT_original_gldm_LargeDependenceHighGrayLevelEmphasis_64bins':
            prep_labels.append('CT LDHGLE 64bins')
        elif label == 'PET_original_gldm_SmallDependenceLowGrayLevelEmphasis':
            prep_labels.append('PET SDHGLE')
        elif label == 'PET_original_glszm_LargeAreaHighGrayLevelEmphasis_64bins':
            prep_labels.append('PET LAHGL 64 bins')
        elif label == 'CT_original_gldm_DependenceVariance_32bins':
            prep_labels.append('CT DependenceVariance')
        elif label == 'CT_original_glrlm_RunLengthNonUniformityNormalized':
            prep_labels.append('CT RLNUM')
        elif label == 'CT_original_glcm_ClusterShade_32bins':
            prep_labels.append('CT ClusterShade')
        else:
            comps = label.split('_')
            if len(comps) == 1:
                prep_labels.append(label.title())
            elif len(comps) == 2:
                new_label = f'{comps[0]}: {comps[1]}'
                prep_labels.append(new_label)
            elif len(comps) == 3:
                filter_type, feature_type, name = comps
                if len(name) > 15:
                    new_label = '{}'.format(name)
                else:
                    new_label = '{}'.format(name)
                prep_labels.append(new_label)
            elif len(comps) == 4:
                image_type, filter_type, feature_type, name = comps
                if len(name) > 15:
                    new_label = f'{image_type} {name}'
                else:
                    new_label = f'{image_type} {name}'
                prep_labels.append(new_label)
            elif len(comps) == 5:
                image_type, _, _, name, _ = comps
                if len(name) > 15:
                    new_label = f'{image_type} {name}'
                else:
                    new_label = f'{image_type} {name}'
                prep_labels.append(new_label)
            else:
                raise ValueError('Label more than 5 comps!')

    return prep_labels


def extract_ranks(ranks, feature_labels, clinical):
    """Collects selected feature votes from experimental results.

    Args:
        votes (): The votes column from results DataFrame.
        X (): The feature matrix used in the experiment.
        feature_labels():

    """

    data = pd.DataFrame({'ranks': ranks, 'labels': feature_labels})

    # Group the encoded clinical variables and scale by number of encoded members:
    # 1. Extract and format the clinical feature votes.
    # 2. Remove the encoded clinical feature votes from the vote bookeeper.
    # 3. Insert the formatted clinical feature votes.

    # To accumulate votes.
    votes_noncoded_clinical = {
        'ICD-10': 0,
        'T Stage': 0,
        'N Stage': 0,
        'Stage': 0,
        'Histology': 0,
        'HPV': 0,
        'ECOG': 0,
        'Charlson': 0,
        'Cisplatin': 0,
        'Stage': 0
    }
    # Delete row from data.
    num_encodings_tracker = {
        'ICD-10': 0,
        'T Stage': 0,
        'N Stage': 0,
        'Stage': 0,
        'Histology': 0,
        'HPV': 0,
        'ECOG': 0,
        'Charlson': 0,
        'Cisplatin': 0,
    }
    to_drop = []
    for col in clinical.columns:
        match = data[data['labels'] == col]
        if not match.empty:
            vote, label = np.squeeze(match.values)
            if 'ICD-10' in label:
                votes_noncoded_clinical['ICD-10'] += float(vote)
                num_encodings_tracker['ICD-10'] += 1
                to_drop.append(int(match.index[0]))
            if 'Histology' in label:
                votes_noncoded_clinical['Histology'] += float(vote)
                num_encodings_tracker['Histology'] += 1
                to_drop.append(int(match.index[0]))
            if 'HPV' in label:
                votes_noncoded_clinical['HPV'] += float(vote)
                num_encodings_tracker['HPV'] += 1
                to_drop.append(int(match.index[0]))
            if 'ECOG' in label:
                votes_noncoded_clinical['ECOG'] += float(vote)
                num_encodings_tracker['ECOG'] += 1
                to_drop.append(int(match.index[0]))
            if 'Charlson' in label:
                votes_noncoded_clinical['Charlson'] += float(vote)
                num_encodings_tracker['Charlson'] += 1
                to_drop.append(int(match.index[0]))
            if 'Cisplatin' in label:
                votes_noncoded_clinical['Cisplatin'] += float(vote)
                num_encodings_tracker['Cisplatin'] += 1
                to_drop.append(int(match.index[0]))
            if 'T Stage' in label:
                votes_noncoded_clinical['T Stage'] += float(vote)
                num_encodings_tracker['T Stage'] += 1
                to_drop.append(int(match.index[0]))
            if 'N Stage' in label:
                votes_noncoded_clinical['N Stage'] += float(vote)
                num_encodings_tracker['N Stage'] += 1
                to_drop.append(int(match.index[0]))
            if 'Stage' in label:
                if 'T Stage' in label:
                    pass
                elif 'T Stage' in label:
                    pass
                else:
                    votes_noncoded_clinical['Stage'] += float(vote)
                    num_encodings_tracker['Stage'] += 1
                    to_drop.append(int(match.index[0]))

    # Scale votes by the number of encodings.
    row_idx = data.index[-1] + 1
    for num, (key, value) in enumerate(votes_noncoded_clinical.items()):
        votes_noncoded_clinical[key] /= (num_encodings_tracker[key] + 1e-20)

    # Drop votes on encoded clinical variable.
    data.drop(to_drop, inplace=True, axis=0)
    data.reset_index(drop=True, inplace=True)

    tmp_df = pd.DataFrame([votes_noncoded_clinical.values(), votes_noncoded_clinical.keys()]).T
    tmp_df.columns = ['ranks', 'labels']
    tmp_df.index = np.arange(data.index[-1] + 1, data.index[-1] + 1 + tmp_df.shape[0])

    data = data.append(tmp_df)

    return data


def hpv_unrel_X_y(return_clinical=False):

    X = pd.read_csv(
        '../../data_source/to_analysis/hpv_splitting/all_features_orig_images_icc_dropped_hpv_group_b.csv',
        index_col=0
    )
    y = pd.read_csv('../../data_source/to_analysis/hpv_splitting/dfs_orig_images_hpv_group_b.csv', index_col=0)
    y = np.squeeze(y.values)

    if return_clinical:
        clinical = pd.read_csv(
            '../../data_source/to_analysis/clinical_params.csv', index_col=0
        )
        clinical = clinical.loc[X.index, :]
        return X, y, clinical

    return X, y


def pairplot_biomarkers():
    # Set outcome hue. Include outcome.

    path_to_features = '../../data_source/to_analysis/hpv_splitting'
    path_to_target = '../../data_source/'

    biomarkers = []

    X = pd.read_csv(path_to_features, index_col=0)
    X = X.loc[:, biomarkers]


def plot_feature_ranking(ranks, n=5, show=False, path_to_figure=None):

    imp = ranks.iloc[:n, 0]
    labels = ranks.iloc[:n, 1]

    # Format feature labels and define ticks coord grid.
    fnames = format_feature_labels(labels)
    y_coords = np.arange(len(fnames))
    # Assign a color to each label by associated feature category.
    colors = []
    keys = feature_categories_from_labels(labels)
    for key in keys:
        colors.append(get_palette_colour(key))

    plt.barh(
        y_coords, imp.values, color=colors, align='center', #alpha=CONFIG.ALPHA
    )
    plt.yticks(y_coords, fnames, ha='right', va='center')
    x_coords = np.linspace(0, np.max(imp.values), 6)
    plt.xticks(x_coords, [f'{num:.02f}' for num in x_coords])
    handles = [
        mpatches.Patch(color=get_palette_colour(key), label=key)
        for key in np.unique(keys)
    ]
    plt.legend(
        handles=handles,
        title='Feature Categories:',
        title_fontsize=18,
        fancybox=True,
        shadow=True,
        ncol=2,
        labelspacing=0.25,
    )
    if path_to_figure is not None:
        plt.savefig(path_to_figure, dpi=CONFIG.DPI, bbox_inches='tight')
    if show:
        plt.show()


def multisurf_ranking():

    n = 10
    # Five vars: (15, 3.5)
    figsize = (15, 7)
    show = False

    path_to_figure = '../../figures/feature_importances/multisurf_ranking.pdf'
    path_to_multisurf_ranks = './../../data_source/results/feature_importance/multisurf_feat_ranks.npy'

    X, y, clinical = hpv_unrel_X_y(return_clinical=True)

    _ranks = np.load(path_to_multisurf_ranks)
    ranks = extract_ranks(_ranks, X.columns, clinical)
    ranks.sort_values('ranks', ascending=False, inplace=True)

    plt.figure(figsize=figsize)
    plt.xlabel('MultiSURF Feature Weight')
    plot_feature_ranking(ranks, n=n, show=show, path_to_figure=path_to_figure)


def get_biomarkers(labels_only=True):

    path_to_shap_values = '../../data_source/results/feature_importance/ms_xgb_shap_values.npy'
    path_to_multisurf_ranks = './../../data_source/results/feature_importance/multisurf_feat_ranks.npy'

    shap_values = np.load(path_to_shap_values)
    X, _ = hpv_unrel_X_y(return_clinical=False)
    ms_ranks = np.load(path_to_multisurf_ranks)

    # Extract the 26 features originally selected by multisurf.
    ms_idx = np.argsort(ms_ranks)[::-1][:26]
    X = X.iloc[:, ms_idx]
    # Extract the four features selected by SHAP values.
    shap_idx = np.where(np.sum(shap_values, axis=0) != 0)
    X = X.iloc[:, np.squeeze(shap_idx)]

    if labels_only:
        return list(X.columns)

    return X


# NOTE: Produces ABSOLUTE VALUE of mean.
# Fig caption: The four features assigned an average SHAP value dfferent from
def shap_values_bar():
    # Horisontal bar plot of sap values != 0.

    show = False
    #figsize = (15, 2.7)
    #path_to_figure = '../../figures/feature_importances/lgbm_shap_bars.pdf'
    #path_to_shap_values = '../../data_source/results/feature_importance/fisher_lgbm_shap_values.npy'
    #path_to_ranks = '../../data_source/results/feature_importance/fisher_feat_ranks.npy'
    #path_to_features = '../../data_source/to_analysis/compressed_features/all_features_orig_images_icc_dropped.csv'
    #X = pd.read_csv(path_to_features, index_col=0)

    figsize = (15, 3.3)
    path_to_figure = '../../figures/feature_importances/xgb_shap_bars.pdf'
    path_to_shap_values = '../../data_source/results/feature_importance/ms_xgb_shap_values.npy'
    path_to_ranks = '../../data_source/results/feature_importance/multisurf_feat_ranks.npy'
    X, _ = hpv_unrel_X_y(return_clinical=False)
    ranks = np.load(path_to_ranks)
    idx = np.argsort(ranks)[::-1]
    X = X.iloc[:, idx]

    shap_values = np.load(path_to_shap_values)
    ranks = np.load(path_to_ranks)

    # Extract the four features selected by SHAP values.
    shap_idx = np.where(np.sum(shap_values, axis=0) != 0)
    avg_shap = np.squeeze(np.mean(np.abs(shap_values[:, shap_idx]), axis=0))

    X = X.iloc[:, np.squeeze(shap_idx)]

    fnames = format_feature_labels(X.columns)
    coords = np.arange(len(fnames))

    # Sort scores.
    idx = np.argsort(avg_shap)[::-1]
    fnames = np.array(fnames)[idx]
    avg_shap = avg_shap[idx]

    colors = []
    keys = feature_categories_from_labels(X.columns)
    for key in np.array(keys)[idx]:
        colors.append(get_palette_colour(key))

    fig = plt.figure(figsize=figsize)
    plt.barh(
        coords, avg_shap,
        color=colors, align='center', #alpha=CONFIG.ALPHA
    )
    plt.xlabel(r'The mean absolute of SHAP values ($mean \left | SHAP \right |$).')
    x_coords = np.linspace(0.0, np.max(avg_shap), 6)
    plt.xticks(x_coords, [f'{tick:.02f}' for tick in x_coords])
    plt.yticks(coords, fnames, ha='right', va='center')
    handles = [
        mpatches.Patch(color=get_palette_colour(key), label=key)
        for key in np.unique(keys)
    ]
    plt.legend(
        handles=handles,
        title='Feature Categories:',
        title_fontsize=18,
        fancybox=True,
        shadow=True,
        ncol=3,
        labelspacing=0.25,
    )
    plt.savefig(path_to_figure, dpi=CONFIG.DPI, bbox_inches='tight')
    if show:
        plt.show()


def pairplot_biomarkers():
    """

    NB: Everything is Z-scored since the same transformation is done prior to
        classification!

    """

    show = False
    path_to_figure = '../../figures/feature_importances/pairplots/'

    _X, y = hpv_unrel_X_y(return_clinical=False)

    biomarkers = get_biomarkers(labels_only=True)
    labels = format_feature_labels(biomarkers)

    X = _X.loc[:, biomarkers]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, index=list(_X.index), columns=labels)

    hue1 = np.squeeze(np.where(y == 0))
    hue2 = np.squeeze(np.where(y == 1))

    palette = CONFIG.base_palette(n=6)

    bins = np.linspace(0, 10, 100)

    nrows, ncols = 4, 4
    for col_num in range(ncols):

        x_label = labels[col_num]
        for row_num in range(nrows):

            y_label = labels[row_num]

            plt.figure()

            X_col_hue1 = X.iloc[hue1, col_num]
            X_col_hue2 = X.iloc[hue2, col_num]

            X_row_hue1 = X.iloc[hue1, row_num]
            X_row_hue2 = X.iloc[hue2, row_num]

            # Plot distribution.
            if x_label == y_label:
                ext = 'dist'
                plt.hist(X_col_hue1, color=palette[1])
                plt.hist(X_col_hue2, color=palette[2])
                #plt.xlabel(f'{y_label} Bins')
                #plt.ylabel('Count')
                print(y_label)

                min_x = min(np.min(X_col_hue1), np.min(X_col_hue2))
                max_x = max(np.max(X_col_hue1), np.max(X_col_hue2))

                #x_coords = np.linspace(min_x, max_x, 6, dtype=float)
                #x_ticks = [f'{tick:.02f}' for tick in x_coords]
                #plt.xticks(x_coords, x_ticks)

                plt.xticks([], []); plt.yticks([], [])

            # Scatter plot.
            else:
                ext = 'scatter'
                plt.scatter(X_col_hue1, X_row_hue1, color=palette[1])
                plt.scatter(X_col_hue2, X_row_hue2, color=palette[2])
                plt.ylabel(y_label)
                plt.xlabel(x_label)

                min_x = min(np.min(X_col_hue1), np.min(X_col_hue2))
                max_x = max(np.max(X_col_hue1), np.max(X_col_hue2))

                x_coords = np.linspace(min_x, max_x, 6, dtype=float)
                x_ticks = [f'{tick:.02f}' for tick in x_coords]
                plt.xticks(x_coords, x_ticks)

                min_y = min(np.min(X_row_hue1), np.min(X_row_hue2))
                max_y = max(np.max(X_row_hue1), np.max(X_row_hue2))

                y_coords = np.linspace(min_y, max_y, 6, dtype=float)
                y_ticks = [f'{tick:.02f}' for tick in y_coords]
                plt.yticks(y_coords, y_ticks)

            _path_to_figure = f'{path_to_figure}{ext}_{col_num}_{row_num}.pdf'
            plt.savefig(_path_to_figure, dpi=CONFIG.DPI, bbox_inches='tight')

            if show:
                plt.show()


def fisher_score_ranking(num_features=None):
    # NOTE: For BigArt. Fisher Score ranking of all features. These features
    # are passed on to LGBM in best model.

    n = 10
    # Five vars: (15, 3.5)
    figsize = (15, 7)
    show = False

    path_to_figure = '../feature_importances/fisher_score_ranking.pdf'
    path_to_scores = './../../data_source/results/feature_importance/fisher_feat_ranks.npy'
    path_to_features = '../../data_source/to_analysis/compressed_features/all_features_orig_images_icc_dropped.csv'
    path_to_target = '../../data_source/to_analysis/target_dfs.csv'
    path_to_clinical = '../../data_source/to_analysis/clinical_params.csv'

    X = pd.read_csv(path_to_features,index_col=0)
    y = pd.read_csv(path_to_target, index_col=0)
    y = np.squeeze(y.values)

    clinical = pd.read_csv(path_to_clinical, index_col=0)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    scores = fisher_score(X_std, y)
    np.save(path_to_scores, scores)

    ranks = extract_ranks(scores, X.columns, clinical)
    ranks.sort_values('ranks', ascending=False, inplace=True)

    plt.figure(figsize=figsize)
    plt.xlabel('Fisher Score')
    plot_feature_ranking(ranks, n=n, show=show, path_to_figure=path_to_figure)



def categorical_feature_score_ranking(multisurf=True, relative=True):

    if multisurf:
        path_to_scores = './../../data_source/results/feature_importance/multisurf_feat_ranks.npy'
        n = 26
        X, _, clinical = hpv_unrel_X_y(return_clinical=True)
    else:
        path_to_scores = './../../data_source/results/feature_importance/fisher_feat_ranks.npy'
        n = 19
        path_to_clinical = '../../data_source/to_analysis/clinical_params.csv'
        path_to_features = '../../data_source/to_analysis/compressed_features/all_features_orig_images_icc_dropped.csv'
        X = pd.read_csv(path_to_features,index_col=0)
        clinical = pd.read_csv(path_to_clinical, index_col=0)

    print('Num orig features', X.shape[1])

    pet = X.filter(regex='CT')
    pet_fs = pet.filter(regex='CT_original_firstorder')
    pet_text = pet.drop(pet_fs.columns, axis=1)
    #print(pet_fs.columns.size)
    #print(pet_text.columns.size)

    # PET FS = 7
    # PET TEXT = 27
    # CT FS = 12
    # CT TEXT = 56
    # Shape = 8
    # Clinical = 42

    #print(X.filter(regex='shape').)


    """

    scores = np.load(path_to_scores)
    ranks = extract_ranks(scores, X.columns, clinical)
    ranks.sort_values('ranks', ascending=False, inplace=True)
    feats = ranks.loc[:, 'labels'].values[:n]

    id_to_cat = feat_category_mappings(id_to_cat=True)
    group_votes = dict.fromkeys(id_to_cat.values(), 0)
    for num, label in enumerate(feats):

        comps = label.split('_')
        if 'PETparam' in comps:
            group_votes['PET Parameter'] += 1
        elif 'PET' in comps[0]:
            if 'firstorder' in comps[2]:
                group_votes['PET First Order'] += 1
            else:
                group_votes['PET Texture'] += 1
        elif 'CT' in comps[0]:
            if 'firstorder' in comps[2]:
                group_votes['CT First Order'] += 1
            else:
                group_votes['CT Texture'] += 1
        elif 'shape' in comps:
            group_votes['Shape'] += 1
        else:
            group_votes['Clinical'] += 1

    print(group_votes)
    """


def shap_feature_values_plot():

    n = 26
    show = False

    path_to_figure = '.pdf'
    path_to_scores = './../../data_source/results/feature_importance/multisurf_feat_ranks.npy'
    path_to_shap_values = '../../data_source/results/feature_importance/ms_xgb_shap_values.npy'

    shap_values = np.load(path_to_shap_values)
    X_marker = get_biomarkers(labels_only=False)

    shap_values = np.load(path_to_shap_values)
    X, _ = hpv_unrel_X_y(return_clinical=False)
    ms_ranks = np.load(path_to_scores)

    # Extract the 26 features originally selected by multisurf.
    ms_idx = np.argsort(ms_ranks)[::-1][:26]
    X = X.iloc[:, ms_idx]
    # Extract the four features selected by SHAP values.
    shap_idx = np.squeeze(np.where(np.sum(shap_values, axis=0) != 0))
    #X = X.iloc[:, shap_idx]

    #shap_values = shap_values[:, shap_idx].ravel()

    shap.summary_plot(shap_values, X,color=plt.cm.viridis)
    plt.show()
    """
    arr_labels = np.repeat(X.columns.values, 67)

    df = pd.DataFrame([shap_values, arr_labels]).T
    df.columns = ['shap_values', 'arr_labels']

    arr_hue = X.unstack().values
    sns.catplot(
        x='shap_values', y='arr_labels', data=df,

    )
    plt.show()



    for num, col in enumerate(X):
        shap_value = shap_values[:, shap_idx[num]]
        s_min, s_max = np.min(shap_value), np.max(shap_value)
        scatt = plt.scatter(
            shap_value, X.loc[:, col],
            cmap='viridis', c=np.arange(67), vmin=s_min, vmax=s_max
        )
        plt.colorbar(scatt)
        x_coords = np.linspace(s_min, s_max, 6)
        x_ticks = [f'{tick:.02f}' for tick in shap_value]
        plt.xticks(x_coords, x_ticks)
    """

    #plt.axvline(x=0, c='cyan')

    #plt.savefig(path_to_figure, dpi=CONFIG.DPI, bbox_inches='tight')
    if show:
        plt.show()



if __name__ == '__main__':
    #categorical_feature_score_ranking()
    pairplot_biomarkers()
    #shap_feature_values_plot()
    #fisher_score_ranking()
    #shap_values_bar()
