import re
import ast

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ast import literal_eval
from collections import OrderedDict
from matplotlib.ticker import StrMethodFormatter


import fig_config as CONFIG

CONFIG.plot_setup()


def set_axis_precision(axis='y'):

    if axis == 'y':
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    elif axis == 'x':
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))


def get_palette_colour(label):

    palette = CONFIG.base_palette(n=7)

    mapping = {
        'Shape': palette[0],
        'Clinical': palette[1],
        'CT First Order': palette[2],
        'First Order': palette[2],
        'CT GLCM': palette[4],
        'CT GLDM': palette[4],
        'CT GLRLM': palette[4],
        'CT GLSZM': palette[4],
        'CT NGTDM': palette[4],
        'PET GLCM': palette[5],
        'PET GLDM': palette[5],
        'PET GLRLM': palette[5],
        'PET GLZSM': palette[5],
        'PET NGTDM': palette[5],
        'PET First Order': palette[3],
        'CT Texture': palette[4],
        'Texture': palette[4],
        'PET Texture': palette[5],
        'PET Parameter': palette[6]
    }
    return mapping[label]


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



def format_feature_labels(labels):
    """Process raw feature labels."""
    prep_labels = []
    for label in labels:
        print(label)
        #print(label)
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


def extract_votes(votes, X, feature_labels, clinical=None):
    """Collects selected feature votes from experimental results.

    Args:
        votes (): The votes column from results DataFrame.
        X (): The feature matrix used in the experiment.
        feature_labels():

    """

    votes_bk = np.zeros(X.shape[1], dtype=int)

    # Format string of feature votes into ndarray and accumulate votes.
    for str_array in votes:
        str_array = str_array[1:-1].strip()
        run_votes = str_array.replace('  ', ' ')
        run_votes = run_votes.strip()
        run_votes = run_votes.replace(' ', ',')
        run_votes = run_votes.lstrip(',')
        run_votes = f'[{run_votes}]'
        votes_arr = np.array(ast.literal_eval(run_votes), dtype=np.int32)
        if np.unique(votes_arr).size > 1:
            votes_bk += votes_arr

    idx = np.argsort(votes_bk)[::-1]
    # Sorted accumulated votes for each individual feature.
    sorted_votes = np.trim_zeros(votes_bk[idx], trim='b')

    labels = np.array(feature_labels)[idx]
    sorted_feature_labels = labels[:len(sorted_votes)]

    data = pd.DataFrame(
        {'votes': sorted_votes, 'labels': sorted_feature_labels}
    )
    if clinical is not None:
        data = merge_clinical_votes(clinical, data)
        data.sort_values(by='votes', inplace=True, ascending=False)

    return data


def gen_feature_group_inidicator(X):

    # NOTE: Clinical features: 0
    col_idx = feat_category_mappings()
    feature_idx = np.zeros(X.shape[1], dtype=np.int32)
    for label in col_idx.keys():
        target_cols = list(X.filter(regex=label).columns)
        i = np.squeeze(np.where(np.isin(X.columns, target_cols)))
        feature_idx[i] = np.tile(col_idx[label], np.size(i))

    return feature_idx


def group_votes_by_category(data,
                            X,
                            scale=None,
                            clinical_labels=None):

    group_id = gen_feature_group_inidicator(X)
    all_cats = np.sum(np.unique(group_id))

    cat_to_id = feat_category_mappings()
    id_to_cat = feat_category_mappings(id_to_cat=True)

    votes = data.loc[:, 'votes']
    labels = data.loc[:, 'labels']

    grouped_votes = dict.fromkeys(id_to_cat.values(), 0)
    group_sizes = dict.fromkeys(grouped_votes.keys(), 0)
    # Match feature with category to accumulate votes.
    for num, label in enumerate(labels):
        comps = label.split('_')
        if 'PETparam' in comps:
            grouped_votes['PET Parameter'] = int(votes[num])
            group_sizes['PET Parameter'] += 1
        elif 'PET' in comps[0]:
            if 'firstorder' in comps[2]:
                grouped_votes['PET First Order'] += int(votes[num])
                group_sizes['PET First Order'] += 1
            else:
                grouped_votes['PET Texture'] += int(votes[num])
                group_sizes['PET Texture'] += 1
        elif 'CT' in comps[0]:
            if 'firstorder' in comps[2]:
                grouped_votes['CT First Order'] += int(votes[num])
                group_sizes['CT First Order'] += 1
            else:
                grouped_votes['CT Texture'] += int(votes[num])
                group_sizes['CT Texture'] += 1
        elif 'shape' in comps:
            grouped_votes['Shape'] += int(votes[num])
            group_sizes['Shape'] += 1
        else:
            grouped_votes['Clinical'] += int(votes[num])
            group_sizes['Clinical'] += 1

    if scale:
        grouped_votes = {
            key: value / group_sizes[key] / scale
            for key, value in grouped_votes.items()
        }
    grouped_votes = pd.DataFrame([grouped_votes.values(), grouped_votes.keys()]).T
    grouped_votes.columns = ['votes', 'labels']
    grouped_votes.sort_values(by='votes', inplace=True)

    return grouped_votes


def merge_clinical_votes(clinical, data):

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
    num_encodings_tracker = dict.fromkeys(votes_noncoded_clinical.keys(), 0)
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
    tmp_df.columns = ['votes', 'labels']
    tmp_df.index = np.arange(data.index[-1] + 1, data.index[-1] + 1 + tmp_df.shape[0])

    data = data.append(tmp_df)

    return data


def plot_feature_votes_by_category(data,
                                   path_to_figure=None,
                                   show=False):
    """Plots the average selected rate from a category.

    """
    keys = data.loc[:, 'labels']
    votes = data.loc[:, 'votes']

    colors = []
    for key in keys:
        colors.append(get_palette_colour(key))

    y_coords = np.arange(len(keys))
    x_coords = np.linspace(0, max(votes), 6)
    plt.barh(y_coords, votes, color=colors, align='center') #alpha=CONFIG.ALPHA)
    x_ticks = [f'{tick:.02f}' for tick in x_coords]
    #plt.xlim([0.00, 0.50])
    plt.yticks(y_coords, keys, ha='right', va='center')
    plt.xticks(x_coords, x_ticks, ha='center', va='top')
    plt.xlabel('Relative Frequency of Feature Selection from Category')

    if path_to_figure is not None:
        plt.savefig(path_to_figure, bbox_inches='tight', dpi=CONFIG.DPI)
    if show:
        plt.show()


def plot_feature_votes(
    data,
    path_to_figure=None,
    n=10,
    show=False,
    legend=True,
    format_labels=True
):
    # NB:
    # * Make sure ecah feature gets assigned the correspnoding feature category
    #   color.
    # * Map feature colors according to category, and not unique features.
    labels, votes = data.iloc[:n, 1], data.iloc[:n, 0]
    labels = list(labels)
    votes = np.squeeze(votes.values)

    keys = CONFIG.feature_categories_from_labels(labels)
    colors = []
    for label in keys:
        colors.append(get_palette_colour(label))

    if format_labels:
        fnames = format_feature_labels(labels)
    else:
        fnames = list(labels)
    y_coords = np.arange(len(fnames))
    x_coords = np.round(np.linspace(0, max(votes), 6), 2)
    x_ticks = [f'{tick:.02f}' for tick in x_coords]
    plt.barh(y_coords, votes, color=colors, align='center') #alpha=CONFIG.ALPHA)
    plt.yticks(y_coords, fnames, ha='right', va='baseline')
    plt.xticks(x_coords, x_ticks, ha='center', va='top')
    plt.xlabel('Relative Frequency of Feature Being Selected')

    handles = [
        mpatches.Patch(color=get_palette_colour(label), label=label)
        for label in np.unique(keys)
    ]
    if legend:
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


def category_importance(data,
                        X,
                        scale=None,
                        clinical_labels=None):

    group_id = gen_feature_group_inidicator(X)
    all_cats = np.sum(np.unique(group_id))

    cat_to_id = feat_category_mappings()
    id_to_cat = feat_category_mappings(id_to_cat=True)

    votes = data.loc[:, 'votes']
    labels = data.loc[:, 'labels']

    max_votes = OrderedDict({key: 0 for key in id_to_cat.values()})
    feature_labels = OrderedDict({key: 0 for key in id_to_cat.values()})

    for num, label in enumerate(labels):
        comps = label.split('_')
        if 'PETparam' in comps:
            if max_votes['PET Parameter'] < int(votes[num]):
                max_votes['PET Parameter'] = int(votes[num])
                feature_labels['PET Parameter'] = label
        elif 'PET' in comps[0]:
            if 'firstorder' in comps[2]:
                if max_votes['PET First Order'] < int(votes[num]):
                    max_votes['PET First Order'] = int(votes[num])
                    feature_labels['PET First Order'] = label
            else:
                if max_votes['PET Texture'] < int(votes[num]):
                    max_votes['PET Texture'] = int(votes[num])
                    feature_labels['PET Texture'] = label
        elif 'CT' in comps[0]:
            if 'firstorder' in comps[2]:
                if max_votes['CT First Order'] < int(votes[num]):
                    max_votes['CT First Order'] += int(votes[num])
                    feature_labels['CT First Order'] = label
            else:
                if max_votes['CT Texture'] < int(votes[num]):
                    max_votes['CT Texture'] = int(votes[num])
                    feature_labels['CT Texture'] = label
        elif 'shape' in comps:
            if max_votes['Shape'] < int(votes[num]):
                max_votes['Shape'] = int(votes[num])
                feature_labels['Shape'] = label
        else:
            if max_votes['Clinical'] < int(votes[num]):
                max_votes['Clinical'] = int(votes[num])
                feature_labels['Clinical'] = label

    if scale is not None:
        max_votes = {
            key: value / scale for key, value in max_votes.items()
        }

    feature_labels = format_feature_labels(feature_labels.values())

    max_votes = pd.DataFrame(
        [max_votes.values(), max_votes.keys(), feature_labels]
    ).T
    max_votes.columns = ['votes', 'category', 'labels']
    max_votes.sort_values(by='votes', inplace=True)

    return max_votes


def plot_category_importance(
    data,
    path_to_figure=None,
    show=False,
    format_labels=True
):

    labels, keys, votes = data.iloc[:, 2], data.iloc[:, 1], data.iloc[:, 0]
    votes = np.squeeze(votes.values)

    fnames = []
    for label in labels:
        if label == 'PETparam: SUVpeak':
            fnames.append('SUVpeak')
        elif label == 'CT GrayLevelNonUniformity':
            fnames.append('CT GLNU 64 bins')
        elif label == 'PETparam: TLG':
            fnames.append('TLG')
        else:
            fnames.append(label)

    colors = []
    for label in keys:
        colors.append(get_palette_colour(label))


    y_coords = np.arange(len(fnames))
    x_coords = np.round(np.linspace(0, max(votes), 6), 2)
    x_ticks = [f'{tick:.02f}' for tick in x_coords]
    plt.barh(y_coords, votes, color=colors, align='center') #alpha=CONFIG.ALPHA)
    plt.yticks(y_coords, fnames, ha='right', va='baseline')
    plt.xticks(x_coords, x_ticks, ha='center', va='top', )
    plt.xlabel('Relative Frequency of Feature Being Selected')

    handles = [
        mpatches.Patch(color=get_palette_colour(label), label=label)
        for label in np.unique(keys)
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


def baseline():

    NUM_RUNS = 7 * 10 * 40
    NUM_RUNS_CV = 5 * NUM_RUNS

    path_to_figure = './../../figures/feature_votes/votes_baseline_radiomics.pdf'
    path_to_max_vote_figure = './../../figures/feature_votes/max_category_votes_baseline_radiomics.pdf'
    path_to_grouped_figure = './../../figures/feature_votes/category_votes_baseline_radiomics.pdf'
    path_to_clinical = './../../data_source/to_analysis/clinical_params.csv'
    path_to_featureset = '../../data_source/to_analysis/original_images/all_features_original_images.csv'
    path_to_mod_comp_results = '../../data_source/results/original_images/results_all_features_original_images.csv'

    X = pd.read_csv(path_to_featureset, index_col=0)
    results = pd.read_csv(path_to_mod_comp_results, index_col=0)
    clinical = pd.read_csv(path_to_clinical, index_col=0)

    votes = extract_votes(
        results['feature_votes'], X, list(X.columns), clinical=clinical
    )
    grouped_votes = group_votes_by_category(
        votes, X, scale=NUM_RUNS_CV, clinical_labels=list(clinical.columns)
    )
    most_selected = category_importance(
        votes, X, clinical_labels=list(clinical.columns), scale=NUM_RUNS_CV
    )
    fig = plt.figure(figsize=(15, 5))
    plot_category_importance(
        most_selected, path_to_figure=path_to_max_vote_figure
    )
    fig = plt.figure(figsize=(15, 5))
    plot_feature_votes_by_category(
        grouped_votes, path_to_figure=path_to_grouped_figure
    )
    votes.loc[:, 'votes'] = votes.loc[:, 'votes'] / NUM_RUNS_CV
    fig = plt.figure(figsize=(15, 7))
    plot_feature_votes(votes, path_to_figure=path_to_figure, format_labels=True)


# NOTE: Seems to be overlap/doubled up on some features.
def removed_damaged_slices():

    NUM_RUNS = 7 * 10 * 40
    NUM_RUNS_CV = 5 * NUM_RUNS

    path_to_figure = './../../figures/feature_votes/votes_damaged_radiomics.pdf'
    path_to_grouped_figure = './../../figures/feature_votes/category_votes_damaged_radiomics.pdf'
    path_to_max_vote_figure = './../../figures/feature_votes/max_category_votes_damaged_radiomics.pdf'
    path_to_clinical = './../../data_source/to_analysis/clinical_params.csv'
    path_to_featureset = '../../data_source/to_analysis/removed_broken_slices/all_features_removed_broken_slices.csv'
    path_to_mod_comp_results = './../../data_source/results/removed_broken_slices/results_all_features_removed_broken_slices.csv'

    X = pd.read_csv(path_to_featureset, index_col=0)
    results = pd.read_csv(path_to_mod_comp_results, index_col=0)
    clinical = pd.read_csv(path_to_clinical, index_col=0)

    votes = extract_votes(
        results['feature_votes'], X, list(X.columns), clinical=clinical
    )
    grouped_votes = group_votes_by_category(
        votes, X, scale=NUM_RUNS_CV, clinical_labels=list(clinical.columns)
    )
    most_selected = category_importance(
        votes, X, clinical_labels=list(clinical.columns), scale=NUM_RUNS_CV
    )
    fig = plt.figure(figsize=(15, 5))
    plot_category_importance(
        most_selected, path_to_figure=path_to_max_vote_figure
    )
    fig = plt.figure(figsize=(15, 5))
    plot_feature_votes_by_category(
        grouped_votes, path_to_figure=path_to_grouped_figure
    )
    votes.loc[:, 'votes'] = votes.loc[:, 'votes'] / NUM_RUNS_CV
    fig = plt.figure(figsize=(15, 7))
    plot_feature_votes(votes, path_to_figure=path_to_figure, format_labels=True)


def hpb_unrelated_clincal():

    NUM_RUNS_CV = 10 * 10 * 7 * 40

    path_to_figure = './../../figures/feature_votes/votes_hpv_unrelated_clinical.pdf'
    path_to_clinical = './../../data_source/to_analysis/clinical_params.csv'
    path_to_mod_comp_results = './../../data_source/results/hpv_splitting/results_hpv_b_clinical_only.csv'

    results = pd.read_csv(path_to_mod_comp_results, index_col=0)
    clinical = pd.read_csv(path_to_clinical, index_col=0)

    votes = extract_votes(
        results['feature_votes'], clinical, list(clinical.columns),
        clinical=clinical
    )
    votes.loc[:, 'votes'] = votes.loc[:, 'votes'] / NUM_RUNS_CV
    votes.loc[0, 'labels'] = 'Male'

    fig = plt.figure(figsize=(15, 7))
    plot_feature_votes(
        votes, path_to_figure=path_to_figure, format_labels=False,
        legend=False, n=13
    )


def clinical_baseline():

    NUM_RUNS_CV = 5 * 10 * 7 * 40

    path_to_figure = './../../figures/feature_votes/baseline_clinical.pdf'
    path_to_clinical = './../../data_source/to_analysis/clinical_params.csv'
    path_to_mod_comp_results = './../../data_source/results/original_images/results_full_clinical.csv'

    results = pd.read_csv(path_to_mod_comp_results, index_col=0)
    clinical = pd.read_csv(path_to_clinical, index_col=0)

    votes = extract_votes(
        results['feature_votes'], clinical, list(clinical.columns),
        clinical=clinical
    )
    votes.loc[:, 'votes'] = votes.loc[:, 'votes'] / NUM_RUNS_CV
    votes.loc[3, 'labels'] = 'Male'

    fig = plt.figure(figsize=(15, 7))
    plot_feature_votes(
        votes, path_to_figure=path_to_figure, format_labels=False,
        legend=False, n=13
    )


if __name__ == '__main__':
    baseline()
    removed_damaged_slices()
    hpb_unrelated_clincal()
    clinical_baseline()
