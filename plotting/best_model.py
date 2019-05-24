import numpy as np
import pandas as pd
import statistics as stat

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from operator import itemgetter

from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
import shap

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


def _plot_train_valid(results, path_to_figure=None, show=False):

    test = results['test_score']
    train = results['train_score']
    test_std = np.sqrt(results['test_score_variance'])
    train_std = np.sqrt(results['train_score_variance'])

    palette = CONFIG.base_palette(n=4)

    # Repeated experiments.
    x_coords = np.arange(np.size(test)) + 1

    plt.plot(
        x_coords,
        train,
        color=palette[1],
        marker='o',
        markersize=5,
        label=f"Training score"
    )
    plt.plot(
        x_coords,
        test,
        color=palette[2],
        linestyle='--',
        marker='s',
        label='Validation score'
    )
    plt.fill_between(
        x_coords,
        train - train_std,
        train + train_std,
        alpha=0.25,
        color=palette[1]
    )
    plt.fill_between(
        x_coords,
        test - test_std,
        test + test_std,
        alpha=0.25,
        color=palette[2]
    )
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fancybox=True,
        shadow=True
    )
    y_coords = np.linspace(0.0, 1.0, 6)
    y_ticks = _ticks(y_coords)
    plt.yticks(y_coords, y_ticks)
    plt.ylim([0.4, 1.01])

    x_coords = np.linspace(np.min(x_coords), np.max(x_coords), 6)
    x_ticks = _ticks(x_coords)
    plt.xticks(x_coords, x_ticks)
    plt.xlim([x_coords[0] - 0.1, x_coords[-1] + 0.1])

    if path_to_figure is not None:
        plt.savefig(path_to_figure,  bbox_inches='tight', dpi=CONFIG.DPI)
    if show:
        plt.show()


def train_val_curves():

    show = False

    #target_result = 'MultiSURFSelection_XGBoost'
    #path_to_figure = './../../figures/best_model/train_val_ms_xgb.pdf'
    #path_to_results = './../../data_source/results/hpv_splitting/results_all_features_hpv_group_b.csv'

    target_result = 'FisherScoreSelection_LightGBM'
    path_to_figure = './../../figures/best_model/train_val_fs_lgbm.pdf'
    path_to_results = './../../data_source/results/dropped_corr/results_all_features_original_images_icc_dropped_correlated.csv'

    results = pd.read_csv(path_to_results, index_col=0)
    experiment = results[results['experiment_id'] == target_result]

    fig = plt.figure()
    _plot_train_valid(experiment, path_to_figure=path_to_figure, show=show)
    plt.ylabel('Average Weighted ROC AUC')
    plt.xlabel('Experiment ID')

    target_result = 'DummySelection_RidgeClassifier'
    path_to_figure = './../../figures/best_model/train_val_dummy_ridge.pdf'
    path_to_results = './../../data_source/results/hpv_splitting/results_all_features_original_images_hpv_group_b.csv'

    results = pd.read_csv(path_to_results, index_col=0)
    experiment = results[results['experiment_id'] == target_result]

    fig = plt.figure()
    _plot_train_valid(experiment, path_to_figure=path_to_figure, show=show)
    plt.ylabel('Average Weighted ROC AUC')
    plt.xlabel('Experiment ID')


def learning_curve():

    path_to_figure = './../../figures/best_model/learning_ms_xgb.pdf'

    train_sizes = np.linspace(0.4, 1.0, 37)

    learn_avg_test = np.load(
        './../../data_source/results/performance_curves/learning_curve/learning_curve_avg_test.npy'
    )
    learn_avg_train = np.load(
        './../../data_source/results/performance_curves/learning_curve/learning_curve_avg_train.npy'
    )
    learn_std_test = np.load(
        './../../data_source/results/performance_curves/learning_curve/learning_curve_std_test.npy'
    )
    learn_std_train = np.load(
        './../../data_source/results/performance_curves/learning_curve/learning_curve_std_train.npy'
    )
    learn_avg_test = np.squeeze(np.mean(learn_avg_test, axis=0))
    learn_avg_train = np.squeeze(np.mean(learn_avg_train, axis=0))
    learn_std_test = np.squeeze(np.mean(learn_std_test, axis=0))
    learn_std_train = np.squeeze(np.mean(learn_std_train, axis=0))

    palette = CONFIG.base_palette(n=4)

    plt.figure()
    plt.plot(
        train_sizes, learn_avg_train, color=palette[1],
        marker='o', markersize=5, label=f"Training score"
    )
    plt.plot(
        train_sizes, learn_avg_test, color=palette[2],
        linestyle='--', marker='s', label=f"Validation score"
    )
    plt.fill_between(
        train_sizes, learn_avg_train - learn_std_train, learn_avg_train + learn_std_train,
        alpha=0.15, color=palette[1]
    )
    plt.fill_between(
        train_sizes, learn_avg_test - learn_std_test, learn_avg_test + learn_std_test,
        alpha=0.15, color=palette[2]
    )
    plt.xlabel('Fraction of Training Set')
    plt.ylabel('Average Weighted ROC AUC')

    x_coords = np.linspace(0.6, 1.0, 6)
    x_ticks = _ticks(x_coords)
    plt.xticks(x_coords, x_ticks)

    y_coords = np.linspace(
        np.min([learn_avg_train - learn_std_train, learn_avg_test - learn_std_test]),
        np.max([learn_avg_train + learn_std_train, learn_avg_test + learn_std_test]),
        6
    )
    y_ticks = _ticks(y_coords)
    plt.yticks(y_coords, y_ticks)
    plt.xlim([0.62, 1.01])
    plt.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True,
        shadow=True
    )
    plt.savefig(path_to_figure, bbox_inches='tight', dpi=CONFIG.DPI)



def train_val_statistics():

    target_result = 'MultiSURFSelection_XGBoost'
    path_to_figure = './../../figures/best_model/train_val_ms_xgb.pdf'
    path_to_results = './../../data_source/results/hpv_splitting/results_all_features_hpv_group_b.csv'

    results = pd.read_csv(path_to_results, index_col=0)
    experiment = results[results['experiment_id'] == target_result]['test_score']

    print('XGB TEST')
    print(np.min(experiment))
    print(np.std(experiment))
    print(np.mean(experiment))
    print(np.max(experiment))
    print()
    experiment = results[results['experiment_id'] == target_result]['train_score']

    print('XGB TRAIN')
    print(np.min(experiment))
    print(np.std(experiment))
    print(np.mean(experiment))
    print(np.max(experiment))
    print()


    target_result = 'DummySelection_RidgeClassifier'
    path_to_figure = './../../figures/best_model/train_val_dummy_ridge.pdf'
    path_to_results = './../../data_source/results/hpv_splitting/results_all_features_original_images_hpv_group_b.csv'

    results = pd.read_csv(path_to_results, index_col=0)
    experiment = results[results['experiment_id'] == target_result]['test_score']

    print('RIDGE TEST')
    print(np.min(experiment))
    print(np.std(experiment))
    print(np.mean(experiment))
    print(np.max(experiment))

    experiment = results[results['experiment_id'] == target_result]['train_score']

    print('RIDGE TRAIN')
    print(np.min(experiment))
    print(np.std(experiment))
    print(np.mean(experiment))
    print(np.max(experiment))


def get_hyperarameters():

    path_to_results = './../../data_source/results/dropped_corr/results_all_features_original_images_icc_dropped_correlated.csv'
    target_model = 'FisherScoreSelection_LightGBM'

    results = pd.read_csv(path_to_results, index_col=0)
    results = results[results['experiment_id'] == target_model]

    # NOTE: Assumes all parameters can be averaged. Otherwise use Counter for a
    # majority vote.
    for col in results:
        fs, clf = target_model.split('_')
        if fs in col:
            var = results.loc[:, col].values
            print(col, np.mean(var))
            #Counter(var)
        elif clf in col:
            var = results.loc[:, col].values
            print(col, np.mean(var))


def lgbm_ranks():
    # Trains a LGBM model and calculates SHAP values.

    path_to_features = '../../data_source/to_analysis/compressed_features/all_features_orig_images_icc_dropped.csv'
    path_to_target = '../../data_source/to_analysis/target_dfs.csv'
    params = {
        'learning_rate': 8.50431339312885,
        'max_depth': 165,
        'min_data_in_leaf': 3,
        'n_estimators': 101,
        'reg_alpha': 11.187043594443924,
        'reg_lambda': 49.295226272075844
    }
    model = LGBMClassifier(
        boosting_type='gbdt',
        class_weight='balanced',
        objective='binary',
        n_jobs=1,
        **params
    )
    X = pd.read_csv(path_to_features, index_col=0)
    y = np.squeeze(pd.read_csv(path_to_target, index_col=0).values)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    model.fit(X_std, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    np.save('lgbm_shap_values.npy', shap_values)


def plot_impact_num_exp_reps():

    show = True
    #path_to_results = './../../data_source/results/original_images/results_all_features_original_images.csv'
    #path_to_results = './../../data_source/results/dropped_corr/results_all_features_original_images_icc_dropped_correlated.csv'
    path_to_results = './../../data_source/results/hpv_splitting/results_all_features_original_images_hpv_group_b.csv'

    target_model = 'MultiSURFSelection_XGBoost'
    #target_model = 'FisherScoreSelection_LightGBM'

    results = pd.read_csv(path_to_results, index_col=0)
    results = results[results['experiment_id'] == target_model]
    scores = results['test_score'].values

    means = []
    for num in range(1, len(scores) + 1):
        avg_score = np.mean(scores[0:num])
        means.append(avg_score)

    plt.figure()
    plt.plot(means)

    if show:
        plt.show()


if __name__ == '__main__':
    #train_val_curves()
    #learning_curve()
    plot_impact_num_exp_reps()
