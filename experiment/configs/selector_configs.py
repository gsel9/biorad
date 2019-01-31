# -*- coding: utf-8 -*-
#
# selector_configs.py
#

"""
Feature selection algorithm setup including hyperparameter configurations.

NB: Make sure to update the number of original features in the data set.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import sys
sys.path.append('./..')

from backend import hyperparams

from hyperopt.pyll import scope

from backend.feature_selection import MutualInformationSelection
from backend.feature_selection import PermutationSelection
from backend.feature_selection import WilcoxonSelection
#from backend.feature_selection import FeatureScreening
from backend.feature_selection import ReliefFSelection
from backend.feature_selection import MRMRSelection

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Globals
SCREENER_LABEL = 'screener'
SELECTOR_LABEL = 'selector'
NUM_ORIG_FEATURES = 200


@scope.define
def screener_name_func(param_name):
    # Label hyperparameters associated with screener.
    global SCREENER_LABEL

    return '{}__{}'.format(SCREENER_LABEL, param_name)


@scope.define
def selector_name_func(param_name):
    # Label hyperparameters associated with selector.
    global SELECTOR_LABEL

    return '{}__{}'.format(SELECTOR_LABEL, param_name)


def _setup_hparam_space(param_configs):
    # Combine multiple hyperparameter spaces.
    hparams = {}
    for config in param_configs:
        hparams.update(config)

    return hparams


selectors = {
    # Random forest classifier permutation importance selection:
    # * Permutation procedure specific parameters are not part of the
    #   optimization objective.
    # * Specify classifier hyperparamters as the only parameters part of the
    #   optimization problem. These parameters are passed to the classifier
    #   throught the set_params method.
    # * Not performing repeated feature permutations of each feature because
    #   the similar effect may be achieved by repeating the procedure for
    #   different random states and averaging the result accross the repeated
    #   experiments.
    MutualInformationSelection.__name__: {
        'selector': [
            (SCREENER_LABEL, MRMRSelection()),
            ('{}_scaler'.format(SELECTOR_LABEL), StandardScaler()),
            (
                SELECTOR_LABEL,
                MutualInformationSelection()
            )
        ],
        'params': _setup_hparam_space(
            [
                hyperparams.mrmr_hparam_space(
                    selector_name_func,
                    k=None,
                    num_features=None,
                    max_num_features=NUM_ORIG_FEATURES
                ),
                hyperparams.mutual_info_param_space(
                    selector_name_func,
                    num_features=None,
                    max_num_features=NUM_ORIG_FEATURES
                )
            ]
        ),
    },
    'rf_{}'.format(PermutationSelection.__name__): {
        'selector': [
            (SCREENER_LABEL, MRMRSelection()),
            ('{}_scaler'.format(SELECTOR_LABEL), StandardScaler()),
            (
                SELECTOR_LABEL, PermutationSelection(
                    model=RandomForestClassifier(
                        n_jobs=-1,
                        verbose=False,
                        oob_score=False,
                        class_weight='balanced'
                    ),
                    score_func=roc_auc_score,
                    test_size=0.15,
                    num_rounds=2,
                )
            )
        ],
        'params': _setup_hparam_space(
            [
                hyperparams.mrmr_hparam_space(
                    selector_name_func,
                    k=None,
                    num_features=None,
                    max_num_features=NUM_ORIG_FEATURES
                ),
                hyperparams.trees_param_space(
                    selector_name_func,
                    min_samples_split=None,
                    min_samples_leaf=None,
                    n_estimators=None,
                    max_features=None,
                    random_state=None,
                    bootstrap=None,
                    max_depth=None,
                ),
            ]
        )
    },
    'logreg_{}'.format(PermutationSelection.__name__): {
        'selector': [
            (SCREENER_LABEL, MRMRSelection()),
            ('{}_scaler'.format(SELECTOR_LABEL), StandardScaler()),
            (
                SELECTOR_LABEL, PermutationSelection(
                    model=LogisticRegression(
                        solver='liblinear',
                        max_iter=1000,
                        verbose=0,
                        n_jobs=1,
                        dual=False,
                        multi_class='ovr',
                        warm_start=False,
                        class_weight='balanced',
                    ),
                    score_func=roc_auc_score,
                    test_size=0.15,
                    num_rounds=2,
                )
            )
        ],
        'params': _setup_hparam_space(
            [
                hyperparams.mrmr_hparam_space(
                    selector_name_func,
                    k=None,
                    num_features=None,
                    max_num_features=NUM_ORIG_FEATURES
                ),
                hyperparams.logreg_hparam_space(
                    selector_name_func,
                    penalty='l1',
                    C=None,
                    tol=None,
                    random_state=None,
                    fit_intercept=True,
                    intercept_scaling=None,
                )
            ]
        )
    },
    # Wilcoxon feature selection:
    WilcoxonSelection.__name__: {
        'selector': [
            (SCREENER_LABEL, MRMRSelection()),
            ('{}_scaler'.format(SELECTOR_LABEL), StandardScaler()),
            (
                SELECTOR_LABEL,
                WilcoxonSelection(thresh=0.05, bf_correction=True)
            )
        ],
        'params': _setup_hparam_space(
            [
                hyperparams.mrmr_hparam_space(
                    selector_name_func,
                    k=None,
                    num_features=None,
                    max_num_features=NUM_ORIG_FEATURES
                ),
            ]
        ),
    },
    # ReliefF feature selection:
    ReliefFSelection.__name__: {
        'selector': [
            (SCREENER_LABEL, MRMRSelection()),
            ('{}_scaler'.format(SELECTOR_LABEL), StandardScaler()),
            (SELECTOR_LABEL, ReliefFSelection())
        ],
        'params': _setup_hparam_space(
            [
                hyperparams.mrmr_hparam_space(
                    selector_name_func,
                    k=None,
                    num_features=None,
                    max_num_features=NUM_ORIG_FEATURES
                ),
                hyperparams.relieff_hparam_space(
                    selector_name_func,
                    num_neighbors=None,
                    num_features=None,
                    max_num_features=NUM_ORIG_FEATURES
                ),
            ]
        )
    },
}

"""
# Maximum relevance minimum redundancy selection:
MRMRSelection.__name__: {
    'selector': [
        (SCREENER_LABEL, MRMRSelection()),
        ('{}_scaler'.format(SELECTOR_LABEL), StandardScaler()),
        (SELECTOR_LABEL, MRMRSelection())
    ],
    'params': _setup_hparam_space(
        [
            hyperparams.feature_screening_hparam_space(
                screener_name_func,
                var_thresh=None,
                num_features=None,
                max_num_features=NUM_ORIG_FEATURES
            ),
            hyperparams.mrmr_hparam_space(
                selector_name_func,
                k=None,
                num_features=None,
                max_num_features=NUM_ORIG_FEATURES
            ),
        ]
    )
}
"""


if __name__ == '__main__':

    print(selectors)
