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


from backend import hyperparams

from hyperopt.pyll import scope

from backend.feature_selection import PermutationSelection
from backend.feature_selection import WilcoxonSelection
from backend.feature_selection import ReliefFSelection
from backend.feature_selection import MRMRSelection

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Globals
CLF_LABEL = 'selector'
NUM_ORIG_FEATURES = 607


@scope.define
def selector_name_func(param_name):

    global CLF_LABEL

    return '{}__{}'.format(CLF_LABEL, param_name)


selectors = {
    # Random forest classifier permutation importance selection:
    # * Num features ensures equal number of features selected in ecah fold.
    # * Permutation procedure specific parameters are not part of the
    #   optimization objective.
    # * Specify classifier hyperparamters as the only parameters part of the
    #   optimization problem. These parameters are passed to the classifier
    #   throught the set_params method.
    # * Not performing repeated feature permutations of each feature because
    #   the similar effect may be achieved by repeating the procedure for
    #   different random states and averaging the result accross the repeated
    #   experiments.
    PermutationSelection.__name__: {
        'selector': [
            (CLF_LABEL, PermutationSelection(
                    model=RandomForestClassifier(
                        n_jobs=-1, verbose=False, oob_score=False,
                    ),
                    score_func=roc_auc_score,
                    num_rounds=1,
                    test_size=0.2,
                )
            )
        ],
        'params': permutation_hparam_space(
            selector_name_func,
            num_features=None,
            hyperparams.trees_param_space(
                selector_name_func,
                n_estimators=None,
                max_features=None,
                max_depth=None,
                min_samples_split=None,
                min_samples_leaf=None,
                bootstrap=None,
                random_state=None,
            ),
        )
    },
    # Wilcoxon feature selection:
    # * Num features ensures equal number of features selected in ecah fold.
    WilcoxonSelection.__name__: {
        'selector': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, WilcoxonSelection(thresh=0.05))
        ],
        'params': hyperparams.wilcoxon_hparam_space(
            selector_name_func, num_features=None
        )
    },
    # ReliefF feature selection:
    # * Num features ensures equal number of features selected in ecah fold.
    ReliefFSelection.__name__: {
        'selector': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, ReliefFSelection())
        ],
        'params': hyperparams.relieff_hparam_space(
            selector_name_func,
            num_neighbors=None,
            num_features=None,
            max_num_features=NUM_ORIG_FEATURES
        ),
    },
    # Maximum relevance minimum redundancy selection:
    # * Num features ensures equal number of features selected in ecah fold.
    MRMRSelection.__name__: {
        'selector': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, MRMRSelection())
        ],
        'params': hyperparams.mrmr_hparam_space(
            selector_name_func,
            k=None,
            num_features=None,
            max_num_features=NUM_ORIG_FEATURES
        ),
    }
}
