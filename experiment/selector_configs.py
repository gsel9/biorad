# -*- coding: utf-8 -*-
#
# selector_configs.py
#

"""
To do's:
* Determine the intial number of features to select from.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


from hyperopt.pyll import scope

from backend import hyperparams

from backend.feature_selection import PermutationSelectionRF
from backend.feature_selection import WilcoxonSelection
from backend.feature_selection import ReliefFSelection
from backend.feature_selection import MRMRSelection

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Globals
CLF_LABEL = 'selector'
# NB WIP: The initial number of features to select from.
NUM_ORIG_FEATURES = 10


@scope.define
def selector_name_func(param_name):

    global CLF_LABEL

    return '{}__{}'.format(CLF_LABEL, param_name)


@scope.define
def sklearn_roc_auc_score(*args, **kwargs):
    """Wrapper for sklearn ROC AUC classifier performance metric function."""

    return roc_auc_score(*args, **kwargs)


selectors = {
        # Random forest classifier permutation importance selection.
    PermutationSelectionRF.__name__: {
        # NOTE: Algorithm wraps a Random Forest Classifier with associated
        # hyperparams as part of the feature selection optimization problem.
        'selector': [
            (CLF_LABEL, PermutationSelectionRF())
        ],
        # Mergeing of permutation importance procedure parameters with
        # wrapped RF classifier hyperparameters (rendering the RF
        # hyperparamters part of the TPE classification problem) occurs in
        # the hyperparams rf_permutation_param_space backend function.
        'params': hyperparams.permutation_param_space(
            procedure_params = {
                selector_name_func('score_func'): sklearn_roc_auc_score,
                selector_name_func('num_rounds'): 10,
                selector_name_func('test_size'): 0.2,
                #selector_name_func('model'): RandomForestClassifier()
                selector_name_func('random_state'): 2,
            },
            model_params=hyperparams.trees_param_space(
                selector_name_func,
                n_estimators=None,
                max_features=None,
                max_depth=None,
                min_samples_split=None,
                min_samples_leaf=None,
                bootstrap=None,
                oob_score=False,
                n_jobs=-1,
                verbose=False,
            )
        ),
    },
    # Wilcoxon feature selection
    WilcoxonSelection.__name__: {
        'selector': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, WilcoxonSelection(
                thresh=0.05, bf_correction=True,
            ))
        ],
        'params': {},
    },
    # ReliefF feature selection
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
    # Maximum relevance minimum redundancy selection
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
