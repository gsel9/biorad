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

from backend import hyperparams

from backend.feature_selection import RFPermutationSelection
from backend.feature_selection import WilcoxonSelection
from backend.feature_selection import ReliefFSelection
from backend.feature_selection import MRMRSelection

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from hyperopt.pyll import scope


# Globals
CLF_LABEL = 'selector'
# NB WIP:
NUM_ORIG_FEATURES = 10
NAME_FUNC = lambda param_name: '{}__{}'.format(CLF_LABEL, param_name)


@scope.define
def sklearn_roc_auc_score(*args, **kwargs):
    """Wrapper for sklearn ROC AUC classifier performance metric function."""

    return roc_auc_score(*args, **kwargs)


selectors = {
    # Random forest classifier permutation importance selection.
    RFPermutationSelection.__name__: {
        # NOTE: Algorithm wraps a Random Forest Classifier with associated
        # hyperparams as part of the feature selection optimization problem.
        'pipe': [
            (CLF_LABEL, RFPermutationSelection())
        ],
        # Mergeing of permutation importance procedure parameters with
        # wrapped RF classifier hyperparameters (rendering the RF
        # hyperparamters part of the TPE classification problem) occurs in
        # the hyperparams rf_permutation_param_space backend function.
        'params': hyperparams.rf_permutation_param_space(
            procedure_params = {
                NAME_FUNC('score_func'): sklearn_roc_auc_score,
                NAME_FUNC('num_rounds'): 10,
                NAME_FUNC('test_size'): 0.2,
            },
            model_params=hyperparams.trees_param_space(
                NAME_FUNC,
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
        'pipe': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, WilcoxonSelection(
                thresh=0.05, bf_correction=True,
            ))
        ],
        'params': {},
    },
    # ReliefF feature selection
    ReliefFSelection.__name__: {
        'pipe': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, ReliefFSelection())
        ],
        'params': hyperparams.relieff_hparam_space(
            NAME_FUNC,
            num_neighbors=None,
            num_features=None,
            max_num_features=NUM_ORIG_FEATURES
        ),
    },
    # Maximum relevance minimum redundancy selection
    MRMRSelection.__name__: {
        'pipe': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, MRMRSelection())
        ],
        'params': hyperparams.mrmr_hparam_space(
            NAME_FUNC,
            k=None,
            num_features=None,
            max_num_features=NUM_ORIG_FEATURES
        ),
    }
}
