# -*- coding: utf-8 -*-
#
# classification.py
#

"""
To do's:
* Algorithm setup including hyperparameters.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

from backend import hyperparams

from backend.feature_selection import PermutationSelection
from backend.feature_selection import WilcoxonSelection
from backend.feature_selection import
from backend.feature_selection import

from sklearn.metrics import roc_auc_score


# Globals
CLF_LABEL = 'selector'
NAME_FUNC = lambda param_name: '{}__{}'.format(CLF_LABEL, param_name),


selectors = {
    # Feature Permutation Importance Selection.
    PermutationSelection.__name__: {
        'pipe': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, PermutationSelection())
        ],
        # NOTE: Algorithm wraps a Random Forest Classifier with associated
        # hyperparams as part of the feature selection optimization problem.
        'params': hyperparams.permutation_param_space(
            NAME_FUNC,
            score_func=roc_auc_score,
            # NOTE: Number of repeated feature permutations is
            # arbitrarily/randomly chosen.
            num_rounds=10,
            # NOTE: Test size is arbitrarily/randomly chosen.
            test_size=0.2,
            model=RandomForestClassifier(),
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
        )
    },
    WilcoxonSelection.__name__: {
        'pipe': [
            ('{}_scaler'.format(CLF_LABEL), StandardScaler()),
            (CLF_LABEL, WilcoxonSelection(thresh=0.05))
        ],
        'params': {},
    },
}
