import sys
sys.path.append('./..')
sys.path.append('./../../model_comparison')

import os
import pytest
import backend
import model_selection
import comparison_frame

from selector_configs import selectors
from estimator_configs import classifiers

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    import sys
    sys.path.append('./../experiment')

    import os
    import backend
    import model_selection
    import comparison

    from selector_configs import selectors
    from estimator_configs import classifiers

    from hyperopt import tpe

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import precision_recall_fscore_support

    # TEMP:
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)

    # SETUP:
    CV = 3
    OOB = 10
    MAX_EVALS = 7
    SCORING = roc_auc_score
    #
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    name = 'PermutationSelection_LogisticRegression'
    pipe, params = pipes_and_params[name]

    results = bbc_cv_selection(
        X, y,
        tpe.suggest,
        name,
        pipe,
        params,
        SCORING,
        CV,
        OOB,
        MAX_EVALS,
        shuffle=True,
        verbose=0,
        random_state=0,
        alpha=0.05,
        balancing=True,
        error_score=np.nan,
        path_tmp_results=None,
    )
    print(results['oob_median_score'])
