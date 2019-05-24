# -*- coding: utf-8 -*-
#
# comparison_schemesn.py
#

"""
Schemes of model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os

from copy import deepcopy
from collections import OrderedDict
from datetime import datetime

from sklearn.utils import check_X_y
from sklearn.model_selection import StratifiedKFold

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

import numpy as np

# Local imports.
from utils import ioutil


from comparison_schemes import SMACSearchCV


def workflow():

    pass


def main():

    path_to_results = ''
    path_to_predictors = ''
    path_to_target = ''

    X =
    y =

    optimizer = SMACSearchCV(
        cv=cv,
        experiment_id=experiment_id,
        workflow=pipeline_inner,
        hparam_space=hparam_space,
        max_evals=max_evals,
        score_func=score_func,
        random_state=random_state,
        verbose=verbose,
        output_dir=output_dir
    )
    best_config = optimizer.fit(X, y)

    # Write results to file.



if __name__ == '__main__':
    main()
