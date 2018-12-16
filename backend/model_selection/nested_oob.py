# -*- coding: utf-8 -*-
#
# model_selection.py
#
# TODO:
# * Optional feature selection
# * Select from median (not mean)
# * Default mechanism to return None if error occurs.
# * Separate directoris with model copmarison schemes. One module per scheme.

"""
Frameworks for performing model selection.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import utils
import ioutil
import model_selection
import feature_selection

import numpy as np
import pandas as pd

from datetime import datetime
from collections import OrderedDict
from sklearn.externals import joblib


def nested_oob(*args, verbose=1, n_jobs=1, score_func=None):
    """Nested bootstrap Out-of-Bag."""
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    # Setup:
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    # Read prelim results if available.
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        if verbose > 0:
            print('Reloading previous results')
    # Compute new results.
    else:
        if verbose > 0:
            start_time = datetime.now()
            print('Entering nested procedure with ID: {}'.format(random_state))
        results = _nested_oob(
            *args, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        if verbose > 0:
            delta_time = datetime.now() - start_time
            print('Collected results in: {}'.format(delta_time))

    return results


def _nested_oob():

    pass
