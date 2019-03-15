# -*- coding: utf-8 -*-
#
# ioutil.py
#

"""
Results handling utility functions for model comparison experiments.
"""

import os
import csv
import shutil

from collections import OrderedDict

import pandas as pd
import numpy as np


def load_target_to_ndarray(path_to_file, index_col=0, classification=True):
    """Load target vector from file.

    """
    df_y = pd.read_csv(path_to_file, index_col=index_col)
    if classification:
        return np.squeeze(df_y.values).astype(np.int32)

    return np.squeeze(df_y.values).astype(np.float32)


def load_predictors_to_ndarray(path_to_file, index_col=0, regex=None):
    """Load predictor matrix from file.

    """
    df_X = pd.read_csv(path_to_file, index_col=index_col)
    if regex is None:
        return np.array(df_X.values, dtype=np.float32)

    target_features = df_X.filter(regex=regex)
    return np.array(df_X.loc[:, target_features].values, dtype=np.float32)


def write_final_results(path_to_file, results):
    """Write the total collection of results to disk."""

    if isinstance(results, pd.DataFrame):
        results.to_csv(path_to_file, columns=results.columns)
    else:
        data = pd.DataFrame([result for result in results])
        data.to_csv(path_to_file)


def setup_tempdir(tempdir, root=None):
    """Returns path and sets up directory if non-existent."""

    if root is None:
        root = os.getcwd()

    path_tempdir = os.path.join(root, tempdir)

    if not os.path.isdir(path_tempdir):
        os.mkdir(path_tempdir)

    return path_tempdir


def teardown_tempdir(path_to_dir):
    """Removes directory even if not empty."""

    shutil.rmtree(path_to_dir)


def read_prelim_result(path_to_file):
    """Read temporary stored results."""

    results = pd.read_csv(path_to_file, index_col=False)

    return OrderedDict(zip(*(results.columns, results.values[0])))


def write_prelim_results(path_to_file, results):
    """Store results in temporary separate files to prevent write conflicts."""

    with open(path_to_file, 'w') as outfile:
        writer = csv.DictWriter(
            outfile, fieldnames=list(results.keys()), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerow(results)
