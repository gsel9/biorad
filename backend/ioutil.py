
import re
import os
import csv
import shutil

import numpy as np
import pandas as pd



def write_final_results(path_to_file, results):
    """Write the total collection of results to disk."""

    if isinstance(results, pd.DataFrame):
        results.to_csv(path_to_file, columns=results.columns)
    else:
        data = pd.DataFrame([result for result in results])
        data.to_csv(path_to_file)

    return None


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

    return None


def read_prelim_result(path_to_file):
    """Read temporary stored results."""

    results = pd.read_csv(path_to_file, index_col=False)

    return OrderedDict(zip(*(results.columns, results.values[0])))


def update_prelim_results(*args):
    """Auxillary function to update results and write preliminary results to
    disk as backup."""
    (
        path_tmp_results,
        avg_train_precision, avg_test_precision,
        train_precisions, test_precisions,
        train_supports, test_supports,
        train_recalls, test_recalls,
        estimator_name, hparams,
        selector_name,
        support_votes,
        support,
        random_state,
        results
    ) = args
    # Update results dict.
    results.update(
        {
            'avg_train_precision': avg_train_precision,
            'avg_test_precision': avg_test_precision,
            'train_precisions': train_precisions,
            'test_precisions': test_precisions,
            'train_target_ratios': train_supports,
            'test_target_ratios': test_supports,
            'train_recalls': train_recalls,
            'test_recalls': test_recalls,
            'estimator': estimator_name,
            'hyperparameters': hparams,
            'selector_name': selector_name,
            'feature_votes': support_votes,
            'feature_support': support,
            'experiment_id': random_state,
        }
    )
    # Write preliminary results to disk.
    path_case_file = os.path.join(
        path_tmp_results, '{}_{}_{}'.format(
            estimator_name, selector_name, random_state
        )
    )
    write_prelim_results(path_case_file, results)

    return results
