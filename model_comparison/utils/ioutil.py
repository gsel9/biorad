
import re
import os
import csv
import shutil

import numpy as np
import pandas as pd

from collections import OrderedDict


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


def write_prelim_results(path_to_file, results):
    """Store results in temporary separate files to prevent write conflicts."""

    with open(path_to_file, 'w') as outfile:
        writer = csv.DictWriter(
            outfile, fieldnames=list(results.keys()), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerow(results)

    return None


def update_prelim_results(*args):
    """Auxillary function to update results and write preliminary results to
    disk as backup."""
    (
        path_tmp_results,
        avg_train_score, avg_test_score,
        train_scores, test_scores,
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
            'avg_train_score': avg_train_score,
            'avg_test_score': avg_test_score,
            'train_scores': train_scores,
            'test_scores': test_scores,
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
