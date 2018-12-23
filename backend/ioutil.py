
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
    """
    Auxillary function to update results and write preliminary results to
    disk as backup.
    """
    (
        results,
        test_scores,
        train_scores,
        gen_test_score,
        gen_train_scores,
        path_tempdir,
        random_state,
        estimator,
        best_params,
        num_votes,
        selector,
        best_support,
    ) = args
    # Update results dict.
    results.update(
        {
            'estimator': estimator.__name__,
            'selector': selector.name,
            'test_scores': test_scores,
            'train_scores': train_scores,
            'gen_test_score': gen_test_score,
            'gen_train_score': gen_train_scores,
            'params': best_params,
            'support': support,
            'size_support': len(support),
            'max_feature_votes': num_votes,
        }
    )
    # Write preliminary results to disk.
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector.name, random_state
        )
    )
    ioutil.write_prelim_results(path_case_file, results)

    return results
