
"""
Note: joblib 0.12.2 restarts workers when a memory leak is detected.
"""

import re
import os

import pandas as pd


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
        num_votes
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
