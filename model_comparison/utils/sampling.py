# -*- coding: utf-8 -*-
#
# utils.py
#

"""
Model comparison framework utility functions.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np

from numba import int32
from numba import jitclass
from numba import jit

from collections import OrderedDict

from imblearn import over_sampling
from sklearn.preprocessing import StandardScaler


class OOBSampler:
    """A bootstrap Out-of-Bag resampler.

    Args:
        n_splits (int): The number of resamplings to perform.
        random_state (int): Seed for the pseudo-random number generator.

    """

    def __init__(self, n_splits, random_state):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, **kwargs):
        """Generates Out-of-Bag samples.

        Args:
            X (array-like): The predictor data.
            y (array-like): The target data.

        Returns:
            (generator): An iterable with X and y sample indicators.

        """
        rgen = np.random.RandomState(self.random_state)

        nrows, _ = np.shape(X)
        sample_indicators = np.arange(nrows)
        for _ in range(self.n_splits):
            train_idx = rgen.choice(
                sample_indicators, size=nrows, replace=True
            )
            # Oberervations not part of training set defines test set.
            test_idx = np.array(
                list(set(sample_indicators) - set(train_idx)), dtype=int
            )
            yield train_idx, test_idx


def balance_data(X, y, random_state):
    """Balance distribution of target classes accoring to the

    Args:
        X (array-like):
        y (array-like):
        random_state (int):

    Returns:
        (tuple): Balanced predictor and target data sets.

    """

    raise NotImplementedError('balancing not implemented!')


def _balance_data(X, y, random_state):
    """Balance distribution of target classes with the Synthetic Minority
    Oversmapling Technique proposed by ...

    Args:
        X (array-like):
        y (array-like):
        random_state (int):

    Returns:
        (tuple): Balanced predictor and target data sets.

    """

    balancer = over_sampling.SMOTE(random_state=random_state)
    _X, _y = balancer.fit_sample(X, y)

    return _X, _y
