# From: Dependence Guided Unsupervised Feature Selection, Guo & Zhu.

import numpy as np

from utils import screening_utils

from scipy.spatial.distance import pdist, squareform

from sklearn.base import BaseEstimator, TransformerMixin

"""
The implementation is based on the MATLAB code:
https://github.com/eeGuoJun/AAAI2018_DGUFS/blob/master/JunGuo_AAAI_2018_DGUFS_code/files/speedUp.m

"""


def feature_screening():
    # PArallelizable work func for feature screening.
    pass
