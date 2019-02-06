# -*- coding: utf-8 -*-
#
# agnostics.py
#

"""


"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'

import lime

import numpy as np

from lime import lime_tabular


# https://www.google.com/search?client=firefox-b-d&q=LimeTabularExplainer
# https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20continuous%20and%20categorical%20features.ipynb
def explainer():
    # Tabular explainers need a training set.  The reason is based on
    # calculation of feaure statistics. If the feature is numerical, the mean
    # and std are calculated, and discretize it into quartiles. If the feature
    # is categorical, the frequency of each value is calculated. The statistics
    # serves two purposes:
    #   1. To scale the data, so distances meaningfully be calculated
    #   2. To sample perturbed instances (sampling from a Normal(0, 1)),
    #      multiplying by the std and adding back the mean.

    _explainer = lime.lime_tabular.LimeTabularExplainer(
        train,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        discretize_continuous=True
    )
    i = 25
    exp = _explainer.explain_instance(test[i], rf.predict, num_features=5)
