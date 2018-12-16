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
