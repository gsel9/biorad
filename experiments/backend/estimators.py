# -*- coding: utf-8 -*-
#
# estimators.py
#

"""
Construct pipelines comprising:
1. preprocessing
2. dimensionality reduction
3. postprocessing
4. classificationg

Notes:
* Each pipeline must have a unique ID.

"""


from sklearn.base import BaseEstimator, TransformerMixin


def estimator_from_config():
    pass


# https://github.com/automl/SMAC3/blob/master/examples/svm.py
def svc_from_config(config):
    pass


def gnb_from_config(config):
    pass


pipe1 = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('logit', LogisticRegression()),
])
