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

# https://github.com/automl/SMAC3/blob/master/examples/svm.py
def svc_from_config(config):
    pass


def gnb_from_config(config):
    pass


class FeatureSelector(TransformerMixin):
    """Representation of a feature selection algorithm.

    Args:
        name (str): Name of feature selection procedure.
        func (function): The feature selection procedure.
        params (dict): Parameters passed to the feature selection function.

    Returns:
        (tuple): Training subset, test subset and selected features support
            indicators.

    """

    def __init__(self, name, func, params, random_state):

        self.name = name
        self.func = func
        self.random_state = random_state

        # NOTE:
        self._X_train = None
        self._X_test = None
        self._support = None

    def fit(self, X, y, **kwargs):

        # Execute feature selection procedure.
        self._X_train, self._X_test, self._support = self.func(X, y, **kwargs)

        return self

    def transform(self):
        """"""

        # Formatting of output includes error handling.
        support = self._check_support(self._support, self._X_train)

        # Support should be a non-empty vector (ensured by _check_support).
        return utils.check_train_test(
            self._X_train[:, support],  self._X_test[:, support]
        )

    def fit_transform(self, X, y, **kwargs):
        """Perform fitting and transformation in a single call.

        Args:
            ():

        Returns:
            ():

        """

        self.fit(X, y, **kwargs)

        return self.transform()
