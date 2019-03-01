"""
Test run for model comparison experiment pipeline functionality.
"""


from sklearn.base import BaseEstimator

import numpy as np


class SphereFunction(BaseEstimator):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization

    SEED = 0

    def __init__(self, params=None):

        super().__init__()

        self.loss = None
        self.params = params

    @property
    def config_space(self):
        """Returns the RF Regression hyperparameter space."""

        parameter = UniformFloatHyperparameter(
            'parameter', lower=1e-6, upper=10,
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameter(parameter

        return config

    def set_params(self, **params):

        self.params = params

        return self

    def get_params(self, deep: bool=True):

        return self.params

    def fit(self, X, y=None, **kwargs):

        self.loss = np.sum(np.power(self.params))

        return self

    def predict(self, X):

        return self.loss


def dummy_metric(y_true, y_pred):
    """Returns the loss calculated by the sphere function."""

    return y_pred


def test_experiment(_):
    pass


if __name__ == '__main__':
    test_experiment(None)
