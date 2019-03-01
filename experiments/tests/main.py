"""
Test run for model comparison experiment pipeline functionality.
"""


import sys
sys.path.append('./../')
sys.path.append('./../../model_comparison')

import comparison
import model_selection

from sklearn.base import BaseEstimator

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import numpy as np
import pandas as pd


class SphereFunction:
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization

    SEED = 0
    NAME = 'SphereFunction'

    def __init__(self, vars=None):

        super().__init__()

        self.vars = vars
        self.loss = None

    def __name__(self):

        return self.NAME

    @property
    def config_space(self):
        """Returns the RF Regression hyperparameter space."""

        vars = UniformFloatHyperparameter(
            'parameter', lower=1e-6, upper=10,
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.SEED)
        config.add_hyperparameter(vars)

        return config

    def set_params(self, **params):

        self.vars = params

        return self

    def get_params(self, deep: bool=True):

        return {'vars': self.vars}

    def transform(self, X, y=None, **kwargs):

        return X

    def fit(self, X, y=None, **kwargs):

        print(self.vars)

        value = list(self.vars.values())[0]
        self.loss = np.sum(value)

        return self

    def predict(self, X):

        return self.loss


def dummy_metric(y_true, y_pred):
    """Returns the loss calculated by the sphere function."""

    return y_pred


def load_target(path_to_target, index_col=0, classify=True):
    """

    """
    var = pd.read_csv(path_to_target, index_col=index_col)
    if classify:
        return np.squeeze(var.values).astype(np.int32)
    else:
        return np.squeeze(var.values).astype(np.float32)


# TODO: To utils!
def load_predictors(path_to_data, index_col=0, regex=None):
    """

    """
    data = pd.read_csv(path_to_data, index_col=index_col)
    if regex is None:
        return np.array(data.values, dtype=np.float32)
    else:
        target_features = data.filter(regex=regex)
        return np.array(data.loc[:, target_features].values, dtype=np.float32)


def test_experiment(_):

    path_to_results = './test_experiment.csv'
    y = load_target('./../../../data_source/to_analysis/target_dfs.csv')
    X = load_predictors('./../../../data_source/to_analysis/no_filter_concat.csv')

    model = {'test':
        (
            (f'{SphereFunction.__name__}_1', SphereFunction()),
            (f'{SphereFunction.__name__}_2', SphereFunction()),
        )
    }

    np.random.seed(seed=2019)
    random_states = np.random.choice(1000, size=5)

    comparison.model_comparison(
        comparison_scheme=model_selection.model_selection,
        X=X, y=y,
        experiments=model,
        score_func=dummy_metric,
        cv=5,
        write_prelim=False,
        max_evals=100,
        output_dir='./parameter_search',
        random_states=random_states,
        path_final_results=path_to_results,
        verbose=1
    )


if __name__ == '__main__':
    test_experiment(None)
