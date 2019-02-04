import sys
sys.path.append('./..')
sys.path.append('./../../model_comparison')

import os
import pytest
import backend
import model_selection
import comparison_frame

from selector_configs import selectors
from estimator_configs import classifiers

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split


SEED = 0


@pytest.fixture
def data(test_size=0.2):

    global SEED

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def pipes_and_params():

    return backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )


def test_pipe_format(pipes_and_params):

    for label, (pipe, params) in pipes_and_params.items():
        pass

def test_pipe_fit(data):

    X_train, _, y_train, _ = data

    for label, (pipe, params) in pipes_and_params.items():
        pipe.fit(X_train, y_train)


def test_pipe_fit_predict(data):

    X_train, X_train, y_train, y_test = data

    for label, (pipe, params) in pipes_and_params.items():
        pipe.fit(X_train, y_train)
        pipe.predict(X_test)


if __name__ == '__main__':

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    for label, (pipe, params) in pipes_and_params.items():

        pipe.fit(X_train, y_train)
        #pipe.predict(X_test)
        """
        print()
        print()
        print('3' * 30)
        print(pipe)
        print(pipe.get_params())
        print('3' * 30)
        print()
        print()
        """
