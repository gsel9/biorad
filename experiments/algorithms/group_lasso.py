import numpy as np

from copy import deepcopy

from scipy.special import expit

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter


class GroupLASSO(BaseEstimator, RegressorMixin):
    """
    Reference:
        Friedman, Hastie, Tibshirani (2010). Regularization Paths for Generalized
        Linear Models via Coordinate Descent, J Statistical Software.

    """
    def __init__(
        self,
        group_idx=None,
        reg_lambda=0.005,
        learning_rate=0.005,
        max_iter=2000,
        tol=1e-4,
        random_state=0
    ):

        self.reg_lambda = reg_lambda
        self.group_idx = group_idx
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.tol = tol
        self.random_state = random_state

        # NOTE:
        self.beta0_ = None
        self.beta_ = None
        self.ynull_ = None

    def copy(self):
        return deepcopy(self)

    @property
    def config_space(self):
        """LightGBM hyperparameter space."""

        # L2 regularization term on weights.
        reg_lambda = UniformFloatHyperparameter(
            'reg_lambda', lower=1e-8, upper=100, default_value=1e-3
        )
        learning_rate = UniformFloatHyperparameter(
            'learning_rate', lower=1e-8, upper=50, default_value=0.01
        )
        # Add hyperparameters to config space.
        config = ConfigurationSpace()
        config.seed(self.random_state)
        config.add_hyperparameters((reg_lambda, learning_rate))

        return config

    def _grad_L2loss(self, X, y):
        """The gradient."""

        num_samples, num_features = np.shape(X)
        scale = 1.0 / num_samples

        z = self.beta0_ + np.dot(X, self.beta_)
        grad_beta0 = np.sum(expit(z) - y) * scale
        grad_beta = np.dot((expit(z) - y).T, X).T * scale

        grad_beta = grad_beta + self.reg_lambda

        gradient = np.zeros(num_features + 1, dtype=np.float64)
        gradient[0] = grad_beta0
        gradient[1:] = grad_beta

        return gradient

    def _lambda(self, X):
        """Conditional intensity function."""

        z = self.beta0_ + np.dot(X, self.beta_)
        return expit(z)

    def _proximal_op(self, beta, thresh):
        """Proximal operator."""

        # Apply group sparsity operator.
        group_norms = np.abs(beta)
        groups = np.unique(self.group_idx)
        for group_id in groups:
            if group_id != 0:
                _norm = np.linalg.norm(beta[self.group_idx == group_id], ord=2)
                group_norms[self.group_idx == group_id] = _norm

        zero_norms = group_norms > 0.0
        over_thresh = group_norms > thresh
        to_update = zero_norms & over_thresh

        result = beta
        change = thresh * beta[to_update] / group_norms[to_update]
        result[to_update] = beta[to_update] - change
        result[~to_update] = 0.0

        return result

    def fit(self, X, y):
        """
        """
        if self.group_idx is None:
            raise ValueError('Missing feature group indicators!')

        _, num_features = np.shape(X)
        if np.size(self.group_idx) != num_features:
            raise ValueError(f'Require {num_features} indices, not'
                             f'{np.size(self.group_idx)}.')

        # Initialize parameters.
        if self.beta0_ is None and self.beta_ is None:
            rng = np.random.RandomState(self.random_state)
            self.beta0_ = 1 / (num_features + 1) * rng.normal(0.0, 1.0, 1)
            self.beta_ = 1 / (num_features + 1) * rng.normal(0.0, 1.0, num_features)

        self.group_idx = np.array(self.group_idx, dtype=np.int32)
        for epoch in range(0, self.max_iter):
            # Calculate gradient.
            grad = self._grad_L2loss(X, y)

            # Check for convergence.
            if (epoch > 1) and (np.linalg.norm(grad) < self.tol):
                print(f'Conveged after {epoch} iterations.')
                break

            # Update coefficients.
            self.beta_ = self.beta_ - self.learning_rate * grad[1:]
            self.beta0_ = self.beta0_ - self.learning_rate * grad[0]

            # Apply proximal operator.
            self.beta_ = self._proximal_op(self.beta_, self.reg_lambda)

        # Update the estimated variables.
        self.ynull_ = np.mean(y)

        return self

    def predict(self, X):

        return self._lambda(X)


if __name__ == '__main__':
    m = GroupLASSO()
    m.set_params(**{'reg_lambda': 0.01})
    print(m.get_params())
