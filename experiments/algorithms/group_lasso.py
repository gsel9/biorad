"""Python implementation of elastic-net regularized GLMs."""

import numpy as np

from copy import deepcopy

from scipy.special import expit
from scipy.stats import norm

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


class GroupLASSO(BaseEstimator):
    """
    Reference:
        Friedman, Hastie, Tibshirani (2010). Regularization Paths for Generalized
        Linear Models via Coordinate Descent, J Statistical Software.

    """
    def __init__(
        self,
        alpha=0.5,
        group_idx=None,
        reg_lambda=0.005,
        learning_rate=0.005,
        max_iter=2000,
        tol=1e-4,
        random_state=0
    ):

        self.alpha = alpha
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

    def _grad_L2loss(self, X, y):
        """The gradient."""

        num_samples, num_features = np.shape(X)
        num_coeffs = np.size(self.beta_)

        inv_cov = np.dot(np.eye(num_coeffs).T, np.eye(num_coeffs))
        inv_cov_beta = np.dot(inv_cov, self.beta_)

        z = self.beta0_ + np.dot(X, self.beta_)
        grad_beta0 = np.sum(expit(z) - y) / num_samples
        grad_beta = np.dot((expit(z) - y).T, X).T / num_samples

        grad_beta += self.reg_lambda * (1 - self.alpha) * inv_cov_beta

        grad = np.zeros(num_features + 1, dtype=np.float64)
        grad[0] = grad_beta0
        grad[1:] = grad_beta

        return grad

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
        beta = np.zeros(num_features + 1, dtype=np.float64)
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
            self.beta_ = self._proximal_op(
                self.beta_, self.reg_lambda * self.alpha
            )
        # Update the estimated variables.
        self.ynull_ = np.mean(y)

        return self

    def predict(self, X):

        return self._lambda(X)
