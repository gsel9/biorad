# From: Dependence Guided Unsupervised Feature Selection, Guo & Zhu.

import numpy as np

from utils import screening_utils

from scipy.spatial.distance import pdist, squareform

from sklearn.base import BaseEstimator, TransformerMixin

"""
The implementation is based on the MATLAB code:
https://github.com/eeGuoJun/AAAI2018_DGUFS/blob/master/JunGuo_AAAI_2018_DGUFS_code/files/speedUp.m

"""


# Checkout: https://github.com/eeGuoJun/AAAI2018_DGUFS/tree/master/JunGuo_AAAI_2018_DGUFS_code
class DGUFS(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        num_features,
        num_clusters,
        num_neighbors,
        reg_alpha,
        reg_beta,
        tol=5e-7,
        max_iter=1e2,
        mu=1e-6,
        max_mu=1e10,
        rho=1.1
    ):

    self.num_features = num_features
    self.num_clusters = num_clusters
    self.num_neighbors = num_neighbors
    self.reg_alpha = reg_alpha
    self.reg_beta = reg_beta
    self.tol = tol
    self.max_iter = max_iter
    self.mu = mu
    self.max_mu = max_mu
    self.rho = rho

    self._sel_features = None
    self._cluster_labels = None

    def fit(self, X, X_sim, y=None, **kwargs):

        # Calc X_sim from Eq. 3 starting from np.zeros

        nrows, ncols = np.shape(X)

        # As in MATLAB implementation.
        X = np.transpose(X)
        X_sim = np.transpose(X_sim)

        # Setup:
        H = np.eye(nrows) - np.ones((nrows, 1)) * np.ones((1, nrows)) / nrows
        H = H / (nrows - 1)

        Y = np.zeros((ncols, nrows))
        Z = np.zeros((ncols, nrows))
        M = np.zeros((nrows, nrows))
        L = np.zeros((nrows, nrows))

        Lambda1 = np.zeros((ncols, nrows))
        Lambda2 = np.zeros((nrows, nrows))

        iter = 0
        while iter <= max_iter:

            # Update Z:
            temp1 = X - Y - ((1 - self.reg_beta) * Y * H * L * H - Lamda1) / self.mu
            Z = X - self._solve_l20(temp1, (ncols - nrows))

            # Update Y:
            temp1 = Z + ((1 - self.reg_beta) * Z * H * L * H + Lamda1) / self.mu;
            Y = self._solve_l20(temp1, nrows)

            # Update L:
            temp2 = ((1 - self.reg_beta) * self._speed_up(H * Y.T * Z * H) + self.ref_beta * X_sim - Lamda2) / self.mu + M
            L = self._solve_rank_lagrange(self._speed_up(temp2), 2 * self.reg_alpha / self.mu)

            # Check if stop criterion is satisfied:
            leq1 = Z - Y
            leq2 = L - M
            # Infinite norm.
            stop_c1 = np.max(np.max(np.abs(leq1)))
            # Infinite norm.
            stop_c2 = np.max(np.max(np.abs(leq2)))

            if iter == 1 or np.mod(iter, 2) == 0 or (stop_c1 < self.tol and stop_c2 < self.tol):
                print('Iter:', iter, 'mu:', self.mu, 'stop C:', stop_c1, stop_c2)

            if stop_c1 < selftol and stop_c2 < self.tol:
                break

            # Update Lagrange multipliers.
            Lamda1 = Lamda1 + self.mu * leq1
            Lamda2 = Lamda2 + self.mu * leq2
            self.mu = np.min(self.max_mu, self.mu * self.rho)

        # Update iter counter.
        iter = iter + 1

    # Obtain labels
    eig_vecs, eig_vals, _ = np.eig(np.max(L, L.T), self.num_clusters)
    self._cluster_labels = eig_vecs * np.sqrt(eig_vals)
    Label = np.max(np.abs(self._cluster_labels), axis=2)
    self._cluster_labels = self._cluster_labels.T

    return self

    def transform(self, X, y=None, **kwargs):
        pass


def dgufs():
    pass
