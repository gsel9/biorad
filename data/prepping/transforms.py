# -*- coding: utf-8 -*-
#
# transforms.py
#
# * For details and NumPy vs. SciPy comparison:
#   https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html
# * TODO: Check requirements/assumptions (e.g. positive definite matrix).
#

"""
The ZCA whitening, or Mahalanobis whitening ensures that the average
covariance between whitened and orginal variables is maximal.

Likewise, ZCA-cor whitening leads to whitened variables that are maximally
correlated on average with the original variables.
"""

__author__ = 'Severin Langberg'
__contact__ = 'langberg91@gmail.com'


import numpy as np

from scipy import linalg

from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator


class Whitening(TransformerMixin, BaseEstimator):
    """Perform a whiten transformation of a data matrix.
    Args:
        center (bool): Perform mean centering of X.
        eps (float): Whitening constant preventing division by zero.
        method (str): Whitening transformation method. Select one of 'zca',
            'zca_cor', 'pca', 'pca_cor', or 'cholesky'.
        corr (bool): Determine to calculate ZCA or ZCA-cor matrix.
    """

    def __init__(self, eps=1e-6, method='zca_cor', z_scoring=True, copy=True):

        self.eps = eps
        self.method = method
        self.z_scoring = z_scoring
        self.copy = copy

        self.W = None

    def fit(self, X, y=None, rowvar=False, **kwargs):
        """
        Args:
            X (array-like): The data matrix.
        Returns:
            (array-like): The transformation matrix.
        """
        X = self._check_X(X)

        # Covariance matrix.
        Sigma = self.covariance_matrix(X)

        if self.method == 'cholesky':
            self.W = self._cholesky(Sigma)
        elif self.method == 'zca':
            self.W = self._zca(Sigma)
        elif self.method == 'pca':
            self.W = self._pca(Sigma)
        elif self.method == 'zca_cor':
            self.W = self._zca_cor(X, Sigma)
        elif self.method == 'pca_cor':
            self.W = self._pca_cor(X, Sigma)
        else:
            raise ValueError('Invalid method {}'.format(self.method))

        return self

    @staticmethod
    def covariance_matrix(X):
        """Construct the covariance matrix depending on the dimensionalilty of
        the feature matrix.

        """
        nrows, ncols = np.shape(X)
        if nrows > ncols:
            Sigma = np.dot(np.transpose(X), X) / X.shape[0]
        else:
            estimator = LedoitWolf()
            Sigma = estimator.fit(X).covariance_

        return Sigma


    def _cholesky(self, Sigma):
        # Singular Value Decomposition.
        U, Lambda, _ = np.linalg.svd(Sigma)
        W = linalg.cholesky(
            np.dot(U, np.dot(np.diag(1.0 / (Lambda + self.eps)), U.T))
        )
        return np.transpose(W)

    def _zca(self, Sigma):
        # Singular Value Decomposition.
        U, Lambda, _ = linalg.svd(Sigma)
        W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + self.eps)), U.T))

        return W

    def _pca(self, Sigma):
        # Singular value decomposition of the co-variance matrix.
        U, Lambda, _ = linalg.svd(Sigma)
        W = np.dot(np.diag(1.0 / np.sqrt(Lambda + self.eps)), U.T)

        return W

    def _zca_cor(self, X, Sigma):

        # The square-root of the diagonal variance matrix.
        V_sqrt = np.diag(np.std(X, axis=0))

        # The correlation matrix.
        P = np.dot(np.dot(linalg.inv(V_sqrt), Sigma), linalg.inv(V_sqrt))

        # Eigendecomposition of the correlation matrix.
        G, Theta, _ = linalg.svd(P)

        # Construct the sphering matrix.
        W = np.dot(
            np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + self.eps)), G.T)),
            V_sqrt
        )
        return W

    def _pca_cor(self, X, Sigma):
        #
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(linalg.inv(V_sqrt), Sigma), linalg.inv(V_sqrt))
        G, Theta, _ = linalg.svd(P)
        W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + self.eps)), G.T), V_sqrt)

        return W

    def transform(self, X, **kwargs):

        X = self._check_X(X)

        return np.dot(X, np.transpose(self.W))

    def _check_X(self, X):
        # Perform type checking and formatting of data matrix.

        if self.copy:
            X = np.copy(X)

        # Apply Z-score transformation.
        if self.z_scoring:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return X


if __name__ == '__main__':
    pass
