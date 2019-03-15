# -*- coding: utf-8 -*-
#
# transforms.py
#

"""
Whitening transforms.
"""

__author__ = 'Severin Langberg'
__contact__ = 'langberg91@gmail.com'


import numpy as np

from scipy import linalg

from sklearn.covariance import LedoitWolf
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.utils import check_array


class Whitening(TransformerMixin, BaseEstimator):
    """Perform a whiten transformation of a data matrix.
    Args:
        center (bool): Perform mean centering of X.
        method (str): Whitening transformation method. Select one of
            * 'zca'
            * 'zca-cor'
            * 'pca',
            * 'pca-cor'
            * 'cholesky'

    """

    NAME = 'Whitening'

    def __init__(self, eps=1e-12, method='zca-cor'):

        self.eps = eps
        self.method = method

        self.W = None

    def __name__(self):

        return self.NAME

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
        elif self.method == 'zca-cor':
            self.W = self._zca_cor(X, Sigma)
        elif self.method == 'pca-cor':
            self.W = self._pca_cor(X, Sigma)
        else:
            raise ValueError('Invalid method {}'.format(self.method))

        return self

    @staticmethod
    def covariance_matrix(X):
        """Construct the covariance matrix depending on the dimensionalilty of
        the feature matrix.

        """
        #nrows, ncols = np.shape(X)
        #if nrows < ncols:
        #    Sigma = np.dot(np.transpose(X), X) / (X.shape[0] - 1)
        #else:
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
        # Wrapper of scikit-learn check array function.
        X = check_array(X)
        return X


if __name__ == '__main__':
    pass
