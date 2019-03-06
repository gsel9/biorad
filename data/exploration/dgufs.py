import numpy as np
import pandas as pd

from scipy import linalg

from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin


def similarity_matrix(X):
    """"""

    S = distance.squareform(distance.pdist(X))
    return -S / np.max(S)


def centering_matrix(nrows):
    """"""

    scaled = (np.ones((1, nrows)) / nrows)
    H = np.eye(nrows) - np.ones((nrows, 1)) * scaled
    # Experimental version where H := H / (n - 1).
    return H / (nrows - 1)

def solve_l20(Q, nfeats):

    # b(i) is the (l2-norm)^2 of the i-th row of Q.
    b = np.sum(Q ** 2, axis=1)[:, np.newaxis]
    idx = np.argsort(b[:, 0])[::-1]

    P = np.zeros(np.shape(Q), dtype=float)
    P[idx[:nfeats], :] = Q[idx[:nfeats], :]

    return P

def speed_up(C):
    """Refer to Simultaneous Clustering and Model Selection (SCAMS),
    CVPR2014.

    """
    diagmask = np.eye(np.shape(C)[0], dtype=bool)
    # Main diagonal = 0.
    C[diagmask] = 0

    # If C is (N x N), then tmp is (N*N x 1).
    tmp = np.reshape(C, (np.size(C), 1))
    # Remove the main diagonal elements of C in tmp. Then tmp has a
    # length of N * (N - 1).
    tmp = np.delete(tmp, np.where(diagmask.ravel()))
    # Scale to [0, 1] range.
    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))

    affmaxo = C
    # affmaxo(~diagmask) is a column vector.
    affmaxo[np.logical_not(diagmask)] = tmp
    C_new = affmaxo

    return C_new


def solve_rank_lagrange(A, eta):
    """"""

    # Guarantee symmetry.
    A = 0.5 * (A + np.transpose(A))
    tempD, tempV = linalg.eig(A)
    # Discard the imaginary part.
    tempV = np.real(tempV)
    tmpD = np.real(tempD)

    tempD = np.real(np.diag(tempD))
    # eta * rank(P)
    tmpD[tmpD <= np.sqrt(eta)] = 0
    tempD = np.diag(tmpD)

    P = tempV.dot(tempD).dot(np.transpose(tempV))

    return P


def solve_l0_binary(Q, gamma):
    """"""

    P = np.copy(Q)
    # Each P_ij is in {0, 1}
    if gamma > 1:
        P[Q > 0.5 * (gamma + 1)] = 1
        P[Q <= 0.5 * (gamma + 1)] = 0
    else:
        P[Q > 1] = 1
        P[Q < np.sqrt(gamma)] = 0

    return P


class DGUFS(BaseEstimator, TransformerMixin):
    """The Dependence Guided Unsupervised Feature Selection (DGUFS) algorithm
    developed by Jun Guo and Wenwu Zhu.

    num_features (int): The number of features to select.
    num_clusters (int):
    alpha (): Regularization parameter from the range (0, 1).
    beta (): Regularization parameter > 0.
    tol (float): Tolerance used to determine optimization convergance. Defaults
        to 10e-6 as suggested in the paper.
    max_iter (): The maximum number of iterations of the
    mu ():
    max_mu ():
    rho ():

    """

    NAME = 'DGUFSSelection'

    def __init__(
        self,
        num_features=2,
        num_clusters=2,
        alpha=0.5,
        beta=0.9,
        tol=1e-10,
        max_iter=5e2,
        mu=1e-6,
        max_mu=1e10,
        rho=1.1
    ):

        self.num_features = num_features
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.mu = mu
        self.max_mu = max_mu
        self.rho = rho

        # NOTE: Attributes set with instance.
        self.S = None
        self.H = None
        self.Y = None
        self.Z = None
        self.M = None
        self.L = None
        self.V = None
        self.Lamda1 = None
        self.Lamda2 = None

    def _setup_matrices(self, nrows, ncols):
        # Setup.
        self.Y = np.zeros((ncols, nrows), dtype=float)
        self.Z = np.zeros((ncols, nrows), dtype=float)

        self.M = np.zeros((nrows, nrows), dtype=float)
        self.L = np.zeros((nrows, nrows), dtype=float)

        self.Lamda1 = np.zeros((ncols, nrows), dtype=float)
        self.Lamda2 = np.zeros((nrows, nrows), dtype=float)

        return self

    def __name__(self):

        return self.NAME

    def _check_X(self, X):
        # Type checking and formatting of feature matrix.
        nrows, ncols = np.shape(X)
        if self.num_features > ncols:
            raise ValueError('Number of features to select exceeds the number '
                             'of columns in X ({})'.format(ncols))
        if nrows < 2:
            raise RuntimeError('Feature selection requires more than two '
                               'samples')
        # NB: From nrows x ncols to ncols x nrows as algorithm given in the
        # paper.
        X_trans = np.transpose(X)

        return X_trans, nrows, ncols

    @property
    def support(self):
        """Returns the column indicators of selected features."""

        # Select features based on where the transformed feature matrix has
        # column sums != 0.
        selected_cols = np.squeeze(np.where(np.sum(self.Y.T, axis=0) != 0))
        # Sanity check.
        assert len(selected_cols) <= self.num_features

        return selected_cols

    @property
    def memberships(self):
        """Return the cluster indicator labels for each obeservation."""

        eigD, eigV = linalg.eig(np.maximum(self.L, np.transpose(self.L)))
        
        # Discard imaginary parts and truncate assuming comps are sorted.
        eigD = np.real(np.diag(eigD)[:self.num_clusters, :self.num_clusters])
        eigV = np.real(eigV[:, :self.num_clusters])
        self.V = np.dot(eigV, np.sqrt(eigD + 1e-12))
        
        # The final cluster labels can be obtained by determining the position
        # of the largest element at each cluster indicator in V.
        return np.argmax(self.V.T, axis=1)

    def fit(self, X, **kwargs):
        """Select features from X.

        Args:
            X (array-like): The feature matrix with shape
                (n samples x n features).

        """
        X_trans, nrows, ncols = self._check_X(X)

        self.S = similarity_matrix(X)
        # Experimental version where H := H / (n - 1).
        self.H = centering_matrix(nrows)

        self._setup_matrices(nrows, ncols)

        i = 1
        while i <= self.max_iter:

            # Alternate optimization of matrices.
            self._update_Z(X_trans, ncols)
            self._update_Y()
            self._update_L()
            self._update_M(nrows)

            # Check if stop criterion is satisfied.
            leq1 = self.Z - self.Y
            leq2 = self.L - self.M
            stopC1 = np.max(np.abs(leq1))
            stopC2 = np.max(np.abs(leq2))
            if (stopC1 < self.tol) and (stopC2 < self.tol):
                i = self.max_iter
            else:
                # Update Lagrange multipliers.
                self.Lamda1 = self.Lamda1 + self.mu * leq1
                self.Lamda2 = self.Lamda2 + self.mu * leq2
                self.mu = min(self.max_mu, self.mu * self.rho);
                # Update counter.
                i = i + 1

        return self

    def _update_Z(self, X, ncols):
        # Updates the Z matrix.
        YHLH = self.Y.dot(self.H).dot(self.L).dot(self.H)
        U = X - self.Y - (((1 - self.beta) * YHLH - self.Lamda1) / self.mu)
        self.Z = X - solve_l20(U, (ncols - self.num_features))

        return self

    def _update_Y(self):
        # Updates the Y matrix.
        ZHLH = self.Z.dot(self.H).dot(self.L).dot(self.H)
        U = self.Z + (((1 - self.beta) * ZHLH + self.Lamda1) / self.mu)
        self.Y = solve_l20(U, self.num_features)

        return self

    def _update_L(self):
        # Updates the L matrix.
        _speed_up = speed_up(
            self.H.dot(np.transpose(self.Y)).dot(self.Z).dot(self.H)
        )
        U = ((1 - self.beta) * _speed_up + self.beta * self.S - self.Lamda2)
        self.L = solve_rank_lagrange(
            speed_up(U / self.mu + self.M), 2 * self.alpha / self.mu
        )
        return self

    def _update_M(self, nrows, gamma=5e-3):
        # Updates the M matrix.
        _M = self.L + self.Lamda2 / self.mu
        _M = solve_l0_binary(_M, 2 * gamma / self.mu)

        self.M = _M - np.diag(np.diag(_M)) + np.eye(nrows)

        return self

    def transform(self, X, **kwargs):
        """Retain selected features from X.

        Args:
            X (array-like): The feature matrix with shape
                (n samples x n features).

        Returns:
            (array-like): The feature matrix containing only the selected
                features.

        """

        if isinstance(X, pd.DataFrame):
            data = X.values
            output = pd.DataFrame(
                data[:, self.support],
                columns=X.columns[self.support],
                index=X.index
            )
        elif isinstance(X, np.ndarray):
            output = X[:, self.support]
        else:
            raise TypeError('Cannot transform data of type {}'.format(type(X)))

        return output
