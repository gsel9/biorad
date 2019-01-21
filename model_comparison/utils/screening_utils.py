#
#
#
#


"""
These implementations are based on the MATLAB code:
https://github.com/eeGuoJun/AAAI2018_DGUFS/blob/master/JunGuo_AAAI_2018_DGUFS_code/files/speedUp.m

"""

# For Hungarian algorithm. See also: https://pypi.org/project/munkres/
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
# For euclidean distance matrix. See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
from sklearn.metrics.pairwise import euclidean_distances


def best_map(L1, L2):
    """Permute labels of L2 match L1 as good as possible.

    """

    if np.size(L1) != np.size(L2):
        raise RuntimeError('Got sizes L1: {} and L2 {}, when should be equal'
                           ''.format(np.size(L1), np.size(L2)))

    Label1 = np.unique(L1); nClass1 = len(Label1)
    Label2 = np.unique(L2); nClass2 = len(Label2)

    nClass = max(nClass1,nClass2)
    G = zeros(nClass)
    for i in range(nClass1):
        for j in range(nClass2)
            G[i, j] = len(np.where(L1 == Label1[i] and L2 == Label2[j]))

    c, t = linear_sum_assignment(-1.0 * G);

    newL2 = np.zeros(np.size(L2))
    for i in range(nClass2):
        newL2[L2 == Label2[i]) = Label1[c[i]]]

    return newL2


def similarity_matrix(X, num_neighbors, gauss_weight=1):
    """Construct a graph in an unsupervised manner."""

    nrows, _ = np.shape(X)

    X = np.transpose(X)


    dist = euclidean_distances(X)
    sigma = 4 * np.mean(np.mean(dist))

    idx = np.argsort(dist, axis=2)
    idx = idx[:, :num_neighbors]

    # TODO:
    G = sparse(repmat([1:nSmp]',[k,1]),idx(:),ones(numel(idx),1),nSmp,nSmp);
    %%% the i_th row of matrix G stores the information of the
    %%% i_th sample's k neighbors. (1: is a neighbor, 0: is not)
    if num ~= 1
        W = (exp(-Dist/sigma)).*G; % Gaussian kernel weight
        W = full(0.5*(W+W')); % guarantee symmetry
    else
        W = G; % 0-1 weight
        W = full(max(W,W')); % guarantee symmetry
    end

    L = full(max(L,L')); % guarantee symmetry


def euclidean_distance_matrix():

    pass


def normalized_mutual_information(y_true, y_pred):
    """Normalized Mutual Information between two clusterings."""

    return normalized_mutual_info_score(y_true, y_pred)


def solve_l0_binary(Q, gamma):
    """
    % solve the following problem:
    % min_P  ||P - Q||_F^2 + gamma*||P||_0
    %  s.t.  each P_ij is in [0,1] or {0,1}
    % gamma <= 1 : [0,1]
    % gamma > 1  : {0,1}
    % P and Q are matrixes.
    """

    P = Q
    if gamma > 1:
        P[Q > 0.5 * (gamma + 1)] = 1
        P[Q <= 0.5 * (gamma + 1)] = 0
    else:
        P[Q > 1] = 1
        P[Q < np.sqrt(gamma)] = 0

    return P


def solve_l20(Q, m):
    """
    % solve the following problem:
    % min_P  ||P - Q||_F^2
    %  s.t.  ||P||_2,0 <= m
    """
    b = np.sum(Q ** 2, axis=2)
    P = np.zeros(np.size(Q))

    idx = np.argsort(b)[::-1]
    P[idx[:m], :] = Q[idx[:m], :]

    return P


def solve_rank_lagrange(A, eta):
    """
    % solve the following problem:
    % min_P  ||P - A||_F^2 + eta*rank(P)
    %  s.t.  P is symmetric and positive semi-definite
    """
    A = 0.5 * (A + A.T)
    eig_vecs, eig_vals, _ = np.eig(A)
    tmpD = np.diag(eig_vals)
    tmpD[eig_vals <= np.sqrt(eta)] = 0
    tempD = np.diag(eig_vals)
    P = eig_vecs * eig_vals * eig_vecs.T

    return P


def speed_up(C):
    # Refer to SCAMS: Simultaneous Clustering and Model Selection, CVPR2014
    diagmask = np.logical(np.eye(np.size(C, axis=1)))
    C[diagmask] = 0

    tmp = C[:]
    tmp[diagmask[:]] = []
    # Mix-max scaling?
    tmp = (tmp - np.min(tmp)) / (np.max(tmp - np.min(tmp)))

    affmaxo = C
    affmaxo[not diagmask] = tmp
    Cnew = affmaxo

    return Cnew
