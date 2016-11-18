"""
Algorithm to compute the nearest positive semi-definite matrix.
Based on the code at http://stackoverflow.com/a/10940283
"""

import numpy as np


def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearcorr(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk


def cholesky(A, max_tries = 10000):
    """
    Tries to use nearcorr to get the nearest correlation matrix. If the obtained matrix is not positive semi-defined
    (due to numerical errors), adds a tiny multiple of the identity to the matrix and tries again. When a positive
    semi-defined matrix is obtained, return its cholesky decomposition.
    Args:
        A:
        max_tries:

    Returns:

    """
    corrected_corr = nearcorr(A)
    n_tries = 0
    while True:
        try:
            return np.linalg.cholesky(corrected_corr)
        except np.linalg.linalg.LinAlgError:
            eigs = np.linalg.eigvals(corrected_corr)
            min_eig = np.real(min(eigs))
            corrected_corr += (-min_eig * n_tries ** 2 + np.spacing(min_eig)) * np.eye(len(corrected_corr))
            n_tries += 1
            if n_tries < max_tries:
                continue
            raise ZeroDivisionError('Could not generate a valid correlation matrix!')
