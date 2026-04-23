"""
legendre.py
-----------
Multi-index generation and Legendre design matrix construction.
Experiment-agnostic — works for any PDE, any dimension.
"""

import numpy as np
from scipy.special import legendre

def multiidx_gen(dim, rule, max_weight, base=0):
    """
    Generate all multi-indices of length `dim` satisfying `rule(idx) <= max_weight`.

    Uses list accumulation internally (O(N) instead of recursive vstack O(N²)).

    Parameters
    ----------
    dim        : int — number of dimensions
    rule       : callable — admissibility rule, e.g. hyperbolic cross
    max_weight : float — maximum weight
    base       : int — starting index (usually 0)

    Returns
    -------
    np.ndarray, shape (N, dim) — each row is a multi-index
    """
    results = []

    def _recurse(partial):
        if len(partial) == dim:
            results.append(partial.copy())
            return
        i = base
        while True:
            candidate = partial + [i]
            # Pad to full length for rule evaluation
            padded = candidate + [base] * (dim - len(candidate))
            if rule(np.array(padded)) > max_weight:
                break
            _recurse(candidate)
            i += 1

    _recurse([])
    return np.array(results, dtype=int)


def build_design_matrix(X, Lambda):
    """
    Build the Legendre design matrix and sparsity weights.

    Parameters
    ----------
    X      : np.ndarray, shape (m, d) — sample points in [-1,1]^d
    Lambda : np.ndarray, shape (N, d) — multi-index set

    Returns
    -------
    A       : np.ndarray, shape (m, N) — design matrix
    weights : np.ndarray, shape (N,)   — sparsity weights
    """
    m, d = X.shape
    N = Lambda.shape[0]

    A = np.ones((m, N))
    weights = np.ones(N)

    for n in range(N):
        for k in range(d):
            deg = Lambda[n, k]
            poly = legendre(deg)
            scale = np.sqrt(2 * deg + 1)
            A[:, n] *= poly(X[:, k]) * scale
            weights[n] *= scale

    return A, weights


def hyperbolic_cross_rule(x):
    """Standard hyperbolic cross: prod(x_i + 1) - 1."""
    return np.prod(x + 1) - 1
