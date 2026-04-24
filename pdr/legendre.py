"""
legendre.py
-----------
Multi-index generation and Legendre design matrix construction.
Experiment-agnostic — works for any PDE, any dimension.
"""

import numpy as np
from scipy.special import legendre

def multiidx_gen(N, rule, w, base = 0, multiidx = np.array([]), MULTI_IDX = np.array([])):
    """
    Code translated to python from 
    %----------------------------------------------------
    % Sparse Grid Matlab Kit
    % Copyright (c) 2009-2018 L. Tamellini, F. Nobile
    % See LICENSE.txt for license
    %----------------------------------------------------
    
    % MULTI_IDX = multiidx_gen(N,rule,w,base,[],[])
    %
    % calculates all multi indexes M_I of length N with elements such that rule(M_I) <= w.
    % M_I's are stored as rows of the matrix MULTI_IDX
    % indices will start from base (either 0 or 1)
    
    % multiidx_gen works recursively, exploring in depth the tree of all possible multiindexes. 
    % the current multi-index is passed as 4-th input argument, and eventually stored in MULTI_IDX.
    % The starting point is the empty multiidx: [], and MULTI_IDX is empty at the first call of the function.
    % That's why the call from keyboard comes with [], [] as input argument: multiidx_gen(L,rule,w,[],[])
    """
    
    if len(multiidx) != N:
        # recursive step: generates all possible leaves from the current node (i.e. all multiindexes with length le+1 starting from
        # the current multi_idx, which is of length le that are feasible w.r.t. rule)
        i = base
    
        while rule(np.append(multiidx,i)) <= w:
            # if [multiidx, i] is feasible further explore the branch of the tree that comes out from it.
            MULTI_IDX = multiidx_gen(N,rule,w,base,np.append(multiidx, i),MULTI_IDX)
            i=i+1

    
    else:
        # base step: if the length of the current multi-index is L then I store it in MULTI_IDX  (the check for feasibility was performed in the previous call  

        #MULTI_IDX=np.vstack((MULTI_IDX, multiidx))
        MULTI_IDX = np.vstack([MULTI_IDX, multiidx]) if MULTI_IDX.size else multiidx
        #MULTI_IDX=[MULTI_IDX; multiidx]

    return MULTI_IDX    



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
